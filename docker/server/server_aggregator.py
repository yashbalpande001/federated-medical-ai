import os
import sys
import json
import time
import io
import copy
import hashlib
from pathlib import Path
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

app = FastAPI(title="Federated Medical AI Real Aggregation Server")

PROJECT_ROOT = Path("/app") if os.path.exists("/app") else Path(".")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data"
TEST_SET_PATH = DATA_DIR / "test_set_900.pt"

if not TEST_SET_PATH.exists():
    alt_path = Path("data/test_set_900.pt")
    if alt_path.exists():
        TEST_SET_PATH = alt_path

if not TEST_SET_PATH.exists():
    print("!" * 80)
    print("CRITICAL SERVER STARTUP ERROR:")
    print(f"  Pre-extracted test set file NOT found at: {TEST_SET_PATH}")
    print("  ACTION REQUIRED: Export test_set_900.pt using kaggle_notebooks/export_test_set.ipynb")
    print("                   and place it at data/test_set_900.pt before starting the server.")
    print("!" * 80)
    sys.exit(1)

print(f"[OK] [SERVER STARTUP] Discovered test set at: {TEST_SET_PATH}")

EXPECTED_CLIENTS = 2
received_submissions = {}

def build_resnet18():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )
    return model

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "expected_clients": EXPECTED_CLIENTS,
        "received_count": len(received_submissions),
        "clients_received": list(received_submissions.keys())
    }

@app.post("/submit")
async def submit_weights(
    request: Request,
    client_id: str = Form(...),
    metadata_json: str = Form(...),
    weights_file: UploadFile = File(...)
):
    client_ip = request.client.host if request.client else "unknown"
    metadata = json.loads(metadata_json)
    contents = await weights_file.read()

    byte_count = len(contents)
    sha256_hash = hashlib.sha256(contents).hexdigest()

    n_samples = metadata.get("partition_size", 0)
    pos_pct = metadata.get("positive_percentage", 0.0)

    print()
    print(f"[SERVER LOG] Submission Received:")
    print(f"  - Source IP       : {client_ip}")
    print(f"  - Client ID       : {client_id}")
    print(f"  - Sample Count    : {n_samples}")
    print(f"  - Positive Rate   : {pos_pct:.2f}%")
    print(f"  - Byte Count      : {byte_count} bytes")
    print(f"  - SHA256 Digest   : {sha256_hash}")

    for prev_id, prev_sub in received_submissions.items():
        if prev_sub["sha256"] == sha256_hash:
            err_msg = (
                f"ABORT CRITICAL ERROR: Client {client_id} submitted identical weight bytes as Client {prev_id} "
                f"(SHA256: {sha256_hash}). Identical submission payloads are strictly prohibited!"
            )
            print(f"[SERVER ABORT] {err_msg}")
            raise HTTPException(status_code=400, detail=err_msg)

    buf = io.BytesIO(contents)
    weights_state_dict = torch.load(buf, map_location="cpu")

    received_submissions[client_id] = {
        "client_id": client_id,
        "source_ip": client_ip,
        "sha256": sha256_hash,
        "byte_count": byte_count,
        "n_samples": n_samples,
        "positive_rate": pos_pct,
        "metadata": metadata,
        "weights": weights_state_dict
    }

    print(f"[SERVER] Validated weight submission from Client {client_id} ({client_ip}). Progress: {len(received_submissions)}/{EXPECTED_CLIENTS}")

    if len(received_submissions) < EXPECTED_CLIENTS:
        return {
            "status": "received",
            "message": f"Client {client_id} submission received. Waiting for {EXPECTED_CLIENTS - len(received_submissions)} more client(s).",
            "received_count": len(received_submissions)
        }
    else:
        print("=" * 85)
        print("  ALL 2 CLIENT WEIGHT SUBMISSIONS RECEIVED! EXECUTING REAL FEDAVG AGGREGATION")
        print("=" * 85)

        global_weights = run_weighted_fedavg(received_submissions)

        metrics = evaluate_global_model_on_test_pt(global_weights)

        results_payload = {
            "rounds_executed": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clients": [
                {
                    "client_id": sub["client_id"],
                    "source_ip": sub["source_ip"],
                    "sha256": sub["sha256"],
                    "byte_count": sub["byte_count"],
                    "n_samples": sub["n_samples"],
                    "positive_rate": sub["positive_rate"],
                    "metadata": sub["metadata"]
                }
                for sub in received_submissions.values()
            ],
            "metrics": metrics
        }

        out_json = OUTPUT_DIR / "round1_results.json"
        with open(out_json, "w") as f:
            json.dump(results_payload, f, indent=4)

        print(f"[SERVER] Exported round results to: {out_json}")
        print(f"[SERVER] Aggregation complete. New global AUC: {metrics['auc']:.4f}, Recall: {metrics['recall']:.4f}.")

        return {
            "status": "aggregation_complete",
            "message": "Real FedAvg aggregation completed successfully across 2 real client compute origins.",
            "metrics": metrics
        }

def run_weighted_fedavg(submissions):
    clients = list(submissions.values())
    total_samples = sum(c["n_samples"] for c in clients)

    global_weights = copy.deepcopy(clients[0]["weights"])

    for key in global_weights.keys():
        if not torch.is_floating_point(global_weights[key]):
            global_weights[key] = clients[0]["weights"][key]
        else:
            global_weights[key] = torch.zeros_like(global_weights[key])
            for c in clients:
                factor = c["n_samples"] / float(total_samples)
                global_weights[key] += c["weights"][key] * factor

    return global_weights

def evaluate_global_model_on_test_pt(global_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18().to(device)
    model.load_state_dict(global_weights)
    model.eval()

    test_payload = torch.load(TEST_SET_PATH, map_location="cpu")

    imgs_uint8 = test_payload['images']
    labels_tensor = test_payload['labels']

    assert len(imgs_uint8) == 900, f"Expected 900 images, got {len(imgs_uint8)}"
    assert labels_tensor.sum().item() == 208, f"Expected 208 positives, got {labels_tensor.sum().item()}"

    imgs_float = imgs_uint8.float() / 255.0
    imgs_3ch = imgs_float.unsqueeze(1).repeat(1, 3, 1, 1)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    batch_size = 32
    all_targets = labels_tensor.numpy().tolist()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(imgs_3ch), batch_size):
            batch_imgs = imgs_3ch[i:i+batch_size]
            batch_norm = torch.stack([norm(img) for img in batch_imgs]).to(device)
            logits = model(batch_norm).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    auc = float(roc_auc_score(all_targets, all_probs))
    preds = (np.array(all_probs) >= 0.5).astype(int)
    f1 = float(f1_score(all_targets, preds, zero_division=0))
    prec = float(precision_score(all_targets, preds, zero_division=0))
    rec = float(recall_score(all_targets, preds, zero_division=0))
    cm = confusion_matrix(all_targets, preds)
    spec = float(cm[0,0] / max(1, cm[0,0] + cm[0,1]))

    print()
    print("=" * 80)
    print("     EVALUATION ON PRE-EXTRACTED TEST SET (test_set_900.pt)")
    print("=" * 80)
    print(f"  - Aggregated Global Test ROC-AUC    : {auc:.4f}")
    print(f"  - Aggregated Global Test Recall     : {rec:.4f}")
    print(f"  - Aggregated Global Test Specificity: {spec:.4f}")
    print(f"  - Aggregated Global Test Precision  : {prec:.4f}")
    print(f"  - Aggregated Global Test F1-Score   : {f1:.4f}")
    print("  - Confusion Matrix:")
    print(cm)
    print("=" * 80)

    return {
        "auc": round(auc, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
