import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import io
import copy
import hashlib
from pathlib import Path
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

app = FastAPI(title="Federated Medical AI Real Aggregation Server")

PROJECT_ROOT = Path("/app") if os.path.exists("/app/server_aggregator.py") else Path(".")
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

TOP_LEVEL_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

HISTORY_PATH = OUTPUT_DIR / "round_history.json"
ROUND1_RESULTS_PATH = OUTPUT_DIR / "round1_results.json"

def build_resnet18():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )
    return model

def find_baseline_checkpoint():
    candidates = [
        PROJECT_ROOT / "research" / "docs" / "step_outputs" / "step5_outputs" / "kaggle" / "working" / "outputs" / "checkpoints" / "best_baseline_model.pt",
        Path("research/docs/step_outputs/step5_outputs/kaggle/working/outputs/checkpoints/best_baseline_model.pt"),
        OUTPUT_DIR / "checkpoints" / "best_baseline_model.pt",
        PROJECT_ROOT / "outputs" / "checkpoints" / "best_baseline_model.pt",
        Path("outputs/checkpoints/best_baseline_model.pt"),
        Path("outputs/outputs_bundle/checkpoints/best_baseline_model.pt"),
        Path("weights/best_baseline_model.pt")
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_baseline_weights():
    bp = find_baseline_checkpoint()
    if bp and bp.exists():
        raw = torch.load(bp, map_location="cpu")
        if isinstance(raw, dict) and "model_state_dict" in raw:
            raw = raw["model_state_dict"]
        clean = {}
        for k, v in raw.items():
            ck = k.replace("resnet.", "") if k.startswith("resnet.") else k
            clean[ck] = v
        return clean
    return None

def compute_weight_diff_norms(new_weights, prev_weights):
    """
    Computes:
    - total_l2_norm: L2 norm of (new_weights - prev_weights) across parameters.
    - per_layer_l2_norm: broken out by top-level module (conv1, bn1, layer1, layer2, layer3, layer4, fc).
    """
    if prev_weights is None:
        return 0.0, {m: 0.0 for m in TOP_LEVEL_MODULES}

    def clean_key(k):
        if k.startswith("resnet."):
            return k[len("resnet."):]
        if k.startswith("module."):
            return k[len("module."):]
        return k

    new_clean = {clean_key(k): v for k, v in new_weights.items()}
    prev_clean = {clean_key(k): v for k, v in prev_weights.items()}

    per_layer_sq = {m: 0.0 for m in TOP_LEVEL_MODULES}
    total_sq = 0.0

    for key, new_tensor in new_clean.items():
        if not torch.is_floating_point(new_tensor):
            continue
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        if key not in prev_clean:
            # Handle potential key discrepancies between nn.Sequential head (fc.1.*) and nn.Linear head (fc.*)
            alt_key = None
            if key == "fc.1.weight" and "fc.weight" in prev_clean:
                alt_key = "fc.weight"
            elif key == "fc.1.bias" and "fc.bias" in prev_clean:
                alt_key = "fc.bias"
            elif key == "fc.weight" and "fc.1.weight" in prev_clean:
                alt_key = "fc.1.weight"
            elif key == "fc.bias" and "fc.1.bias" in prev_clean:
                alt_key = "fc.1.bias"
            if alt_key and prev_clean[alt_key].shape == new_tensor.shape:
                prev_tensor = prev_clean[alt_key].to(new_tensor.device)
            else:
                continue
        else:
            prev_tensor = prev_clean[key].to(new_tensor.device)

        diff = (new_tensor - prev_tensor).float()
        sq = torch.sum(diff ** 2).item()
        total_sq += sq

        top_mod = key.split('.')[0]
        if top_mod in per_layer_sq:
            per_layer_sq[top_mod] += sq
        else:
            for m in TOP_LEVEL_MODULES:
                if top_mod.startswith(m):
                    per_layer_sq[m] += sq
                    break

    tot_l2 = float(np.sqrt(total_sq))
    per_layer_l2 = {m: round(float(np.sqrt(s)), 6) for m, s in per_layer_sq.items()}
    return round(tot_l2, 6), per_layer_l2

def init_history():
    history = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r") as f:
                d = json.load(f)
                if isinstance(d, dict):
                    if "rounds" in d and isinstance(d["rounds"], list):
                        history = d["rounds"]
                    elif "by_round" in d and isinstance(d["by_round"], dict):
                        history = list(d["by_round"].values())
                elif isinstance(d, list):
                    history = d
                # Ensure is_test_submission flag exists on all loaded rounds
                for r in history:
                    if "is_test_submission" not in r:
                        r["is_test_submission"] = False
                print(f"[OK] Loaded {len(history)} round(s) from {HISTORY_PATH}")
        except Exception as e:
            print(f"[WARNING] Could not read {HISTORY_PATH}: {e}")

    if not history and ROUND1_RESULTS_PATH.exists():
        try:
            with open(ROUND1_RESULTS_PATH, "r") as f:
                r1 = json.load(f)

            base_w = load_baseline_weights()
            r1_w_path = OUTPUT_DIR / "global_weights_round_1.pt"
            if r1_w_path.exists():
                r1_w = torch.load(r1_w_path, map_location="cpu")
            else:
                cand_p = Path("research/docs/step_outputs/step11_outputs/kaggle/working/outputs/checkpoints/best_fedprox_model_mu_0.1.pt")
                if cand_p.exists():
                    r1_w = torch.load(cand_p, map_location="cpu")
                    torch.save(r1_w, r1_w_path)
                else:
                    r1_w = base_w

            tot_l2, per_layer_l2 = compute_weight_diff_norms(r1_w, base_w)

            submitted_lrs = []
            for c in r1.get("clients", []):
                meta = c.get("metadata", {})
                if "learning_rate" in meta:
                    submitted_lrs.append(meta["learning_rate"])
                elif "lr" in meta:
                    submitted_lrs.append(meta["lr"])
            lr_val = submitted_lrs[0] if submitted_lrs else 0.0001

            r1_entry = {
                "round": 1,
                "rounds_executed": 1,
                "timestamp": r1.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                "auc": r1.get("auc", 0.8101),
                "recall": r1.get("recall", 0.9856),
                "precision": r1.get("precision", 0.2916),
                "specificity": r1.get("specificity", 0.2803),
                "f1": r1.get("f1", 0.4501),
                "confusion_matrix": r1.get("confusion_matrix", [[194, 498], [3, 205]]),
                "learning_rate": lr_val,
                "total_weight_l2_norm": tot_l2,
                "per_layer_l2_norm": per_layer_l2,
                "clients": r1.get("clients", []),
                "metrics": r1.get("metrics", {
                    "auc": r1.get("auc", 0.8101),
                    "recall": r1.get("recall", 0.9856),
                    "precision": r1.get("precision", 0.2916),
                    "specificity": r1.get("specificity", 0.2803),
                    "f1": r1.get("f1", 0.4501),
                    "confusion_matrix": r1.get("confusion_matrix", [[194, 498], [3, 205]])
                })
            }
            history = [r1_entry]
            by_round = {"1": r1_entry}
            with open(HISTORY_PATH, "w") as f:
                json.dump({"rounds": history, "by_round": by_round}, f, indent=4)
            print(f"[OK] Initialized Round 1 history from {ROUND1_RESULTS_PATH} (L2 norm vs baseline: {tot_l2:.6f})")
        except Exception as e:
            print(f"[WARNING] Error backfilling Round 1: {e}")

    return history

round_history = init_history()

def get_previous_global_weights():
    if round_history:
        latest_round_num = round_history[-1].get("round", len(round_history))
        w_path = OUTPUT_DIR / f"global_weights_round_{latest_round_num}.pt"
        if w_path.exists():
            return torch.load(w_path, map_location="cpu")
        latest_path = OUTPUT_DIR / "latest_global_weights.pt"
        if latest_path.exists():
            return torch.load(latest_path, map_location="cpu")
    return load_baseline_weights()

@app.get("/health")
def health_check():
    submissions_list = []
    for sub in received_submissions.values():
        cid = sub["client_id"]
        try:
            cid_int = int(str(cid).replace("client_", "").replace("client", ""))
        except ValueError:
            cid_int = cid

        meta = sub.get("metadata", {})
        partition = meta.get("partition_index", meta.get("partition_idx", meta.get("partition", sub.get("partition", 0))))

        pos_val = sub.get("positive_rate", meta.get("positive_percentage", 0.0))
        if float(pos_val) > 1.0:
            pos_rate = float(pos_val) / 100.0
        else:
            pos_rate = float(pos_val)

        submissions_list.append({
            "client_id": cid_int,
            "source_ip": sub["source_ip"],
            "partition": int(partition),
            "n_samples": int(sub["n_samples"]),
            "positive_rate": round(pos_rate, 3),
            "bytes": int(sub["byte_count"]),
            "sha256": sub["sha256"]
        })

    return {
        "status": "online",
        "current_round": len(round_history) + 1,
        "total_rounds_completed": len(round_history),
        "expected_clients": EXPECTED_CLIENTS,
        "submissions_received": len(received_submissions),
        "submissions": submissions_list
    }

@app.get("/results")
def get_results():
    if not round_history:
        results_path = OUTPUT_DIR / "round1_results.json"
        if not results_path.exists():
            results_path_alt = Path("outputs/round1_results.json")
            if results_path_alt.exists():
                results_path = results_path_alt
            else:
                raise HTTPException(
                    status_code=404,
                    detail="round1_results.json not found. No aggregation has been performed yet."
                )
        with open(results_path, "r") as f:
            data = json.load(f)
        return data

    latest = round_history[-1]
    res = dict(latest)
    res["rounds"] = round_history
    res["history"] = round_history
    return res

@app.get("/history")
def get_history():
    by_round = {str(r.get("round", idx + 1)): r for idx, r in enumerate(round_history)}
    return {
        "total_rounds": len(round_history),
        "rounds": round_history,
        "by_round": by_round
    }

@app.post("/submit")
async def submit_weights(
    request: Request,
    client_id: str = Form(...),
    metadata_json: str = Form(...),
    weights_file: UploadFile = File(...),
    is_test_submission: str = Form("false")
):
    global round_history
    client_ip = request.client.host if request.client else "unknown"
    metadata = json.loads(metadata_json)
    contents = await weights_file.read()

    # Explicit is_test_submission flag checking (Form field, query param, or metadata payload)
    is_test = False
    if str(is_test_submission).strip().lower() in ("true", "1", "yes"):
        is_test = True
    elif request.query_params.get("test", "").lower() in ("true", "1", "yes"):
        is_test = True
    elif metadata.get("is_test_submission") in (True, "true", "True", 1) or metadata.get("is_test") in (True, "true", "True", 1):
        is_test = True

    byte_count = len(contents)
    sha256_hash = hashlib.sha256(contents).hexdigest()

    n_samples = metadata.get("partition_size", metadata.get("n_samples", 0))
    pos_pct = metadata.get("positive_percentage", metadata.get("positive_rate", 0.0))

    current_round_num = len(round_history) + 1
    print()
    print(f"[SERVER LOG] Submission Received for Round {current_round_num}{' (TEST SUBMISSION)' if is_test else ''}:")
    print(f"  - Source IP       : {client_ip}")
    print(f"  - Client ID       : {client_id}")
    print(f"  - Sample Count    : {n_samples}")
    print(f"  - Positive Rate   : {pos_pct:.2f}%")
    print(f"  - Byte Count      : {byte_count} bytes")
    print(f"  - SHA256 Digest   : {sha256_hash}")
    print(f"  - Is Test Sub     : {is_test}")

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
        "is_test_submission": is_test,
        "metadata": metadata,
        "weights": weights_state_dict
    }

    print(f"[SERVER] Validated weight submission from Client {client_id} ({client_ip}). Progress: {len(received_submissions)}/{EXPECTED_CLIENTS}")

    if len(received_submissions) < EXPECTED_CLIENTS:
        return {
            "status": "received",
            "message": f"Client {client_id} submission received for Round {current_round_num}. Waiting for {EXPECTED_CLIENTS - len(received_submissions)} more client(s).",
            "received_count": len(received_submissions),
            "expected_clients": EXPECTED_CLIENTS,
            "current_round": current_round_num,
            "is_test_submission": is_test
        }
    else:
        is_round_test = any(sub.get("is_test_submission", False) for sub in received_submissions.values())
        print("=" * 85)
        print(f"  ALL {EXPECTED_CLIENTS} CLIENT WEIGHT SUBMISSIONS RECEIVED! EXECUTING REAL FEDAVG (ROUND {current_round_num}{' - TEST' if is_round_test else ''})")
        print("=" * 85)

        global_weights = run_weighted_fedavg(received_submissions)
        metrics = evaluate_global_model_on_test_pt(global_weights)

        prev_weights = get_previous_global_weights()
        tot_l2, per_layer_l2 = compute_weight_diff_norms(global_weights, prev_weights)

        # Read actual learning rate submitted by clients
        submitted_lrs = []
        for sub in received_submissions.values():
            meta = sub.get("metadata", {})
            if "learning_rate" in meta:
                submitted_lrs.append(meta["learning_rate"])
            elif "lr" in meta:
                submitted_lrs.append(meta["lr"])
        round_lr = submitted_lrs[0] if submitted_lrs else 0.0001

        round_payload = {
            "round": current_round_num,
            "rounds_executed": current_round_num,
            "is_test_submission": is_round_test,
            "auc": metrics["auc"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "learning_rate": round_lr,
            "total_weight_l2_norm": tot_l2,
            "per_layer_l2_norm": per_layer_l2,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clients": [
                {
                    "client_id": sub["client_id"],
                    "source_ip": sub["source_ip"],
                    "sha256": sub["sha256"],
                    "byte_count": sub["byte_count"],
                    "n_samples": sub["n_samples"],
                    "positive_rate": sub["positive_rate"],
                    "is_test_submission": sub.get("is_test_submission", False),
                    "metadata": sub["metadata"]
                }
                for sub in received_submissions.values()
            ],
            "metrics": metrics
        }

        # Save round weights
        torch.save(global_weights, OUTPUT_DIR / f"global_weights_round_{current_round_num}.pt")
        torch.save(global_weights, OUTPUT_DIR / "latest_global_weights.pt")

        # Accumulate history
        round_history.append(round_payload)
        by_round = {str(r.get("round", idx + 1)): r for idx, r in enumerate(round_history)}
        with open(HISTORY_PATH, "w") as f:
            json.dump({"rounds": round_history, "by_round": by_round}, f, indent=4)

        # Maintain round1_results.json for backward compatibility (only for live rounds or round 1)
        if not is_round_test or not ROUND1_RESULTS_PATH.exists():
            with open(ROUND1_RESULTS_PATH, "w") as f:
                json.dump(round_payload, f, indent=4)

        # Clear submissions for the next round
        received_submissions.clear()

        print(f"[SERVER] Exported round {current_round_num} results to {HISTORY_PATH} (is_test_submission={is_round_test})")
        print(f"[SERVER] Total weight-update L2 norm: {tot_l2:.6f}")
        print(f"[SERVER] Aggregation complete. New global AUC: {metrics['auc']:.4f}, Recall: {metrics['recall']:.4f}.")

        return {
            "status": "aggregation_complete",
            "round": current_round_num,
            "is_test_submission": is_round_test,
            "message": f"Real FedAvg aggregation completed successfully for Round {current_round_num}{' (Test Submission)' if is_round_test else ''}.",
            "metrics": metrics,
            "total_weight_l2_norm": tot_l2,
            "per_layer_l2_norm": per_layer_l2,
            "learning_rate": round_lr
        }

# Register StaticFiles catch-all mount AFTER all API routes
static_path = "static" if os.path.exists("static") else str(PROJECT_ROOT / "static")
if not os.path.exists(static_path):
    alt_static = Path("docker/server/static")
    if alt_static.exists():
        static_path = str(alt_static)
    else:
        os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="ui")

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

    # Clean keys if prefixed with resnet.
    clean_w = {}
    for k, v in global_weights.items():
        ck = k.replace("resnet.", "") if k.startswith("resnet.") else k
        clean_w[ck] = v
    model.load_state_dict(clean_w)
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
