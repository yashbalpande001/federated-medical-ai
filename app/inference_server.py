"""
FastAPI Medical Inference Server (Port 8090).

Provides standalone single-image inference for Pneumonia screening (RSNA Chest X-Ray).
Reuses the exact Step 5 RSNADICOMDataset preprocessing pipeline (resize to 224x224, 
channel expansion to 3ch, ImageNet normalization).
Enforces strict checkpoint validation on startup (fail loudly if missing).
Logs each request metadata to outputs/inference_log.jsonl for audit compliance.
Patient images are processed strictly in-memory and NEVER written to disk.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

from PIL import Image
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rsna_resnet import RSNABaselineResNet18

DISCLAIMER_TEXT = "Research prototype. Not a diagnostic device. Not for clinical use."
OUTPUT_LOG_PATH = PROJECT_ROOT / "outputs" / "inference_log.jsonl"
OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CHECKPOINT RESOLUTION & LOADING
# ---------------------------------------------------------------------------
# DECISION RATIONALE:
# We explicitly default to the Step 5 Centralized Baseline model (best_baseline_model.pt
# or best_model.pt, verified Test ROC-AUC: 0.8536).
# We explicitly DO NOT use the Step 12b live aggregated federated weights because that
# model has only ever seen a single round of training on two positive-skewed partitions
# (57.0% and 29.8% positive rates), and its out-of-distribution generalization has not
# been verified. The Step 5 baseline represents the verified clinical upper bound.
# ---------------------------------------------------------------------------

def resolve_checkpoint_path() -> Path:
    """
    Resolves the model checkpoint path from environment variable or explicit default.
    Fails loudly with FileNotFoundError if the designated checkpoint does not exist.
    """
    env_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if env_path:
        ckpt_path = Path(env_path)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            err_msg = (
                f"CRITICAL ERROR: MODEL_CHECKPOINT_PATH is set to '{env_path}' "
                f"but file does not exist at '{ckpt_path}'. "
                "Failing loudly: will NOT initialize server with random weights."
            )
            print(f"[FATAL] {err_msg}", file=sys.stderr)
            raise FileNotFoundError(err_msg)
        return ckpt_path

    # Default checkpoint search: prioritize best_baseline_model.pt, fallback to best_model.pt
    default_candidates = [
        PROJECT_ROOT / "outputs" / "checkpoints" / "best_baseline_model.pt",
        PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt",
    ]
    for cand in default_candidates:
        if cand.exists():
            return cand

    err_msg = (
        "CRITICAL ERROR: No checkpoint specified via MODEL_CHECKPOINT_PATH and default "
        "Step 5 baseline checkpoint was not found in outputs/checkpoints/ "
        "(looked for 'best_baseline_model.pt' and 'best_model.pt'). "
        "Failing loudly: will NOT initialize server with random weights."
    )
    print(f"[FATAL] {err_msg}", file=sys.stderr)
    raise FileNotFoundError(err_msg)


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[RSNABaselineResNet18, str]:
    """
    Instantiates RSNABaselineResNet18 and strictly loads checkpoint weights.
    """
    print(f"[*] Loading model weights from: {checkpoint_path}")
    model = RSNABaselineResNet18(pretrained=False, freeze_backbone=False, num_classes=1)
    
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    elif isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data
    else:
        raise ValueError(f"Unrecognized checkpoint data format in {checkpoint_path}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"[OK] Model successfully loaded and set to eval mode on device: {device}")
    return model, checkpoint_path.name


# Determine compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model at module level (or during lifespan/startup) - will fail loudly if missing
loaded_checkpoint_path = resolve_checkpoint_path()
inference_model, checkpoint_filename = load_model(loaded_checkpoint_path, device)


# ---------------------------------------------------------------------------
# STEP 5 IDENTICAL PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------

def preprocess_image_bytes(
    file_bytes: bytes,
    filename: str,
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocesses uploaded image bytes identically to RSNADataset._load_image (Step 5):
      1. DICOM (.dcm): pydicom parse, pixel_array min-max rescale to [0, 1].
         Handles MONOCHROME1 inversion if present.
      2. Standard image (.png, .jpg, .jpeg): PIL parse, grayscale convert, / 255.0 to [0, 1].
      3. Interpolation: Bilinear resize to target_size (224, 224).
      4. Channel expansion: Repeat single channel to 3 channels [3, H, W].
      5. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
      6. Batch dimension: returns tensor [1, 3, 224, 224].
    """
    fname_lower = filename.lower()
    is_dcm = fname_lower.endswith(".dcm") or file_bytes.startswith(b"\x00" * 128 + b"DICM") or file_bytes.startswith(b"DICM")

    if is_dcm:
        try:
            import pydicom
            dcm = pydicom.dcmread(io.BytesIO(file_bytes))
            img_arr = dcm.pixel_array.astype(np.float32)

            # Rescale DICOM values to [0, 1] matching RSNADataset._load_image
            img_min, img_max = float(img_arr.min()), float(img_arr.max())
            if img_max > img_min:
                img_arr = (img_arr - img_min) / (img_max - img_min)
            else:
                img_arr = np.zeros_like(img_arr)

            # Check PhotometricInterpretation (MONOCHROME1 requires inversion)
            photo_mode = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).upper().strip()
            if photo_mode == "MONOCHROME1":
                img_arr = 1.0 - img_arr

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupt or invalid DICOM file payload: {str(e)}"
            )
    else:
        try:
            img_pil = Image.open(io.BytesIO(file_bytes)).convert("L")
            img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupt or unreadable image file (must be PNG, JPG, or DICOM): {str(e)}"
            )

    # Convert to 4D tensor for interpolation [1, 1, H, W]
    tensor_2d = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)
    if (tensor_2d.shape[2], tensor_2d.shape[3]) != target_size:
        tensor_2d = torch.nn.functional.interpolate(
            tensor_2d, size=target_size, mode="bilinear", align_corners=False
        )

    # Squeeze batch, repeat to 3 channels [3, H, W]
    tensor_3ch = tensor_2d.squeeze(0).repeat(3, 1, 1)

    # Standard ImageNet normalization matching Step 5
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (tensor_3ch - mean) / std

    # Add batch dimension [1, 3, 224, 224]
    return normalized.unsqueeze(0)


# ---------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Federated Medical AI Inference Service",
    description="Pneumonia diagnostic screening inference service serving ResNet-18 (Step 5 baseline).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for Doctor UI
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, summary="Doctor UI Web Application")
async def serve_doctor_ui():
    """Serves the single-page Doctor UI web application."""
    ui_path = STATIC_DIR / "doctor_ui.html"
    if not ui_path.exists():
        return HTMLResponse(
            content="<h2>Doctor UI not found in app/static/doctor_ui.html</h2>",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    with open(ui_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health", summary="Service Health & Checkpoint Status")
async def health():
    """Returns service health status and loaded model checkpoint information."""
    return {
        "status": "healthy",
        "port": 8090,
        "device": str(device),
        "model_checkpoint": checkpoint_filename,
        "checkpoint_path": str(loaded_checkpoint_path),
        "threshold": 0.5,
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.post("/predict", summary="Pneumonia Screening Inference")
async def predict(file: UploadFile = File(...)):
    """
    Accepts one image file (jpg/png/dcm).
    Preprocesses identically to RSNADataset (Step 5).
    Runs forward inference and returns prediction JSON.
    Logs request metadata to outputs/inference_log.jsonl.
    Zero patient images are stored on disk.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided in upload.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    # Preprocess image bytes
    input_tensor = preprocess_image_bytes(contents, file.filename)
    input_tensor = input_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        logits = inference_model(input_tensor)
        prob = float(torch.sigmoid(logits).item())

    # Decision logic (standard 0.5 decision threshold)
    threshold = 0.5
    prediction_label = "pneumonia_suspected" if prob >= threshold else "normal"

    response_payload = {
        "pneumonia_probability": round(prob, 4),
        "threshold": threshold,
        "prediction": prediction_label,
        "model_checkpoint": checkpoint_filename,
        "disclaimer": DISCLAIMER_TEXT,
    }

    # Audit logging: append to outputs/inference_log.jsonl
    # Note: RAW PATIENT IMAGE BYTES ARE NEVER WRITTEN TO DISK.
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": file.filename,
        "file_size_bytes": len(contents),
        "output": response_payload,
    }

    try:
        with open(OUTPUT_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    except Exception as log_err:
        print(f"[WARN] Failed writing to inference log: {log_err}", file=sys.stderr)

    return JSONResponse(content=response_payload)


if __name__ == "__main__":
    print(f"[*] Starting Medical AI Inference Service on http://0.0.0.0:8090 ...")
    uvicorn.run(app, host="0.0.0.0", port=8090)
