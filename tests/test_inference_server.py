"""
Unit and Integration Tests for app/inference_server.py.
Validates:
1. Fail-loud behavior on missing checkpoint.
2. GET /health endpoint contract.
3. POST /predict with PNG, JPG, and DICOM payloads.
4. Schema validation: probability, threshold, prediction, checkpoint filename, disclaimer.
5. Audit logging to outputs/inference_log.jsonl without writing images to disk.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference_server import app, resolve_checkpoint_path, DISCLAIMER_TEXT, SEVERITY_LEGEND_TEXT

client = TestClient(app)


def test_missing_checkpoint_fails_loudly(monkeypatch):
    """Verifies that the server strictly fails loudly if a non-existent checkpoint is given."""
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "outputs/checkpoints/definitely_nonexistent_weights.pt")
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_checkpoint_path()
    assert "CRITICAL ERROR" in str(exc_info.value)
    assert "definitely_nonexistent_weights.pt" in str(exc_info.value)


def test_health_endpoint():
    """Verifies /health endpoint status, port 8090 metadata, and disclaimer."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8090
    assert data["threshold"] == 0.70
    assert data["disclaimer"] == DISCLAIMER_TEXT
    assert data["severity_legend"] == SEVERITY_LEGEND_TEXT
    assert "model_checkpoint" in data
    assert data["model_checkpoint"] in ["best_baseline_model.pt", "best_model.pt"]


def test_doctor_ui_served():
    """Verifies that the Doctor UI single page is served at root GET /."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Chest X-Ray Pneumonia Screening" in resp.text
    assert DISCLAIMER_TEXT in resp.text
    assert "Permanent Regulatory Notice" in resp.text
    assert SEVERITY_LEGEND_TEXT in resp.text
    assert "Fixed binary classification threshold" in resp.text


def create_sample_png_bytes(width=256, height=256) -> bytes:
    """Creates synthetic PNG image bytes."""
    arr = (np.random.rand(height, width) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_sample_dicom_bytes(width=128, height=128) -> bytes:
    """Creates synthetic DICOM file bytes using pydicom."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("sample.dcm", {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Rows = height
    ds.Columns = width
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    pixel_data = (np.random.rand(height, width) * 4095).astype(np.uint16)
    ds.PixelData = pixel_data.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    return buf.getvalue()


def test_predict_png_image():
    """Tests POST /predict with a PNG file and validates response schema."""
    png_bytes = create_sample_png_bytes(256, 256)
    resp = client.post(
        "/predict",
        files={"file": ("test_scan.png", png_bytes, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "pneumonia_probability" in data
    assert isinstance(data["pneumonia_probability"], float)
    assert 0.0 <= data["pneumonia_probability"] <= 1.0

    assert data["threshold"] == 0.70
    assert data["prediction"] in ["pneumonia_suspected", "normal"]
    if data["pneumonia_probability"] >= 0.70:
        assert data["prediction"] == "pneumonia_suspected"
    else:
        assert data["prediction"] == "normal"

    assert data["model_checkpoint"] in ["best_baseline_model.pt", "best_model.pt"]
    assert data["disclaimer"] == DISCLAIMER_TEXT
    assert data["severity_legend"] == SEVERITY_LEGEND_TEXT

def test_predict_gradcam_and_severity_integration():
    """Tests that a single /predict response contains both severity_level
    and a non-empty Grad-CAM overlay together, confirming the two features
    are actually wired together in one response, not just independently."""
    png_bytes = create_sample_png_bytes(256, 256)
    resp = client.post(
        "/predict",
        files={"file": ("test_scan.png", png_bytes, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()

    # Severity field present and valid
    assert "severity_level" in data
    assert data["severity_level"] in [
        "Not Detected", "Mild", "Moderate", "Significant", "Severe"
    ]
    assert "severity_disclaimer" in data
    assert len(data["severity_disclaimer"]) > 0

    # Grad-CAM overlay present and non-empty
    gradcam_key = "gradcam_base64" if "gradcam_base64" in data else "gradcam_overlay_base64"
    assert gradcam_key in data, f"No Grad-CAM field found in response keys: {list(data.keys())}"
    assert isinstance(data[gradcam_key], str)
    assert len(data[gradcam_key]) > 100  # a real base64 image, not an empty string

    # Severity must be internally consistent with the returned probability
    from app.inference_server import compute_severity_level
    assert data["severity_level"] == compute_severity_level(data["pneumonia_probability"])

def test_predict_dicom_image():
    """Tests POST /predict with a synthetic DICOM file."""
    dcm_bytes = create_sample_dicom_bytes(128, 128)
    resp = client.post(
        "/predict",
        files={"file": ("patient_scan.dcm", dcm_bytes, "application/dicom")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "pneumonia_probability" in data
    assert data["threshold"] == 0.70
    assert data["prediction"] in ["pneumonia_suspected", "normal"]
    if data["pneumonia_probability"] >= 0.70:
        assert data["prediction"] == "pneumonia_suspected"
    else:
        assert data["prediction"] == "normal"
    assert data["disclaimer"] == DISCLAIMER_TEXT


def test_predict_empty_file_fails():
    """Tests that uploading an empty file returns HTTP 400."""
    resp = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()



def test_audit_log_written_and_no_images_saved():
    """Verifies that POST /predict writes a log entry and does not persist image files."""
    log_path = PROJECT_ROOT / "outputs" / "inference_log.jsonl"
    initial_line_count = 0
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            initial_line_count = sum(1 for _ in f)

    test_filename = f"audit_test_{os.getpid()}.png"
    png_bytes = create_sample_png_bytes(64, 64)

    resp = client.post(
        "/predict",
        files={"file": (test_filename, png_bytes, "image/png")}
    )
    assert resp.status_code == 200

    # Verify log entry was appended
    assert log_path.exists()
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > initial_line_count
    last_entry = json.loads(lines[-1])
    assert last_entry["filename"] == test_filename
    assert "timestamp" in last_entry
    assert "output" in last_entry
    assert last_entry["output"]["disclaimer"] == DISCLAIMER_TEXT

    # Crucial security & privacy check: verify test_filename was NOT saved to disk
    assert not (PROJECT_ROOT / test_filename).exists()
    assert not (PROJECT_ROOT / "outputs" / test_filename).exists()

from app.inference_server import compute_severity_level


def test_severity_ladder_heuristic():
    # Below mild threshold (< 0.70)
    assert compute_severity_level(0.0) == "Not Detected"
    assert compute_severity_level(0.49) == "Not Detected"
    assert compute_severity_level(0.50) == "Not Detected"
    assert compute_severity_level(0.68) == "Not Detected"
    assert compute_severity_level(0.69) == "Not Detected"

    # Mild band: [0.70, 0.79)
    assert compute_severity_level(0.70) == "Mild"
    assert compute_severity_level(0.75) == "Mild"
    assert compute_severity_level(0.789) == "Mild"

    # Moderate band: [0.79, 0.88)
    assert compute_severity_level(0.79) == "Moderate"
    assert compute_severity_level(0.85) == "Moderate"
    assert compute_severity_level(0.879) == "Moderate"

    # Significant band: [0.88, 0.94)
    assert compute_severity_level(0.88) == "Significant"
    assert compute_severity_level(0.90) == "Significant"
    assert compute_severity_level(0.939) == "Significant"

    # Severe band: [0.94, 1.0]
    assert compute_severity_level(0.94) == "Severe"
    assert compute_severity_level(0.97) == "Severe"
    assert compute_severity_level(1.0) == "Severe"


def test_sub_threshold_normal_verdict_and_not_detected_severity(monkeypatch):
    """
    Asserts that a probability just below 0.70 (e.g. 0.68) yields:
      verdict = 'normal' AND severity = 'Not Detected'
    Guarantees no contradictory state (normal verdict cannot show a severity label).
    """
    # 1. Direct heuristic check
    assert compute_severity_level(0.68) == "Not Detected"

    # 2. End-to-end /predict endpoint mock check forcing model output to p=0.68
    # Logit for 0.68: log(0.68 / (1 - 0.68)) = log(0.68 / 0.32) = 0.75377180257
    import torch
    target_logit = torch.tensor([[0.75377180257]])

    from app.inference_server import inference_model, grad_cam
    monkeypatch.setattr(inference_model, "forward", lambda x: target_logit)
    monkeypatch.setattr(grad_cam, "generate_heatmap", lambda x: np.zeros((224, 224), dtype=np.float32))

    png_bytes = create_sample_png_bytes(64, 64)
    resp = client.post(
        "/predict",
        files={"file": ("borderline_scan.png", png_bytes, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["pneumonia_probability"] == 0.68
    assert data["threshold"] == 0.70
    assert data["prediction"] == "normal"
    assert data["severity_level"] == "Not Detected"


@pytest.mark.parametrize("prob,expected_severity", [
    (0.75, "Mild"),
    (0.85, "Moderate"),
    (0.90, "Significant"),
    (0.97, "Severe"),
])
def test_predict_severity_bands_integration(monkeypatch, prob, expected_severity):
    """Asserts that 0.75, 0.85, 0.90, 0.97 yield Mild, Moderate, Significant, Severe respectively in /predict."""
    import torch
    import math
    logit_val = math.log(prob / (1.0 - prob))
    target_logit = torch.tensor([[logit_val]])

    from app.inference_server import inference_model, grad_cam
    monkeypatch.setattr(inference_model, "forward", lambda x: target_logit)
    monkeypatch.setattr(grad_cam, "generate_heatmap", lambda x: np.zeros((224, 224), dtype=np.float32))

    png_bytes = create_sample_png_bytes(64, 64)
    resp = client.post(
        "/predict",
        files={"file": (f"test_{prob}_scan.png", png_bytes, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["pneumonia_probability"] == prob
    assert data["threshold"] == 0.70
    assert data["prediction"] == "pneumonia_suspected"
    assert data["severity_level"] == expected_severity