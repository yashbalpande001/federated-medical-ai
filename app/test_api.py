"""
FastAPI Integration & Latency Benchmark Script (Step 19).
Tests /health and /predict endpoints with sample X-Ray, CT, and MRI scans.
"""

import sys
import io
import time
from pathlib import Path

from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

client = TestClient(app)


def create_sample_image_bytes(width: int, height: int) -> bytes:
    """Helper creating synthetic sample PNG image bytes for testing API uploads."""
    img_arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_endpoints():
    print("\n" + "=" * 70)
    print("      STEP 19: FASTAPI INFERENCE API INTEGRATION TESTS")
    print("=" * 70)

    # 1. Test /health Endpoint
    print("\n[*] Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected status 200, got {resp.status_code}"
    health_data = resp.json()
    print(f"--> [OK] /health Response: Status='{health_data['status']}' | Environment='{health_data['environment']}' | PyTorch Device='{health_data['pytorch_device']}'")

    # 2. Test /predict Endpoint with X-Ray scan (Square 224x224)
    print("\n[*] Testing POST /predict (Chest X-Ray Sample)...")
    xray_bytes = create_sample_image_bytes(224, 224)
    t0 = time.time()
    resp_xray = client.post(
        "/predict",
        files={"file": ("chest_xray.png", xray_bytes, "image/png")},
        params={"patient_id": "PAT-TEST-XRAY"},
    )
    latency_xray = (time.time() - t0) * 1000.0

    assert resp_xray.status_code == 200, f"Expected 200, got {resp_xray.status_code}: {resp_xray.text}"
    xray_data = resp_xray.json()
    report = xray_data["report"]
    print(f"--> [OK] X-Ray Case Report Generated (ID: {report['metadata']['report_id']})")
    print(f"     Modality: {report['imaging_modality']['detected_modality']} | Triage: {report['triage_summary']['title']}")
    print(f"     Latency: {xray_data['inference_latency_ms']} ms (TestClient total: {latency_xray:.2f} ms)")

    # 3. Test /predict Endpoint with CT scan (Landscape 300x224)
    print("\n[*] Testing POST /predict (CT Scan Sample)...")
    ct_bytes = create_sample_image_bytes(300, 224)
    resp_ct = client.post(
        "/predict",
        files={"file": ("ct_scan.png", ct_bytes, "image/png")},
        params={"patient_id": "PAT-TEST-CT"},
    )
    assert resp_ct.status_code == 200, f"Expected 200, got {resp_ct.status_code}"
    ct_report = resp_ct.json()["report"]
    print(f"--> [OK] CT Case Report Generated (ID: {ct_report['metadata']['report_id']})")
    print(f"     Modality: {ct_report['imaging_modality']['detected_modality']} | Triage: {ct_report['triage_summary']['title']}")

    # 4. Test /predict Endpoint with MRI scan (Portrait 180x240)
    print("\n[*] Testing POST /predict (MRI Scan Sample)...")
    mri_bytes = create_sample_image_bytes(180, 240)
    resp_mri = client.post(
        "/predict",
        files={"file": ("mri_scan.png", mri_bytes, "image/png")},
        params={"patient_id": "PAT-TEST-MRI"},
    )
    assert resp_mri.status_code == 200, f"Expected 200, got {resp_mri.status_code}"
    mri_report = resp_mri.json()["report"]
    print(f"--> [OK] MRI Case Report Generated (ID: {mri_report['metadata']['report_id']})")
    print(f"     Modality: {mri_report['imaging_modality']['detected_modality']} | Triage: {mri_report['triage_summary']['title']}")

    print("\n" + "-" * 70)
    print("[SUCCESS] ALL FASTAPI INFERENCE ENDPOINT TESTS PASSED CLEANLY!")
    print("-" * 70)


if __name__ == "__main__":
    test_api_endpoints()
