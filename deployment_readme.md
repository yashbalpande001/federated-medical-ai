# 🚀 Deployment Guide: Federated Medical AI Services

This guide provides instructions for deploying and running the operational services of the **Federated Medical AI** repository.

---

## 1. Clinical Demonstration Inference Service (Port 8090)

The primary live service is a standalone FastAPI web application providing single-image pneumonia screening with Grad-CAM explainability and a 5-tier clinical confidence heuristic.

### Prerequisites & Setup
Ensure dependencies from the root `requirements.txt` are installed:

```bash
pip install -r requirements.txt
```

### Local Execution
Start the inference server on port `8090`:

```bash
python app/inference_server.py
```

Or run via Uvicorn with auto-reload:

```bash
uvicorn app.inference_server:app --host 0.0.0.0 --port 8090
```

### Endpoints
- **Doctor Web UI**: `http://localhost:8090/` (single-page interactive interface).
- **Health Check**: `http://localhost:8090/health` (returns JSON status, loaded checkpoint, port, threshold).
- **Inference**: `http://localhost:8090/predict` (POST multipart/form-data image: `.dcm`, `.png`, `.jpg`).
- **Interactive OpenAPI Docs**: `http://localhost:8090/docs`.

### Checkpoint Resolution & Environment Variables
- `MODEL_CHECKPOINT_PATH`: Optional custom path to model weights. By default, the server resolves to `outputs/checkpoints/best_baseline_model.pt`.
- If the checkpoint file cannot be found, the server **fails loudly on startup** to prevent serving uninitialized random weights.

---

## 2. Federated Transport Container Demo (Port 8080)

The repository includes a containerized federated learning transport demonstration running over Tailscale or local bridge networking.

### Architecture
- **Server Aggregator**: `docker/server/server_aggregator.py` runs a FastAPI service on port `8080` that collects client PyTorch model state dicts, verifies SHA-256 provenance hashes, and computes weighted FedAvg aggregation.
- **Client Transfer**: `docker/client/client_transfer.py` runs in a lightweight Alpine container without PyTorch, reading local weights and streaming them via HTTP POST.

### Launch with Docker Compose
```bash
docker-compose up --build
```

Access the aggregator dashboard at `http://localhost:8080/`.

---

## 3. Automated Verification

Verify the live inference server and all schema contracts:

```bash
python -m pytest tests/test_inference_server.py -v
```

All 9 tests will execute, validating:
1. Fail-loud behavior on missing checkpoint
2. `/health` contract and regulatory disclaimers
3. Doctor UI root serving
4. PNG and DICOM image inference
5. Zero-byte upload error handling
6. In-memory privacy compliance (images never saved to disk)
7. Audit log creation in `outputs/inference_log.jsonl`
8. Severity ladder heuristic mapping (<0.50 to >=0.90)
9. End-to-end Grad-CAM heatmap generation

---

## 4. Offline Research & Multi-Agent Prototypes

For historical experiments, multi-agent hierarchical routing, synthetic CT/MRI experts, or Flower federated simulations, refer to the self-contained archive in [`research/README.md`](research/README.md).
