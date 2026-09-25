# 🏥 Federated Medical AI System: Chest X-Ray Pneumonia Detection

A privacy-preserving Medical AI system designed for diagnostic imaging. Combines an operational **Chest X-Ray Pneumonia Screening Service with Explainability (Port 8090)** and a **Multi-Machine Federated Learning Transport Demo (Port 8080)** on the RSNA Pneumonia Detection Challenge DICOM dataset, with exploratory research archived in `/research`.

---

## 🏗️ System Overview & Two Operating Modes

1. **Live Clinical Demonstration Service (Port 8090)**:
   - Standalone FastAPI web application serving an interactive Doctor UI (`http://localhost:8090/`).
   - Ingests DICOM (`.dcm`), PNG, JPG, and JPEG chest scans.
   - Evaluates scans using the centralized RSNA ResNet-18 baseline checkpoint (`outputs/checkpoints/best_baseline_model.pt`).
   - Produces a 0.5-threshold binary screening decision ("Pneumonia" vs. "Normal"), a 5-tier confidence-based clinical severity heuristic, and an in-memory Grad-CAM anatomical saliency overlay (`layer4[1].conv2`).
   - Enforces strict patient privacy (images never written to disk; in-memory processing only) and audit logging to `outputs/inference_log.jsonl`.

2. **Decentralized Federated Learning Demo (Port 8080)**:
   - Dockerized multi-machine transport architecture (`docker/server/`, `docker/client/`) demonstrating real parameter transfer and weighted FedAvg aggregation over Tailscale networks without exposing raw patient data.

3. **Academic & Thesis Research Archive (`/research`)**:
   - Preserves offline training pipelines, Flower federated simulations (FedAvg / FedProx hyperparameter sweeps), multi-agent routing prototypes, synthetic CT/MRI expert models, differential privacy ablations, and historical benchmark reports.

---

## 📊 Benchmark Results Matrix

> [!NOTE]
> All metrics below reflect empirical GPU evaluations on the patient-disjoint 900-patient held-out test set ($N=900$, 208 positives).

| Milestone / Strategy | System Topology | Test ROC-AUC | Test Recall (Sens) | Test Specificity | Test F1-Score | Test Precision | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 5 Centralized Baseline** | Centralized GPU (ResNet-18) | **0.8536** | **0.8173** | **0.7384** | **0.6082** | **0.4843** | **Active Live Model** (`best_baseline_model.pt`) |
| **Step 10 Standard FedAvg** | 5 Dirichlet Clients ($\alpha=0.5$) | **0.8291** | **0.0865** | **0.9783** | **0.1494** | **0.5455** | Verified Non-IID Drift (Archived) |
| **Step 11 FedProx ($\mu=0.1$)** | 5 Dirichlet Clients ($\alpha=0.5$) | **0.8249** | **0.2067** | **0.9538** | **0.3094** | **0.6230** | Verified Mitigation (Archived) |
| **Live Multi-Machine Round** | 2 Real Compute Origins | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | Container Demo Ready (`docker/`) |

---

## 📂 Repository Structure (Post-Pruning)

```text
federated-medical-ai/
├── app/                                 # LIVE CLINICAL INFERENCE SERVICE
│   ├── inference_server.py              # FastAPI server (Port 8090)
│   ├── static/
│   │   └── doctor_ui.html               # Doctor UI single-page interface with Grad-CAM popover
│   └── __init__.py
├── src/                                 # LIVE SERVICE BACKBONE MODULES
│   ├── models/
│   │   ├── rsna_resnet.py               # RSNABaselineResNet18 architecture
│   │   └── __init__.py
│   ├── explainability/
│   │   ├── gradcam.py                   # PyTorch Grad-CAM hook implementation
│   │   └── __init__.py
│   └── utils/
│       ├── env_config.py
│       └── __init__.py
├── tests/                               # LIVE VERIFICATION SUITE
│   ├── test_inference_server.py         # 9 automated tests for health, predict, audit, severity
│   └── __init__.py
├── data/                                # RUNTIME DATA
│   ├── test_set_900.pt                  # Pre-extracted 900-patient test tensor
│   └── README.md
├── outputs/                             # LIVE RUNTIME ARTIFACTS
│   ├── checkpoints/
│   │   └── best_baseline_model.pt       # Canonical active model checkpoint (44.8 MB)
│   ├── inference_log.jsonl              # Real-time request metadata audit log
│   └── splits/                          # RSNA patient split CSVs (train/val/test)
├── docker/                              # FEDERATED TRANSPORT DEMO (Port 8080)
│   ├── client/                          # Lightweight client upload container
│   ├── server/                          # Real FedAvg aggregator container
│   └── docker-compose.yml
├── kaggle_notebooks/                    # STANDALONE REPRODUCIBLE NOTEBOOKS
│   ├── export_test_set.ipynb
│   ├── kaggle_step5_baseline.ipynb
│   ├── kaggle_step6_imbalance.ipynb
│   ├── kaggle_step7_gradcam.ipynb
│   ├── kaggle_step10_fedavg.ipynb
│   └── kaggle_step11_fedprox.ipynb
├── docs/                                # SYSTEM DOCUMENTATION & GUIDES
│   ├── PROJECT_STATE.md                 # System ledger and active state
│   ├── SERVER_SETUP.md                  # Server aggregator deployment guide
│   ├── TEAMMATE_SETUP.md                # Teammate client execution guide
│   └── gradcam_findings.md              # Radiological explainability findings
├── research/                            # ARCHIVED OFFLINE RESEARCH & THESIS ASSETS
│   ├── agents/                          # Hierarchical multi-agent router & synthetic experts
│   ├── federated/                       # Flower FL simulations, FedAvg/FedProx, DP ablation
│   ├── training/                        # Offline training scripts (baseline, focal loss, U-Net)
│   ├── models/                          # MobileNetV3, U-Net, modality classifier, SimpleCNN
│   ├── evaluations/                     # System benchmarks & comparison scripts
│   ├── checkpoints/                     # Research checkpoints (FedAvg, FedProx, CT/MRI, U-Net)
│   ├── docs/                            # Step-by-step reports, final report, presentation outline
│   ├── tests/                           # Offline research test suite
│   ├── configs/                         # YAML hyperparameter configurations
│   ├── data/                            # Research dataset loaders and EDA scripts
│   ├── prototype_app/                   # Legacy Step 19 multi-agent prototype
│   ├── requirements-research.txt        # Research-only dependencies
│   └── README.md                        # Research directory index
├── requirements.txt                     # RUNTIME DEPENDENCIES (live service only)
└── README.md                            # PRIMARY DOCUMENTATION
```

---

## 🚀 Quickstart Guide

### 1. Launch the Clinical Demonstration Service (Port 8090)

```bash
# Install live service dependencies
pip install -r requirements.txt

# Run the inference service
python app/inference_server.py
# Or via Uvicorn:
uvicorn app.inference_server:app --host 0.0.0.0 --port 8090
```

Open your browser to:
- **Doctor UI**: [http://localhost:8090/](http://localhost:8090/)
- **Health Check**: [http://localhost:8090/health](http://localhost:8090/health)

### 2. Run Live Automated Tests

```bash
python -m pytest tests/test_inference_server.py -v
```

### 3. Federated Aggregator Demo (Port 8080)

To run the federated transport demo across containers:
- See [`docs/SERVER_SETUP.md`](docs/SERVER_SETUP.md) for aggregator launch instructions.
- See [`docs/TEAMMATE_SETUP.md`](docs/TEAMMATE_SETUP.md) for client weight upload instructions.

---

## 🩺 Clinical Demonstration & Severity Ladder Details

- **Binary Decision**: Threshold = `0.50` (`< 0.50` -> Normal, `>= 0.50` -> Pneumonia).
- **Severity Ladder** (confidence-based clinical heuristic):
  - `< 0.50`: **Not detected**
  - `0.50 - 0.65`: **Mild**
  - `0.65 - 0.80`: **Moderate**
  - `0.80 - 0.90`: **Significant**
  - `>= 0.90`: **Severe**
- **Visual Explainability**: In-memory Grad-CAM saliency map targeting `inference_model.resnet.layer4[1].conv2`. The UI provides an interactive thumbnail with click-to-expand modal view.
- **Audit Logging**: Every prediction logs timestamp, filename, SHA-256 hash, latency, probability, prediction, and severity to `outputs/inference_log.jsonl`.
- **Privacy Assurance**: Patient images are strictly processed in RAM and never written to disk.

---

## ⚠️ Regulatory Notice & Research Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL PROTOTYPE ONLY — NOT A MEDICAL DEVICE**
> This system is an academic research prototype. It is **not** cleared, certified, or approved by the FDA, EMA, CDSCO, or any regulatory body for medical diagnosis. It must **never** be used as a standalone diagnostic device or to direct patient therapy without certified radiologist review.
