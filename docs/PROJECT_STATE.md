# Project State & System Ledger

## Project Overview
- **Repository**: `federated-medical-ai`
- **Primary Operational Focus**: Production-Grade Chest X-Ray Pneumonia Screening Service with 5-tier Clinical Severity Heuristic & Grad-CAM Visual Explainability (Port 8090).
- **Secondary Research Focus**: Offline Federated Learning (FedAvg / FedProx), Multi-Agent Hierarchy Prototypes, and Differential Privacy Ablations archived in `/research`.
- **Dataset**: RSNA Pneumonia Detection Challenge DICOM Subset (6,000 unique patients, seed=42 split: 4,200 Train / 900 Val / 900 Test).
- **Verified Benchmark Test Set**: 900 held-out samples (`data/test_set_900.pt`, 208 positives, 23.11% positive rate).

---

## 1. Post-Pruning Architecture & Directory Layout

The codebase enforces a strict separation of concerns between the **live operational service** and the **offline research archive**:

```text
federated-medical-ai/
├── app/                                 # LIVE CLINICAL INFERENCE SERVICE (Port 8090)
│   ├── inference_server.py              # Standalone FastAPI service (RSNA ResNet-18, Grad-CAM, audit logger)
│   ├── static/doctor_ui.html            # Zero-build Doctor UI (popover Grad-CAM, severity heuristic)
│   └── __init__.py
├── src/                                 # LIVE SERVICE CORE MODULES
│   ├── models/
│   │   ├── rsna_resnet.py               # Active ResNet-18 transfer learning architecture
│   │   └── __init__.py
│   ├── explainability/
│   │   ├── gradcam.py                   # PyTorch forward/backward hook Grad-CAM implementation
│   │   └── __init__.py
│   ├── utils/
│   │   ├── env_config.py                # Environment configuration utilities
│   │   └── __init__.py
│   └── __init__.py
├── tests/                               # LIVE VERIFICATION SUITE
│   ├── test_inference_server.py         # 9 automated tests (fail-loud, health, predict, audit, severity)
│   └── __init__.py
├── data/                                # ACTIVE RUNTIME DATA
│   ├── test_set_900.pt                  # Pre-extracted 900-patient test tensor
│   └── README.md
├── outputs/                             # LIVE RUNTIME OUTPUTS
│   ├── checkpoints/
│   │   └── best_baseline_model.pt       # Canonical active model checkpoint (44.8 MB)
│   ├── inference_log.jsonl              # Real-time request metadata audit log
│   └── splits/                          # RSNA patient split CSVs (train/val/test)
├── docker/                              # FEDERATED TRANSPORT DEMO (Port 8080)
│   ├── client/                          # Lightweight client upload container
│   ├── server/                          # Real FedAvg aggregator container (server_aggregator.py)
│   └── docker-compose.yml
├── kaggle_notebooks/                    # STANDALONE REPRODUCIBLE NOTEBOOKS
│   ├── export_test_set.ipynb
│   ├── kaggle_step5_baseline.ipynb
│   ├── kaggle_step6_imbalance.ipynb
│   ├── kaggle_step7_gradcam.ipynb
│   ├── kaggle_step10_fedavg.ipynb
│   └── kaggle_step11_fedprox.ipynb
├── docs/                                # SYSTEM DOCUMENTATION
│   ├── PROJECT_STATE.md                 # Canonical system ledger (this file)
│   ├── SERVER_SETUP.md                  # Docker server deployment guide
│   ├── TEAMMATE_SETUP.md                # Teammate client submission guide
│   └── gradcam_findings.md              # Radiological Grad-CAM localization analysis
├── research/                            # ARCHIVED OFFLINE RESEARCH & THESIS ARTIFACTS
│   ├── agents/                          # Hierarchical multi-agent router & synthetic experts
│   ├── federated/                       # Flower FL simulation, FedAvg, FedProx, DP-SGD ablation
│   ├── training/                        # Offline training scripts (baseline, imbalance, segmentation)
│   ├── models/                          # MobileNetV3, U-Net, modality classifier, simple CNN
│   ├── evaluations/                     # System benchmarks, comparison scripts, batch Grad-CAM
│   ├── checkpoints/                     # Research checkpoints (FedAvg, FedProx, CT/MRI experts, U-Net)
│   ├── docs/                            # Step-by-step reports, final report, presentation outline
│   ├── tests/                           # Offline research test suite
│   ├── configs/                         # YAML hyperparameter configurations
│   ├── data/                            # Research dataset loaders and EDA scripts
│   ├── prototype_app/                   # Legacy Step 19 multi-agent prototype (main.py, test_api.py)
│   ├── requirements-research.txt        # Research-only dependencies
│   └── README.md                        # Research directory index and guide
├── requirements.txt                     # TRIMMED RUNTIME DEPENDENCIES (live service only)
└── README.md                            # PRIMARY PROJECT DOCUMENTATION
```

---

## 2. Verified Model Benchmarks & Status

| Milestone / Component | Model Architecture | Method | Test ROC-AUC | Test Accuracy | Status |
|---|---|---|---|---|---|
| **Step 5 Centralized Baseline** | ResNet-18 | Centralized PyTorch (RSNA DICOM) | **0.8536** | 82.44% | **LIVE SERVICE ACTIVE** (`best_baseline_model.pt`) |
| **Step 6 Imbalance Handling** | ResNet-18 | Focal Loss ($\gamma=2.0$) | 0.8490 | 81.67% | **ARCHIVED** (`research/checkpoints/best_imbalance_model.pt`) |
| **Step 7 Grad-CAM Explainability** | ResNet-18 | Saliency via `layer4[1].conv2` | Anatomical Lung Focus | - | **LIVE SERVICE ACTIVE** (`src/explainability/gradcam.py`) |
| **Step 10 Federated Avg** | ResNet-18 | Flower FedAvg (5 Clients, Dirichlet $\alpha=0.5$) | 0.8210 | 79.80% | **ARCHIVED** (`research/checkpoints/best_fedavg_model.pt`) |
| **Step 11 FedProx** | MobileNetV3 | FedProx ($\mu=0.01$) | **0.8350** | 81.20% | **ARCHIVED** (`research/checkpoints/best_fedprox_mu_0.01_model.pt`) |
| **Step 12 Multi-Machine Round** | ResNet-18 | Docker Tailscale 2-Client FedAvg | Verified Provenance | - | **CONTAINER PROTOTYPE READY** (`docker/`) |
| **Step 16-17 Multi-Agent Hierarchy** | ResNet-18 | Router + CT/MRI Synthetic Experts | Synthetic 1.0000 | - | **ARCHIVED RESEARCH PROTOTYPE** (`research/agents/`) |

---

## 3. Live Service Operational Details (Port 8090)

- **Entry Point**: `app/inference_server.py`
- **UI Route**: `GET /` serves `app/static/doctor_ui.html`
- **Health Route**: `GET /health` returns JSON health status, port `8090`, threshold `0.5`, and model checkpoint filename.
- **Inference Route**: `POST /predict` accepts DICOM (`.dcm`), PNG, JPG, JPEG.
  - Returns: `pneumonia_probability`, `prediction` ("Pneumonia" or "Normal"), `severity_level` ("Not detected", "Mild", "Moderate", "Significant", "Severe"), `severity_disclaimer`, and `gradcam_overlay_base64`.
  - Privacy: Images are processed entirely in memory; strictly **never written to disk**.
  - Audit: Request metadata (timestamp, filename, sha256, latency, probability, prediction, severity) logged to `outputs/inference_log.jsonl`.
- **Target Conv Layer**: `inference_model.resnet.layer4[1].conv2`.

---

## 4. Multi-Machine Client Partition Mapping (Step 12 Deployment)

- **Dataset Split**: 6,000 unique RSNA patients ($70\% / 15\% / 15\%$, `seed=42`).
- **5-Client Dirichlet Partition Index** ($\alpha=0.5, \text{seed}=42$):
  - **Client 1** (Friend's PC: Ryzen 7, RTX 4050, 16GB RAM): Partition Index `0` (1,366 samples, 57.0% positive rate).
  - **Client 2** (Second PC: i5, CPU-only, 16GB RAM): Partition Index `3` (305 samples, 29.8% positive rate).
