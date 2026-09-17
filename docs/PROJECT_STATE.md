# Project State & System Ledger

## Project Overview
- **Repository**: `federated-medical-ai`
- **Architecture**: Federated Learning (FedAvg / FedProx) + Multi-Agent Routing System
- **Dataset**: RSNA Pneumonia Detection DICOM Subset (6,000 unique patients, seed=42 split: 4,200 Train / 900 Val / 900 Test)
- **Verified Benchmark Test Set**: 900 samples (208 positives, 23.11% positive rate)

---

## 1. Verified Model Benchmarks

| Milestone / Notebook | Model Architecture | Method | Test ROC-AUC | Test Accuracy | Status |
|---|---|---|---|---|---|
| Step 5 (`kaggle_step5_baseline.ipynb`) | ResNet-18 | Centralized Baseline | **0.8536** | 82.44% | **VERIFIED** |
| Step 6 (`kaggle_step6_imbalance.ipynb`) | ResNet-18 | Focal Loss ($\gamma=2.0$) | 0.8490 | 81.67% | **VERIFIED** |
| Step 7 (`kaggle_step7_gradcam.ipynb`) | ResNet-18 | Grad-CAM Saliency | Anatomical Lung Focus Verified | - | **VERIFIED** |
| Step 10 (`kaggle_step10_fedavg.ipynb`) | ResNet-18 | FedAvg (5 Clients, Dirichlet $\alpha=0.5$) | 0.8210 | 79.80% | **VERIFIED** |
| Step 11 (`kaggle_step11_fedprox.ipynb`) | ResNet-18 | FedProx ($\mu=0.1$) | **0.8412** | 81.20% | **VERIFIED** |
| Step 12 (Multi-Machine Real Round) | ResNet-18 | Live Tailscale 2-Client FedAvg | Pending Execution | Pending Execution | **PENDING** |

---

## 2. Multi-Machine Client Partition Mapping (Step 12 Deployment)

- **Dataset Split**: 6,000 unique RSNA patients ($70\% / 15\% / 15\%$, `seed=42`).
- **5-Client Dirichlet Partition Index** ($\alpha=0.5, \text{seed}=42$):
  - **Client 1** (Friend's PC: Ryzen 7, RTX 4050, 16GB RAM): Partition Index `0` (1,366 samples, 57.0% positive rate).
  - **Client 2** (Second PC: i5, CPU-only, 16GB RAM): Partition Index `3` (305 samples, 29.8% positive rate).

---

## 3. Directory Layout & Key Components

- `docs/`: Deployment setup guides (`SERVER_SETUP.md`, `TEAMMATE_SETUP.md`), explainability report (`gradcam_findings.md`), and canonical state ledger (`PROJECT_STATE.md`).
- `kaggle_notebooks/`: Standalone ready-to-run `.ipynb` notebooks for Kaggle GPU execution.
- `docker/`: Server aggregator container (`docker/server/`) and client upload container (`docker/client/`).
- `data/`: Local directory holding `test_set_900.pt` tensor for server evaluation.
- `outputs/`: Round aggregation metrics (`round1_results.json`).

---

## 4. Security & Privacy Controls
- All Tailscale IPs dynamically retrieved via environment variables (`SERVER_TAILSCALE_IP`, `SERVER_PORT`).
- Global pattern rules in `.gitignore` strictly exclude all `.pt`, `.pth`, `.ckpt`, `.dcm`, `.env`, and secret credential files (`kaggle.json`).
