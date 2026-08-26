# Federated Medical AI - Baseline CNN & ResNet-18 Transfer Learning

This repository serves as the baseline CNN and Transfer Learning setup for the `federated-medical-ai` project. It provides dataset loading, model architectures (`SimpleCNN`, `ResNet-18` Transfer Learning), training, evaluation, model comparison, and unit testing on CIFAR-10.

## Project Structure

```
federated-medical-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── cnn.yaml
│   └── resnet.yaml
├── data/
│   └── README.md
├── src/
│   ├── data/
│   │   └── cifar10.py
│   ├── models/
│   │   ├── simple_cnn.py
│   │   └── resnet_transfer.py
│   ├── training/
│   │   ├── train_cifar10.py
│   │   └── train_resnet_transfer.py
│   └── evaluation/
│       ├── evaluate.py
│       └── compare_models.py
├── outputs/
└── tests/
    ├── test_model.py
    └── test_resnet.py
```

## Setup

Install the required python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training Baseline CNN
Train the baseline `SimpleCNN` model on CIFAR-10 ($32 \times 32$ images):

```bash
python src/training/train_cifar10.py
```
Saves best checkpoint to `outputs/simple_cnn_best.pt` and metrics to `outputs/training_log.csv`.

---

### 2. Training ResNet-18 Transfer Learning
Train ResNet-18 (pretrained on ImageNet, upsampled to $224 \times 224$) in either feature extractor or fine-tuning mode:

- **Frozen Mode (Feature Extractor):**
  ```bash
  python src/training/train_resnet_transfer.py --mode frozen
  ```
  Saves best checkpoint to `outputs/resnet_frozen_best.pt` and metrics to `outputs/resnet_frozen_log.csv`.

- **Finetune Mode (End-to-End):**
  ```bash
  python src/training/train_resnet_transfer.py --mode finetune
  ```
  Saves best checkpoint to `outputs/resnet_finetune_best.pt` and metrics to `outputs/resnet_finetune_log.csv`.

---

### 3. Evaluating & Comparing Models

- **Single Model Evaluation (SimpleCNN):**
  ```bash
  python src/evaluation/evaluate.py
  ```

- **Compare All Models (Validation Accuracy per Epoch & Parameters):**
  ```bash
  python src/evaluation/compare_models.py
  ```
  Generates `outputs/model_comparison.png` comparing validation accuracy curves across models and prints parameter counts and training speed summary.

---

### 4. Running Unit Tests

Run the full pytest suite:

```bash
pytest
```
