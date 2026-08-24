# Federated Medical AI - Baseline CNN

This repository serves as the baseline CNN setup for the `federated-medical-ai` project. It provides data loading, model definition, training, evaluation, and unit testing using the CIFAR-10 dataset as a baseline.

## Project Structure

```
federated-medical-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── cnn.yaml
├── data/
│   └── README.md
├── src/
│   ├── data/
│   │   └── cifar10.py
│   ├── models/
│   │   └── simple_cnn.py
│   ├── training/
│   │   └── train_cifar10.py
│   └── evaluation/
│       └── evaluate.py
├── outputs/
└── tests/
    └── test_model.py
```

## Setup

Install the required python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training
Train the baseline `SimpleCNN` model on CIFAR-10:

```bash
python src/training/train_cifar10.py
```

This script saves the best model checkpoint to `outputs/simple_cnn_best.pt` and metrics log to `outputs/training_log.csv`.

### 2. Evaluation
Evaluate the best saved model checkpoint on the test set:

```bash
python src/evaluation/evaluate.py
```

This computes classification metrics (Accuracy, Precision, Recall, F1) and produces `outputs/confusion_matrix.png`.

### 3. Running Tests
Run the unit test suite:

```bash
pytest
```
