import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rsna_resnet import RSNABaselineResNet18
from src.training.train_baseline import (
    compute_class_pos_weight,
    calculate_comprehensive_metrics,
    train_centralized_baseline,
)
from src.utils.env_config import get_paths


def test_rsna_resnet_model():
    model = RSNABaselineResNet18(pretrained=False, num_classes=1)
    dummy_input = torch.randn(4, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (4,)
    total_params, trainable_params = model.get_trainable_params_count()
    assert total_params > 10_000_000
    assert trainable_params == total_params


def test_calculate_comprehensive_metrics():
    y_true = [0, 0, 1, 1, 0, 1]
    y_probs = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
    metrics = calculate_comprehensive_metrics(y_true, y_probs)

    assert "auc" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "specificity" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_train_baseline_dry_run():
    results = train_centralized_baseline(epochs=1, lr=1e-4, dry_run=True)

    paths = get_paths()
    best_model_path = paths.output_root / "checkpoints" / "best_model.pt"
    results_json_path = paths.output_root / "results.json"
    report_md_path = paths.output_root / "baseline_report.md"

    assert best_model_path.exists()
    assert results_json_path.exists()
    assert report_md_path.exists()
    assert "test_metrics" in results
    assert "auc" in results["test_metrics"]
