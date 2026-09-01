import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rsna_dataset import get_rsna_dataloaders
from src.training.train_imbalance import (
    create_weighted_sampler_loader,
    run_imbalance_experiment,
)
from src.utils.env_config import get_paths


def test_create_weighted_sampler_loader():
    train_loader, _, _, _ = get_rsna_dataloaders(override_batch_size=16)
    sampler_loader = create_weighted_sampler_loader(train_loader, batch_size=16)

    assert sampler_loader is not None
    assert len(sampler_loader) > 0


def test_train_imbalance_dry_run():
    results = run_imbalance_experiment(epochs=1, dry_run=True)

    paths = get_paths()
    comp_json_path = paths.output_root / "comparison_results.json"
    report_md_path = paths.output_root / "imbalance_report.md"
    best_model_path = paths.output_root / "checkpoints" / "best_imbalance_model.pt"

    assert comp_json_path.exists()
    assert report_md_path.exists()
    assert best_model_path.exists()
    assert "focal_loss" in results
    assert "weighted_sampler" in results
    assert "winner" in results
