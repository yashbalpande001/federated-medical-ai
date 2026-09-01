import sys
from pathlib import Path
import pytest
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rsna_dataset import (
    RSNADataset,
    generate_synthetic_metadata,
    split_rsna_dataset,
    get_rsna_dataloaders,
)
from src.utils.env_config import load_compute_budget


def test_generate_synthetic_metadata():
    num_patients = 50
    df = generate_synthetic_metadata(num_patients=num_patients, seed=42)
    assert len(df) == num_patients
    assert "patientId" in df.columns
    assert "Target" in df.columns
    assert "bboxes" in df.columns
    assert set(df["Target"].unique()).issubset({0, 1})


def test_patient_disjoint_split():
    df = generate_synthetic_metadata(num_patients=100, seed=42)
    train_df, val_df, test_df = split_rsna_dataset(
        df, subset_size=100, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10

    # Ensure patient sets are strictly disjoint
    train_pids = set(train_df["patientId"])
    val_pids = set(val_df["patientId"])
    test_pids = set(test_df["patientId"])

    assert len(train_pids.intersection(val_pids)) == 0
    assert len(train_pids.intersection(test_pids)) == 0
    assert len(val_pids.intersection(test_pids)) == 0


def test_compute_budget_subset_enforcement():
    df = generate_synthetic_metadata(num_patients=200, seed=42)
    subset_limit = 50
    train_df, val_df, test_df = split_rsna_dataset(
        df, subset_size=subset_limit, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    total_split_patients = len(train_df) + len(val_df) + len(test_df)
    assert total_split_patients == subset_limit


def test_rsna_dataset_item():
    df = generate_synthetic_metadata(num_patients=10, seed=42)
    images_dir = PROJECT_ROOT / "data" / "rsna" / "stage_2_train_images"
    dataset = RSNADataset(df, images_dir=images_dir, is_synthetic=True, image_size=(224, 224))

    assert len(dataset) == 10
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert label in [0, 1]


def test_get_rsna_dataloaders():
    train_loader, val_loader, test_loader, summary = get_rsna_dataloaders(override_batch_size=8)

    assert summary["batch_size"] == 8
    assert summary["active_split"] == "subset"
    assert "total_patients" in summary

    # Verify train loader output batch
    images, labels = next(iter(train_loader))
    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)
