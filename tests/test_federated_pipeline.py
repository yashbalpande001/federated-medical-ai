import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.client import RSNAFlowerClient
from src.federated.server import SaveAndEvaluateFedAvg
from src.utils.env_config import get_paths


def test_rsna_flower_client_get_set_parameters(tmp_path):
    """Verifies RSNAFlowerClient weight serialization/deserialization."""
    client = RSNAFlowerClient(
        client_id=0,
        partition_csv=tmp_path / "client_0.csv",
        images_dir=tmp_path / "images",
        batch_size=4,
        epochs_per_round=1,
        is_synthetic=True,
    )

    params = client.get_parameters(config={})
    assert isinstance(params, list)
    assert len(params) > 0
    assert isinstance(params[0], np.ndarray)

    # Mutate parameters and set back
    mutated = [p + 0.01 for p in params]
    client.set_parameters(mutated)

    retrieved = client.get_parameters(config={})
    np.testing.assert_allclose(retrieved[0], mutated[0], rtol=1e-5)


def test_rsna_flower_client_fit_step(tmp_path):
    """Verifies that 1 fit epoch updates client weights cleanly."""
    client = RSNAFlowerClient(
        client_id=0,
        partition_csv=tmp_path / "client_0.csv",
        images_dir=tmp_path / "images",
        batch_size=4,
        epochs_per_round=1,
        is_synthetic=True,
    )

    initial_params = client.get_parameters(config={})
    updated_params, num_samples, metrics = client.fit(initial_params, config={})

    assert len(updated_params) == len(initial_params)
    assert num_samples > 0
    assert "train_loss" in metrics
    assert isinstance(metrics["train_loss"], float)
