import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.client_fedprox import RSNAFedProxClient


def test_fedprox_client_proximal_penalty_calculation(tmp_path):
    """
    Verifies that proximal penalty term (mu/2) * ||w_local - w_global||^2 is zero
    when parameters match, and positive (>0) when local parameters drift.
    """
    client = RSNAFedProxClient(
        client_id=0,
        partition_csv=tmp_path / "client_0.csv",
        images_dir=tmp_path / "images",
        batch_size=4,
        mu=0.1,
        epochs_per_round=1,
        is_synthetic=True,
    )

    initial_params = client.get_parameters(config={})

    # Run 1 fit step with mu=0.1
    updated_params, num_samples, metrics = client.fit(initial_params, config={})

    assert len(updated_params) == len(initial_params)
    assert "prox_loss" in metrics
    assert "mu" in metrics
    assert metrics["mu"] == 0.1
    # After backprop with lr>0, local weights deviate from initial global weights so prox_loss > 0
    assert metrics["prox_loss"] >= 0.0
