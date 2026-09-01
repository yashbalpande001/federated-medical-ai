import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.toy_fl.client import ToyFlowerClient
from src.federated.toy_fl.run_simulation import run_toy_fl_simulation
from src.federated.toy_fl.server import create_fedavg_strategy, weighted_metrics_aggregation


def test_toy_client_parameters_roundtrip():
    """Test get_parameters and set_parameters on ToyFlowerClient."""
    client = ToyFlowerClient(client_id=0)
    params_original = client.get_parameters(config={})

    assert isinstance(params_original, list)
    assert len(params_original) > 0

    # Modify parameters slightly
    params_modified = [p + 0.1 for p in params_original]
    client.set_parameters(params_modified)

    params_retrieved = client.get_parameters(config={})

    for p_mod, p_ret in zip(params_modified, params_retrieved):
        np.testing.assert_allclose(p_mod, p_ret, rtol=1e-5, atol=1e-5)


def test_toy_client_fit_and_evaluate():
    """Test local fit and evaluate execution on ToyFlowerClient."""
    client = ToyFlowerClient(client_id=1, epochs_per_round=1)
    params = client.get_parameters(config={})

    # Execute fit
    updated_params, num_samples, metrics_fit = client.fit(params, config={})
    assert len(updated_params) == len(params)
    assert num_samples > 0
    assert "train_loss" in metrics_fit
    assert metrics_fit["train_loss"] >= 0.0

    # Execute evaluate
    loss, num_val_samples, metrics_eval = client.evaluate(updated_params, config={})
    assert loss >= 0.0
    assert num_val_samples > 0
    assert "accuracy" in metrics_eval
    assert 0.0 <= metrics_eval["accuracy"] <= 1.0


def test_weighted_metrics_aggregation():
    """Test FedAvg weighted metrics aggregation logic."""
    metrics = [
        (100, {"accuracy": 0.80, "loss": 0.40}),
        (200, {"accuracy": 0.90, "loss": 0.30}),
    ]

    agg = weighted_metrics_aggregation(metrics)
    # Expected weighted accuracy: (100*0.80 + 200*0.90) / 300 = (80 + 180)/300 = 260/300 = 0.8667
    assert "accuracy" in agg
    assert "loss" in agg
    assert pytest.approx(agg["accuracy"], abs=1e-3) == 0.8667


def test_toy_fl_simulation_short():
    """Test short 2-client 2-round end-to-end Flower simulation."""
    res = run_toy_fl_simulation(num_clients=2, num_rounds=2)

    assert "accuracy_history" in res
    assert "loss_history" in res
    assert len(res["accuracy_history"]) == 2
    assert len(res["loss_history"]) == 2
