import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rsna_dataset import generate_synthetic_metadata
from src.federated.dataset_partitioner import partition_dataset_dirichlet


def test_partition_dataset_patient_isolation():
    """Verify that no patientId exists on more than one client (zero patient leakage)."""
    df = generate_synthetic_metadata(num_patients=200, seed=42)
    client_dfs = partition_dataset_dirichlet(df, num_clients=5, alpha=0.5, seed=42)

    seen_pids = set()
    for client_id, client_df in client_dfs.items():
        client_pids = set(client_df["patientId"].unique())
        # Intersect with already seen patient IDs
        overlap = seen_pids.intersection(client_pids)
        assert len(overlap) == 0, f"Patient leakage detected on Client {client_id}: {overlap}"
        seen_pids.update(client_pids)

    # Total partitioned patient count must equal initial patient count
    assert len(seen_pids) == len(df["patientId"].unique())


def test_partition_dataset_non_empty():
    """Verify that all virtual clients receive non-empty DataFrames."""
    df = generate_synthetic_metadata(num_patients=100, seed=123)
    client_dfs = partition_dataset_dirichlet(df, num_clients=4, alpha=0.5, seed=123)

    assert len(client_dfs) == 4
    for i in range(4):
        assert len(client_dfs[i]) > 0, f"Client {i} received empty partition"
