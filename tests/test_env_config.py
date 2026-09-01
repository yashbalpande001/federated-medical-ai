import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_config import get_environment, get_paths, EnvPaths, load_compute_budget


def test_get_environment():
    env = get_environment()
    assert env in ["kaggle", "colab", "local"]


def test_get_paths():
    paths = get_paths()
    assert isinstance(paths, EnvPaths)
    assert "data_root" in paths
    assert "output_root" in paths
    assert "checkpoint_root" in paths
    assert "config_root" in paths
    assert "project_root" in paths
    assert "rsna_raw_dir" in paths
    assert "rsna_labels_path" in paths
    assert "rsna_images_dir" in paths

    # Test attribute access
    assert paths.data_root == paths["data_root"]
    assert paths.output_root == paths["output_root"]
    assert paths.checkpoint_root == paths["checkpoint_root"]
    assert paths.rsna_raw_dir == paths["rsna_raw_dir"]

    # Verify directories exist
    assert paths.data_root.exists()
    assert paths.output_root.exists()
    assert paths.checkpoint_root.exists()


def test_load_compute_budget():
    budget = load_compute_budget()
    assert isinstance(budget, dict)
    assert "dataset_split" in budget
    assert "active_split" in budget["dataset_split"]
    assert budget["dataset_split"]["active_split"] in ["subset", "full"]

