import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_config import get_environment, get_paths, EnvPaths


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

    # Test attribute access
    assert paths.data_root == paths["data_root"]
    assert paths.output_root == paths["output_root"]
    assert paths.checkpoint_root == paths["checkpoint_root"]

    # Verify directories exist
    assert paths.data_root.exists()
    assert paths.output_root.exists()
    assert paths.checkpoint_root.exists()
