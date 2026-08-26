import os
import sys
from pathlib import Path


class EnvPaths(dict):
    """Dictionary subclass supporting attribute-style access for paths."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'EnvPaths' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def get_environment() -> str:
    """Detect whether code is running on Kaggle, Google Colab, or Local machine."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    elif "google.colab" in sys.modules or Path("/content").exists():
        return "colab"
    else:
        return "local"


def get_paths() -> EnvPaths:
    """
    Return dictionary of platform-aware directory paths.
    Supports both dict indexing (paths['output_root']) and attribute access (paths.output_root).
    """
    env = get_environment()
    file_project_root = Path(__file__).resolve().parent.parent.parent

    if env == "kaggle":
        base_dir = Path("/kaggle/working")
        data_dir = base_dir / "data"
        output_dir = base_dir / "outputs"
    elif env == "colab":
        base_dir = Path("/content")
        data_dir = base_dir / "data"
        output_dir = base_dir / "outputs"
    else:
        base_dir = file_project_root
        data_dir = base_dir / "data"
        output_dir = base_dir / "outputs"

    config_dir = file_project_root / "configs"
    checkpoint_dir = output_dir

    # Ensure output and data directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return EnvPaths({
        "environment": env,
        "project_root": file_project_root,
        "data_root": data_dir,
        "output_root": output_dir,
        "checkpoint_root": checkpoint_dir,
        "config_root": config_dir,
    })
