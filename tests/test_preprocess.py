import sys
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    check_gatekeeper_outliers,
    resize_and_expand_channels,
    run_preprocessing_and_caching,
)
from src.utils.env_config import get_paths


def test_gatekeeper_outlier_checks():
    # 1. Normal contrast image -> Should Pass
    arr_normal = np.random.normal(loc=120, scale=40, size=(100, 100)).clip(0, 255).astype(np.uint8)
    is_flagged, reason = check_gatekeeper_outliers(arr_normal)
    assert is_flagged is False
    assert reason == "Passed"

    # 2. Blank / Zero contrast image (std < 5.0) -> Should Flag
    arr_blank = np.full((100, 100), fill_value=128, dtype=np.uint8)
    is_flagged, reason = check_gatekeeper_outliers(arr_blank)
    assert is_flagged is True
    assert "Blank or low contrast" in reason

    # 3. Extreme dark image (mean < 10.0) -> Should Flag
    arr_dark = np.full((100, 100), fill_value=2, dtype=np.uint8)
    is_flagged, reason = check_gatekeeper_outliers(arr_dark)
    assert is_flagged is True
    assert "Extreme dark outlier" in reason

    # 4. Extreme bright image (mean > 245.0) -> Should Flag
    arr_bright = np.full((100, 100), fill_value=250, dtype=np.uint8)
    is_flagged, reason = check_gatekeeper_outliers(arr_bright)
    assert is_flagged is True
    assert "Extreme bright outlier" in reason


def test_resize_and_expand_channels():
    img_2d = np.ones((500, 500), dtype=np.uint8) * 120
    tensor_3ch = resize_and_expand_channels(img_2d, target_size=(224, 224))

    assert isinstance(tensor_3ch, torch.Tensor)
    assert tensor_3ch.shape == (3, 224, 224)
    assert tensor_3ch.dtype == torch.uint8


def test_run_preprocessing_and_caching():
    stats = run_preprocessing_and_caching(batch_size=100)

    paths = get_paths()
    cache_dir = paths.output_root / "cache"
    log_file = paths.output_root / "preprocessing_log.md"
    plot_file = paths.output_root / "dicom_vs_processed_comparison.png"

    assert log_file.exists()
    assert plot_file.exists()
    assert cache_dir.exists()

    cache_files = list(cache_dir.glob("*.pt"))
    assert len(cache_files) > 0

    # Load first cached batch file and verify tensor structure
    batch_data = torch.load(str(cache_files[0]), weights_only=False)
    assert "images" in batch_data
    assert "labels" in batch_data
    assert "patient_ids" in batch_data
    assert batch_data["images"].dim() == 4  # [B, 3, 224, 224]
    assert batch_data["images"].shape[1:] == (3, 224, 224)
