import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import GradCAM
from src.models.rsna_resnet import RSNABaselineResNet18


def test_gradcam_heatmap_generation():
    """Test that GradCAM generates normalized 2D heatmaps matching original image dimensions."""
    model = RSNABaselineResNet18(pretrained=False)
    target_layer = model.resnet.layer4[-1]

    grad_cam = GradCAM(model, target_layer)

    # Input tensor shape: [1, 3, 224, 224]
    input_tensor = torch.randn(1, 3, 224, 224)

    heatmap = grad_cam.generate_heatmap(input_tensor)

    # Assertions
    assert isinstance(heatmap, np.ndarray), "Heatmap should be a numpy ndarray"
    assert heatmap.shape == (224, 224), f"Expected shape (224, 224), got {heatmap.shape}"
    assert heatmap.dtype == np.float32, f"Expected float32, got {heatmap.dtype}"
    assert np.min(heatmap) >= 0.0, "Heatmap min should be >= 0.0"
    assert np.max(heatmap) <= 1.0, "Heatmap max should be <= 1.0"

    grad_cam.remove_hooks()


def test_gradcam_3d_input_tensor():
    """Test GradCAM handling 3D input tensor [3, H, W]."""
    model = RSNABaselineResNet18(pretrained=False)
    target_layer = model.resnet.layer4[-1]

    grad_cam = GradCAM(model, target_layer)
    input_tensor = torch.randn(3, 224, 224)

    heatmap = grad_cam.generate_heatmap(input_tensor)

    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0

    grad_cam.remove_hooks()
