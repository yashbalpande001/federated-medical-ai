import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet_transfer import build_resnet18


def test_resnet_frozen_mode():
    model = build_resnet18(num_classes=10, mode="frozen")
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    
    # Check backbone parameters are frozen
    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True, f"fc parameter {name} should be trainable"
        else:
            assert param.requires_grad is False, f"Backbone parameter {name} should be frozen"


def test_resnet_finetune_mode():
    model = build_resnet18(num_classes=10, mode="finetune")
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    
    # Check all parameters are trainable
    for name, param in model.named_parameters():
        assert param.requires_grad is True, f"Parameter {name} should be trainable in finetune mode"


def test_resnet_invalid_mode():
    with pytest.raises(ValueError):
        build_resnet18(num_classes=10, mode="invalid_mode")
