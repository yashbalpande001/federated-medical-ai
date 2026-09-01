import sys
from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.focal_loss import BinaryFocalLoss


def test_focal_loss_initialization():
    criterion = BinaryFocalLoss(alpha=0.778, gamma=2.0)
    assert criterion.alpha == 0.778
    assert criterion.gamma == 2.0


def test_focal_loss_forward_shape_and_grad():
    criterion = BinaryFocalLoss(alpha=0.778, gamma=2.0)
    logits = torch.randn(8, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.float32)

    loss = criterion(logits, targets)

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_focal_loss_perfect_prediction():
    criterion = BinaryFocalLoss(alpha=0.778, gamma=2.0)
    # High positive logit for y=1, low negative logit for y=0
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = criterion(logits, targets)
    assert loss.item() < 0.01  # Loss should be near 0 for confident correct predictions
