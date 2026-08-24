import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simple_cnn import SimpleCNN


def test_simple_cnn_output_shape():
    model = SimpleCNN()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"
