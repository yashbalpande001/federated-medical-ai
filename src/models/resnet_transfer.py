import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes: int = 10, mode: str = "frozen") -> nn.Module:
    """
    Build ResNet-18 for Transfer Learning on CIFAR-10 or custom datasets.

    Args:
        num_classes (int): Number of output classes (default: 10).
        mode (str): 'frozen' (feature extractor with frozen backbone) or
                    'finetune' (end-to-end training of all layers).

    Returns:
        nn.Module: Configured ResNet-18 model.
    """
    if mode not in ["frozen", "finetune"]:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'frozen' or 'finetune'.")

    # Load ResNet-18 with default ImageNet weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if mode == "frozen":
        # Freeze all backbone layers
        for param in model.parameters():
            param.requires_grad = False
    elif mode == "finetune":
        # Ensure all backbone layers are trainable
        for param in model.parameters():
            param.requires_grad = True

    # Replace final fully connected layer for target dataset classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # The new linear layer's parameters are unfrozen (requires_grad=True) by default

    return model
