from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class RSNABaselineResNet18(nn.Module):
    """
    ResNet-18 Transfer Learning architecture for RSNA Chest X-Ray Binary Pneumonia Detection.
    Replaces default ImageNet head with a single output logit for binary classification.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False, num_classes: int = 1):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace final classification head
        in_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor batch [B, 3, 224, 224]
        Returns:
            Logits tensor [B, 1] or [B]
        """
        logits = self.resnet(x)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits

    def get_trainable_params_count(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

