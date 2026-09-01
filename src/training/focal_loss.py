import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for handling severe class imbalance in binary classification.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight factor for positive class (e.g. 0.778 for 22.24% positive class).
               If alpha is float, positive class gets alpha, negative gets (1 - alpha).
        gamma: Focusing parameter for hard examples (default = 2.0).
        reduction: 'mean' (default) or 'sum' or 'none'.
    """

    def __init__(self, alpha: float = 0.778, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model raw output logits [B] or [B, 1]
            targets: Binary targets [B] or [B, 1] (values 0.0 or 1.0)
        Returns:
            Scalar loss tensor
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        # p_t is probability corresponding to true class target
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)

        # alpha_t weights positive class by alpha and negative class by (1 - alpha)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Focal factor: (1 - p_t)^gamma
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
