import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet architectures.
    Hooks into the specified target layer to capture activations and compute gradients.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch classification model (e.g., RSNABaselineResNet18).
            target_layer: Module instance inside the model (e.g., model.resnet.layer4[-1]).
        """
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        """Forward hook callback to capture layer activations."""
        self.activations = output.detach()

    def _save_gradients(
        self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]
    ) -> None:
        """Backward hook callback to capture gradients w.r.t layer output."""
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates a 2D normalized Grad-CAM heatmap [0, 1] for an input image tensor.

        Args:
            input_tensor: Image tensor of shape [1, 3, H, W] or [3, H, W].
            target_class: Index of target class (optional for binary logit).

        Returns:
            2D numpy array heatmap scaled to [0.0, 1.0] of shape (H, W).
        """
        self.model.eval()

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Ensure gradient tracking even if caller is in torch.no_grad() context
        with torch.enable_grad():
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            h_orig, w_orig = input_tensor.shape[2], input_tensor.shape[3]

            # Forward pass
            self.model.zero_grad()
            logits = self.model(input_tensor)

            if logits.dim() == 0:
                score = logits
            elif logits.dim() == 1:
                score = logits[0]
            else:
                score = logits[0, 0] if logits.shape[1] == 1 else logits[0, target_class or 0]

            # Backward pass to calculate gradients
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM failed to capture activations or gradients. Verify target_layer setup.")

        # Compute global average pooling of gradients: alpha_k = (1/Z) * sum_{i,j} (grad_k)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: [1, C, 1, 1]

        # Weighted combination of activation maps: sum_k (alpha_k * A_k)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # Shape: [1, 1, H_feat, W_feat]

        # Apply ReLU to keep only features that positively correlate with target score
        cam = F.relu(cam)

        # Interpolate CAM to match original image dimensions (H, W)
        cam = F.interpolate(cam, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-Max Normalization to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam_normalized = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam)

        return cam_normalized.astype(np.float32)

    def remove_hooks(self) -> None:
        """Removes forward and backward hooks from target layer."""
        if hasattr(self, "forward_hook") and self.forward_hook:
            self.forward_hook.remove()
        if hasattr(self, "backward_hook") and self.backward_hook:
            self.backward_hook.remove()

    def __del__(self):
        self.remove_hooks()
