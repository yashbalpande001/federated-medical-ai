import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rsna_dataset import get_rsna_dataloaders
from src.utils.env_config import get_paths


def visualize_batch():
    paths = get_paths()
    train_loader, val_loader, test_loader, summary = get_rsna_dataloaders(override_batch_size=8)

    print("Fetching sample batch from train_loader...")
    images, labels = next(iter(train_loader))

    class_map = {0: "No Pneumonia (0)", 1: "Pneumonia / Lung Opacity (1)"}

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("RSNA Chest X-Ray Sample Batch", fontsize=14, fontweight="bold")

    # Unnormalize ImageNet normalization for visualization: img = std * tensor + mean
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img_tensor = images[i] * std + mean
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            img_np = img_tensor.permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            label_name = class_map.get(labels[i].item(), f"Class {labels[i].item()}")
            ax.set_title(label_name, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    output_path = paths.output_root / "rsna_sample_batch.png"
    plt.savefig(output_path, dpi=150)
    print(f"Sample visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    visualize_batch()
