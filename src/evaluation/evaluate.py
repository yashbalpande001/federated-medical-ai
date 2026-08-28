import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_config import get_paths
from src.data.cifar10 import get_dataloaders
from src.models.simple_cnn import SimpleCNN


def main():
    paths = get_paths()
    output_dir = paths["output_root"]
    data_dir = paths["data_root"]
    checkpoint_path = output_dir / "simple_cnn_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model on device: {device}")

    _, _, test_loader = get_dataloaders(batch_size=64, data_dir=str(data_dir))

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('CIFAR-10 Confusion Matrix')
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved evaluation metrics to {output_dir / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
