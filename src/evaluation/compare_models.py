import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_config import get_paths
from src.models.simple_cnn import SimpleCNN
from src.models.resnet_transfer import build_resnet18


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    paths = get_paths()
    output_dir = paths["output_root"]

    log_files = {
        "Simple CNN": output_dir / "training_log.csv",
        "ResNet18 (Frozen)": output_dir / "resnet_frozen_log.csv",
        "ResNet18 (Finetune)": output_dir / "resnet_finetune_log.csv"
    }

    # Instantiate models to compute exact parameter counts
    models = {
        "Simple CNN": SimpleCNN(),
        "ResNet18 (Frozen)": build_resnet18(num_classes=10, mode="frozen"),
        "ResNet18 (Finetune)": build_resnet18(num_classes=10, mode="finetune")
    }

    plt.figure(figsize=(10, 6))
    summary_data = []

    print("\n=========================================================================================")
    print("                              MODEL COMPARISON SUMMARY                                   ")
    print("=========================================================================================")

    for name, filepath in log_files.items():
        model = models[name]
        total_params, trainable_params = count_parameters(model)

        if not filepath.exists():
            print(f"[{name}] Log file not found at {filepath}. Skipping from comparison.")
            continue

        df = pd.read_csv(filepath)
        epochs = df["epoch"].tolist()
        val_acc = df["val_acc"].tolist()

        # Calculate average epoch time if present
        if "epoch_time" in df.columns:
            avg_epoch_time = df["epoch_time"].mean()
            time_str = f"{avg_epoch_time:.2f}s / epoch"
        else:
            time_str = "N/A"

        best_val_acc = max(val_acc)
        final_val_acc = val_acc[-1]

        plt.plot(epochs, val_acc, marker='o', linewidth=2, label=f"{name} (Best: {best_val_acc:.4f})")

        summary_data.append({
            "Model": name,
            "Best Val Acc": f"{best_val_acc:.4f}",
            "Final Val Acc": f"{final_val_acc:.4f}",
            "Total Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Avg Epoch Time": time_str
        })

    if not summary_data:
        print("No training logs found in outputs/. Please run training scripts first.")
        return

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=========================================================================================\n")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title("CIFAR-10 Model Comparison: Validation Accuracy per Epoch", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
