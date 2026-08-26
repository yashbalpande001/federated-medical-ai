import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_config import get_paths, get_environment
from src.data.cifar10 import get_dataloaders_resnet
from src.models.resnet_transfer import build_resnet18


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 via Transfer Learning")
    parser.add_argument("--mode", type=str, choices=["frozen", "finetune"], default="frozen",
                        help="Transfer learning mode: 'frozen' (feature extractor) or 'finetune' (end-to-end)")
    args = parser.parse_args()
    mode = args.mode

    # Retrieve platform-independent paths via env_config
    paths = get_paths()
    env_name = get_environment()
    config_path = paths["config_root"] / "resnet.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 2)

    # Learning Rate Selection:
    # We use a lower learning rate for 'finetune' mode (1e-4) than 'frozen' mode (1e-3).
    # Rationale: Small parameter updates during fine-tuning protect the rich pre-trained ImageNet
    # feature representations across all backbone layers from catastrophic forgetting or gradient corruption,
    # whereas in frozen mode only the newly initialized final classification head is being trained.
    if mode == "frozen":
        lr = config.get("frozen_lr", 0.001)
    else:
        lr = config.get("finetune_lr", 0.0001)

    # Device selection: use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== ResNet-18 Transfer Learning ({mode.upper()} mode) ===")
    print(f"Environment: {env_name} | Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"Data Root       : {paths['data_root']}")
    print(f"Output Root     : {paths['output_root']}")
    print(f"Checkpoint Root : {paths['checkpoint_root']}\n")

    train_loader, val_loader, _ = get_dataloaders_resnet(
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=str(paths["data_root"])
    )
    model = build_resnet18(num_classes=10, mode=mode).to(device)

    criterion = nn.CrossEntropyLoss()
    # Filter optimizer parameters to only those with requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    best_val_acc = 0.0
    history = []

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        start_epoch_time = time.time()

        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 200 == 0 or batch_idx == total_batches:
                current_acc = correct / total
                print(f"  [Epoch {epoch:02d}] Batch {batch_idx:04d}/{total_batches:04d} - Loss: {loss.item():.4f} - Acc: {current_acc:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_epoch_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({epoch_time:.2f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_file = paths["checkpoint_root"] / f"resnet_{mode}_best.pt"
            torch.save(model.state_dict(), checkpoint_file)

    total_training_time = time.time() - start_total_time

    log_file = paths["output_root"] / f"resnet_{mode}_log.csv"
    log_df = pd.DataFrame(history)
    log_df.to_csv(log_file, index=False)

    print("\n=========================================================================")
    print(f"Training Complete ({mode.upper()} mode)")
    print(f"Total Training Time : {total_training_time:.2f} seconds ({total_training_time/60:.2f} mins)")
    print(f"Best Validation Acc : {best_val_acc:.4f}")
    print(f"Log saved to        : {log_file}")
    print(f"Checkpoint saved to : {paths['checkpoint_root'] / f'resnet_{mode}_best.pt'}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
