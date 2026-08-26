import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Set up project root and imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import get_dataloaders
from src.models.simple_cnn import SimpleCNN


def main():
    config_path = PROJECT_ROOT / "configs" / "cnn.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 64)
    lr = config.get("learning_rate", 0.001)
    device_setting = config.get("device", "auto")

    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)

    print(f"Using device: {device}")

    output_dir = PROJECT_ROOT / "outputs"
    data_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, data_dir=str(data_dir))
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
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

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "simple_cnn_best.pt")

    log_df = pd.DataFrame(history)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}. Model saved to {output_dir / 'simple_cnn_best.pt'}")


if __name__ == "__main__":
    main()
