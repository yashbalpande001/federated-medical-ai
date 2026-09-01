import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import flwr as fl
from flwr.client import Client, NumPyClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rsna_dataset import RSNADataset
from src.models.rsna_resnet import RSNABaselineResNet18
from src.training.focal_loss import BinaryFocalLoss
from src.utils.env_config import get_paths


class RSNAFlowerClient(NumPyClient):
    """
    Flower Client for local client training on RSNA dataset partitions.
    """

    def __init__(
        self,
        client_id: int,
        partition_csv: Path,
        images_dir: Path,
        batch_size: int = 32,
        lr: float = 1e-4,
        epochs_per_round: int = 1,
        device: Optional[torch.device] = None,
        is_synthetic: bool = False,
    ):
        self.client_id = client_id
        self.partition_csv = Path(partition_csv)
        self.images_dir = Path(images_dir)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs_per_round = epochs_per_round
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_synthetic = is_synthetic

        # Load client dataset partition
        if self.partition_csv.exists():
            self.df = pd.read_csv(self.partition_csv)
        else:
            # Fallback for dry-runs if partition missing
            self.df = pd.DataFrame(
                {
                    "patientId": [f"synth_c{client_id}_{i}" for i in range(20)],
                    "Target": [0 if i < 15 else 1 for i in range(20)],
                }
            )
            self.is_synthetic = True

        self.dataset = RSNADataset(self.df, images_dir=self.images_dir, is_synthetic=self.is_synthetic)
        self.train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        # Model & Loss (Step 6 Focal Loss)
        self.model = RSNABaselineResNet18(pretrained=True, freeze_backbone=False).to(self.device)
        self.criterion = BinaryFocalLoss(gamma=2.0, alpha=0.75)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Returns local model weights as a list of numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets local model weights from a list of numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Trains local model on client's partition for epochs_per_round."""
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for _ in range(self.epochs_per_round):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

        avg_loss = total_loss / max(1, total_samples)
        updated_params = self.get_parameters(config={})
        return updated_params, len(self.dataset), {"train_loss": float(avg_loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluates local model on client's local partition."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                total_loss += loss.item() * len(labels)
                correct += int((preds == labels).sum().item())
                total += len(labels)

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return float(avg_loss), total, {"accuracy": float(accuracy)}
