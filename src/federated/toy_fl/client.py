import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.client import NumPyClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ToyNet(nn.Module):
    """Simple 2-layer PyTorch MLP model for toy binary classification."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc2(self.relu(self.fc1(x)))
        return logits.squeeze(-1)


def get_toy_data(
    client_id: int, num_samples: int = 300, n_features: int = 20, seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Generates synthetic dataset for a specific virtual client."""
    X, y = make_classification(
        n_samples=num_samples,
        n_features=n_features,
        n_informative=12,
        n_classes=2,
        random_state=seed + client_id,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed + client_id)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader


class ToyFlowerClient(NumPyClient):
    """Standard Flower NumPyClient for local model training and evaluation."""

    def __init__(self, client_id: int, epochs_per_round: int = 2, lr: float = 0.01):
        self.client_id = client_id
        self.epochs_per_round = epochs_per_round
        self.lr = lr
        self.model = ToyNet()
        self.train_loader, self.val_loader = get_toy_data(client_id)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Returns model weights as a list of numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets model weights from a list of numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Trains model on client's local training data."""
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for _ in range(self.epochs_per_round):
            for x_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)

        avg_loss = total_loss / max(1, total_samples)
        updated_params = self.get_parameters(config={})
        return updated_params, len(self.train_loader.dataset), {"train_loss": float(avg_loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluates model on client's local validation data."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                total_loss += loss.item() * len(y_batch)
                correct += int((preds == y_batch).sum().item())
                total += len(y_batch)

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return float(avg_loss), total, {"accuracy": float(accuracy)}
