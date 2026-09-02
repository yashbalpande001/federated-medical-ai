import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rsna_resnet import RSNABaselineResNet18
from src.training.focal_loss import BinaryFocalLoss
from src.utils.env_config import get_paths


class SaveAndEvaluateFedProx(FedAvg):
    """
    Custom FedProx Strategy extending Flower FedAvg:
    1. Saves global model weights to disk after EVERY round for a specific mu setting.
    2. Evaluates the global model on the centralized held-out test dataset (AUC, F1, Recall, Precision).
    """

    def __init__(
        self,
        test_loader: DataLoader,
        checkpoints_dir: Path,
        mu: float = 0.01,
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.test_loader = test_loader
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.mu = mu
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_model = RSNABaselineResNet18(pretrained=False, freeze_backbone=False).to(self.device)
        self.criterion = BinaryFocalLoss(gamma=2.0, alpha=0.75)
        self.eval_history: List[Dict[str, float]] = []

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluates global model on the centralized test loader after each round.
        Computes Test Loss, Accuracy, AUC, F1, Recall, Precision.
        """
        weights_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        params_dict = zip(self.eval_model.state_dict().keys(), weights_ndarrays)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.eval_model.load_state_dict(state_dict, strict=True)

        # Save Checkpoint after every round
        checkpoint_path = self.checkpoints_dir / f"fedprox_mu_{self.mu}_round_{server_round}.pt"
        best_model_path = self.checkpoints_dir / f"best_fedprox_mu_{self.mu}_model.pt"

        torch.save(
            {
                "round": server_round,
                "mu": self.mu,
                "model_state_dict": self.eval_model.state_dict(),
            },
            checkpoint_path,
        )

        self.eval_model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                logits = self.eval_model(images)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)

                total_loss += loss.item() * len(labels)
                all_targets.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        total_samples = max(1, len(all_targets))
        avg_loss = total_loss / total_samples

        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        all_preds = (all_probs >= 0.5).astype(float)

        accuracy = float((all_preds == all_targets).mean())

        # Compute classification metrics
        if len(np.unique(all_targets)) > 1:
            auc = float(roc_auc_score(all_targets, all_probs))
        else:
            auc = 0.5

        precision = float(precision_score(all_targets, all_preds, zero_division=0))
        recall = float(recall_score(all_targets, all_preds, zero_division=0))
        f1 = float(f1_score(all_targets, all_preds, zero_division=0))

        # Save best model checkpoint based on test AUC
        prev_best_auc = max([h["auc"] for h in self.eval_history], default=0.0)
        if auc >= prev_best_auc:
            torch.save(
                {
                    "round": server_round,
                    "mu": self.mu,
                    "model_state_dict": self.eval_model.state_dict(),
                    "auc": auc,
                    "f1": f1,
                    "recall": recall,
                    "precision": precision,
                },
                best_model_path,
            )

        rec = {
            "round": server_round,
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "auc": float(auc),
            "f1": float(f1),
            "recall": float(recall),
            "precision": float(precision),
        }
        self.eval_history.append(rec)

        print(
            f"--> [FedProx mu={self.mu} | Round {server_round:02d}] Test Evaluation | "
            f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}"
        )

        return float(avg_loss), {"test_loss": float(avg_loss), "test_auc": float(auc), "test_f1": float(f1)}
