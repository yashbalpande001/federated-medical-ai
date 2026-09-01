import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rsna_resnet import RSNABaselineResNet18
from src.training.focal_loss import BinaryFocalLoss
from src.utils.env_config import get_paths


class SaveAndEvaluateFedAvg(FedAvg):
    """
    Custom FedAvg Strategy extending Flower FedAvg:
    1. Saves global model weights to disk after EVERY round.
    2. Evaluates the global model on the centralized held-out test dataset after EVERY round.
    """

    def __init__(
        self,
        test_loader: DataLoader,
        checkpoints_dir: Path,
        device: Optional[torch.device] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.test_loader = test_loader
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_model = RSNABaselineResNet18(pretrained=False, freeze_backbone=False).to(self.device)
        self.criterion = BinaryFocalLoss(gamma=2.0, alpha=0.75)
        self.eval_history: List[Dict[str, float]] = []

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluates global model on the centralized test loader after each round.
        """
        # Convert Parameters to PyTorch state_dict
        weights_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        params_dict = zip(self.eval_model.state_dict().keys(), weights_ndarrays)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.eval_model.load_state_dict(state_dict, strict=True)

        # Save Checkpoint after every round
        checkpoint_path = self.checkpoints_dir / f"fedavg_round_{server_round}.pt"
        best_model_path = self.checkpoints_dir / "best_fedavg_model.pt"

        torch.save(
            {
                "round": server_round,
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

        # Compute AUC if test set has both classes
        if len(np.unique(all_targets)) > 1:
            auc = float(roc_auc_score(all_targets, all_probs))
        else:
            auc = 0.5

        # Save best model checkpoint based on test AUC
        prev_best_auc = max([h["auc"] for h in self.eval_history], default=0.0)
        if auc >= prev_best_auc:
            torch.save(
                {
                    "round": server_round,
                    "model_state_dict": self.eval_model.state_dict(),
                    "auc": auc,
                },
                best_model_path,
            )

        metrics = {
            "test_loss": float(avg_loss),
            "test_accuracy": float(accuracy),
            "test_auc": float(auc),
        }
        self.eval_history.append({"round": server_round, "loss": avg_loss, "accuracy": accuracy, "auc": auc})

        print(
            f"--> [Round {server_round:02d}] Centralized Test Evaluation | "
            f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | AUC: {auc:.4f}"
        )

        return float(avg_loss), metrics
