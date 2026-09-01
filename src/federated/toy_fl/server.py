import sys
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Metrics, Scalar
from flwr.server.strategy import FedAvg

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def weighted_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """
    Computes sample-weighted average metrics (accuracy, loss) across all reporting clients.
    """
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    metric_keys = metrics[0][1].keys() if metrics else []

    for key in metric_keys:
        weighted_sum = sum(num_samples * float(m[key]) for num_samples, m in metrics if key in m)
        aggregated[key] = round(weighted_sum / total_samples, 4)

    return aggregated


def create_fedavg_strategy(
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 3,
    min_evaluate_clients: int = 3,
    min_available_clients: int = 3,
) -> FedAvg:
    """Creates a configured Flower FedAvg strategy with metric aggregation callbacks."""
    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_metrics_aggregation,
        evaluate_metrics_aggregation_fn=weighted_metrics_aggregation,
    )
