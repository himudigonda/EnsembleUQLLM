import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict

def calculate_uncertainty_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    labels = results["labels"]
    mean_preds = results["mean_predictions"]
    entropy = results["entropy"]

    auc = roc_auc_score(labels, mean_preds[:, 1])
    avg_precision = average_precision_score(labels, mean_preds[:, 1])
    avg_entropy = entropy.mean()

    return {
        "AUC": auc,
        "Average Precision": avg_precision,
        "Average Entropy": avg_entropy
    }
