import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict

def calculate_uncertainty_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    labels = results["labels"]
    mean_preds = results["mean_predictions"]
    entropy = results["entropy"]

    # Ensure mean_preds is a 1D array of probabilities for the positive class
    if mean_preds.ndim == 2 and mean_preds.shape[1] == 2:
        mean_preds = mean_preds[:, 1]

    # Calculate metrics without extra indexing
    auc = roc_auc_score(labels, mean_preds)
    avg_precision = average_precision_score(labels, mean_preds)
    avg_entropy = entropy.mean()

    return {
        "AUC": auc,
        "Average Precision": avg_precision,
        "Average Entropy": avg_entropy
    }
