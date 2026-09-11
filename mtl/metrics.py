"""Evaluation metrics.

Regression: RMSE, MAE, R^2, Pearson correlation coefficient.
Classification (active if pIC50 >= 5.0): ROC-AUC, PR-AUC.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 5.0) -> Dict[str, float]:
    """ROC-AUC and PR-AUC for active/inactive labels (active: pIC50 >= threshold)."""
    labels = (np.asarray(y_true) >= threshold).astype(int)
    probs = np.asarray(y_pred)
    n_pos = int(labels.sum())
    out: Dict[str, float] = {}
    if n_pos >= 1 and n_pos < len(labels):
        out["ROC-AUC"] = float(roc_auc_score(labels, probs))
        out["PR-AUC"] = float(average_precision_score(labels, probs))
    else:
        out["ROC-AUC"] = float("nan")
        out["PR-AUC"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, R^2, Pearson r."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson r": pearson_r(y_true, y_pred),
    }


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 5.0) -> Dict[str, float]:
    """Full metric set used for reporting."""
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(classification_metrics(y_true, y_pred, threshold))
    return metrics
