"""Evaluation metrics for DDI prediction."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> float:
    """Precision@K over top-K scored samples."""
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top_idx]))


def compute_metrics(
    y_true: np.ndarray,
    y_logits: np.ndarray,
    threshold: float = 0.5,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute ROC-AUC, PR-AUC, Precision, Recall, F1, and Precision@K.

    Parameters
    ----------
    y_true : binary labels
    y_logits : model logits (sigmoid applied internally)
    """
    y_true = np.asarray(y_true).astype(int).flatten()
    y_prob = _sigmoid(np.asarray(y_logits).flatten())

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    k_values = k_values or [10, 50, 100]
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_prob, k=k)

    return metrics


def get_roc_curve_data(
    y_true: np.ndarray,
    y_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FPR, TPR for ROC plotting."""
    y_prob = _sigmoid(np.asarray(y_logits).flatten())
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr
