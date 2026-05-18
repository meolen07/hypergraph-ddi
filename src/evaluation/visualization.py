"""Training curves, ROC plots, and optional t-SNE."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def plot_training_curves(
    history: Dict[str, List[float]],
    out_path: str | Path,
) -> None:
    """Plot train/val loss and validation AUC curves."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get("train_loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(history.get("val_roc_auc", []), label="ROC-AUC")
    axes[1].plot(history.get("val_pr_auc", []), label="PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].set_title("Validation metrics")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    out_path: str | Path,
) -> None:
    """Save ROC curve figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    out_path: str | Path,
    perplexity: float = 30.0,
    seed: int = 42,
) -> None:
    """Optional t-SNE visualization of node embeddings."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = embeddings.shape[0]
    perp = min(perplexity, max(1, n - 1))
    z = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(embeddings)
    plt.figure(figsize=(7, 6))
    if labels is not None:
        sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=labels, palette="tab10", s=40, legend=False)
    else:
        plt.scatter(z[:, 0], z[:, 1], s=40, alpha=0.7)
    plt.title("t-SNE of drug embeddings")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
