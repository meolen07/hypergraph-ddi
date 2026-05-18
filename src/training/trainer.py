"""Training loop with BCE loss, Adam, early stopping, and checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


class Trainer:
    """
    YAML-config-driven trainer for DDI models.

    Supports HGNN (hypergraph) and graph/MLP baselines via forward_fn.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        device: torch.device,
        forward_fn: Callable[..., torch.Tensor],
        train_extra: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.forward_fn = forward_fn
        self.train_extra = train_extra or {}
        self.logger = logger

        tcfg = cfg.get("training", {})
        self.lr = float(tcfg.get("lr", 1e-3))
        self.weight_decay = float(tcfg.get("weight_decay", 1e-5))
        self.epochs = int(tcfg.get("epochs", 100))
        self.patience = int(tcfg.get("early_stopping_patience", 15))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_pr_auc": [],
        }
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_metric = float("-inf")
        self.best_epoch = 0

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            logits = self.forward_fn(batch)
            loss = self.criterion(logits, batch["label"])
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.forward_fn(batch)
            loss = self.criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        metrics = compute_metrics(labels_cat.numpy(), logits_cat.numpy())
        metrics["loss"] = total_loss / max(n, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str | Path,
    ) -> Dict[str, Any]:
        """Train with early stopping; save best checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        patience_counter = 0
        monitor = self.cfg.get("training", {}).get("monitor_metric", "roc_auc")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_score = val_metrics.get(monitor, val_metrics["roc_auc"])

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])

            self._log(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"val_roc_auc={val_metrics['roc_auc']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f}"
            )

            if val_score > self.best_val_metric:
                self.best_val_metric = val_score
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.best_state,
                        "val_metrics": val_metrics,
                        "cfg": self.cfg,
                    },
                    checkpoint_dir / "best.pt",
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self._log(f"Early stopping at epoch {epoch}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "history": self.history,
        }
