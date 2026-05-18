"""Inference utilities for trained DDI models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn

from src.baselines.factory import build_model, make_trainer_forward
from src.utils.config import load_config, resolve_paths


def load_checkpoint(
    checkpoint_path: str | Path,
    cfg: Dict[str, Any] | None = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if cfg is None:
        cfg = ckpt.get("cfg", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


@torch.no_grad()
def predict_pairs(
    model: nn.Module,
    forward_fn,
    pairs: List[Tuple[int, int]],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Predict interaction probabilities for drug pairs."""
    model.eval()
    probs = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        drug_a = torch.tensor([p[0] for p in chunk], dtype=torch.long, device=device)
        drug_b = torch.tensor([p[1] for p in chunk], dtype=torch.long, device=device)
        logits = forward_fn({"drug_a": drug_a, "drug_b": drug_b, "label": torch.zeros(len(chunk))})
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs)
