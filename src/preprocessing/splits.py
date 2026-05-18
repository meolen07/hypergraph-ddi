"""Train/val/test splits without edge leakage across splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.preprocessing.loaders import InteractionRecord


def _edge_key(nodes: List[int]) -> frozenset:
    return frozenset(nodes)


def split_interactions(
    interactions: List[InteractionRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[InteractionRecord]]:
    """
    Random edge split: each hyperedge appears in exactly one split.

    Prevents leakage of the same interaction across train/val/test.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(interactions))
    rng.shuffle(indices)

    n = len(interactions)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": [interactions[i] for i in train_idx],
        "val": [interactions[i] for i in val_idx],
        "test": [interactions[i] for i in test_idx],
    }


def save_splits(splits: Dict[str, List[InteractionRecord]], out_dir: str | Path) -> None:
    """Save splits as JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        serializable = [{"nodes": n, "label": l} for n, l in data]
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f)


def load_splits(splits_dir: str | Path) -> Dict[str, List[InteractionRecord]]:
    """Load splits from JSON files."""
    splits_dir = Path(splits_dir)
    splits: Dict[str, List[InteractionRecord]] = {}
    for path in splits_dir.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        splits[path.stem] = [(d["nodes"], d["label"]) for d in data]
    return splits


def get_train_hyperedges(
    train_interactions: List[InteractionRecord],
) -> List[List[int]]:
    """Hyperedges used to build the hypergraph (train only)."""
    return [nodes for nodes, _ in train_interactions]
