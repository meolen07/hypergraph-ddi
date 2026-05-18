"""PyTorch Dataset for drug interaction prediction with efficient batching."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.loaders import InteractionRecord


class DDIDataset(Dataset):
    """
    Dataset of (drug_a, drug_b, label) pairs derived from hyperedges.

    For hyperedges with >2 drugs, expands to all pairs within the hyperedge
    for pair-level prediction (configurable via pair_only).
    """

    def __init__(
        self,
        samples: List[InteractionRecord],
        node_features: torch.Tensor,
        pair_only: bool = True,
    ) -> None:
        self.node_features = node_features
        self.pairs: List[Tuple[int, int, float]] = []
        for nodes, label in samples:
            nodes = sorted(set(nodes))
            if pair_only and len(nodes) >= 2:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        self.pairs.append((nodes[i], nodes[j], float(label)))
            elif len(nodes) == 2:
                self.pairs.append((nodes[0], nodes[1], float(label)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a, b, y = self.pairs[idx]
        return {
            "drug_a": torch.tensor(a, dtype=torch.long),
            "drug_b": torch.tensor(b, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.float32),
        }


def collate_ddi_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of pair samples."""
    return {
        "drug_a": torch.stack([b["drug_a"] for b in batch]),
        "drug_b": torch.stack([b["drug_b"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def build_node_features(
    n_nodes: int,
    feat_dim: int,
    seed: int = 42,
    use_identity: bool = True,
) -> torch.Tensor:
    """
  Build initial node features.

    If use_identity and n_nodes <= feat_dim, uses one-hot rows.
    Otherwise uses Xavier random features (common when no external descriptors).
    """
    rng = torch.Generator().manual_seed(seed)
    if use_identity and n_nodes <= feat_dim:
        x = torch.zeros(n_nodes, feat_dim)
        x[:, :n_nodes] = torch.eye(n_nodes)
        return x
    return torch.randn(n_nodes, feat_dim, generator=rng) * 0.1
