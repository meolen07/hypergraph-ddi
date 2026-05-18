"""Base utilities for link prediction models."""

from __future__ import annotations

import torch
import torch.nn as nn


def pair_score(
    z: torch.Tensor,
    drug_a: torch.Tensor,
    drug_b: torch.Tensor,
    scorer: str = "dot",
) -> torch.Tensor:
    """Compute interaction scores for drug pairs from node embeddings."""
    za = z[drug_a]
    zb = z[drug_b]
    if scorer == "dot":
        return (za * zb).sum(dim=-1)
    if scorer == "cosine":
        za = nn.functional.normalize(za, dim=-1)
        zb = nn.functional.normalize(zb, dim=-1)
        return (za * zb).sum(dim=-1)
    if scorer == "mlp":
        raise NotImplementedError("Use model-specific MLP head")
    raise ValueError(f"Unknown scorer: {scorer}")


class PairPredictor(nn.Module):
    """MLP head on concatenated pair embeddings."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        drug_a: torch.Tensor,
        drug_b: torch.Tensor,
    ) -> torch.Tensor:
        za = z[drug_a]
        zb = z[drug_b]
        return self.mlp(torch.cat([za, zb], dim=-1)).squeeze(-1)
