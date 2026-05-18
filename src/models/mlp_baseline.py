"""MLP baseline without graph structure."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import PairPredictor


class MLPBaseline(nn.Module):
    """Two-layer MLP on node features without message passing."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.predictor = PairPredictor(out_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        drug_a: torch.Tensor,
        drug_b: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x)
        return self.predictor(z, drug_a, drug_b)
