"""GAT baseline via PyTorch Geometric."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from src.models.base import PairPredictor


class GATBaseline(nn.Module):
    """Graph Attention Network for DDI link prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if n_layers == 1:
            self.convs.append(
                GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(
                GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
            )
            for _ in range(n_layers - 2):
                self.convs.append(
                    GATConv(
                        hidden_dim * heads,
                        hidden_dim,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
            self.convs.append(
                GATConv(
                    hidden_dim * heads,
                    out_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )
        self.predictor = PairPredictor(out_dim)

    def encode(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        data: Data,
        drug_a: torch.Tensor,
        drug_b: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(data)
        return self.predictor(z, drug_a, drug_b)
