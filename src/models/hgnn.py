"""Hypergraph Neural Network: node -> hyperedge -> node message passing."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import PairPredictor


class HGNNLayer(nn.Module):
    """
    One HGNN layer with optional attention on hyperedge aggregation.

    Node -> hyperedge (aggregate nodes in each hyperedge)
    Hyperedge -> node (distribute hyperedge features back to nodes)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_attention: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.dropout = dropout
        self.node_lin = nn.Linear(in_dim, out_dim)
        self.edge_lin = nn.Linear(in_dim, out_dim)
        if use_attention:
            self.att_src = nn.Linear(out_dim, 1, bias=False)
            self.att_dst = nn.Linear(out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)

    def forward(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        node_deg: torch.Tensor,
        edge_deg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (n_nodes, in_dim)
        H : sparse COO (n_nodes, n_edges) incidence
        node_deg, edge_deg : normalization vectors
        """
        n_nodes, n_edges = H.shape
        device = x.device

        # Node -> hyperedge
        H_dense = H.coalesce()
        idx = H_dense.indices()
        vals = H_dense.values()

        # Gather node features for each incidence
        node_idx = idx[0]
        edge_idx = idx[1]
        node_feats = x[node_idx]  # (nnz, in_dim)

        edge_embed = torch.zeros(n_edges, x.size(-1), device=device)
        edge_embed.index_add_(0, edge_idx, node_feats)

        if self.use_attention:
            h_node = self.node_lin(x)
            attn_n = self.att_src(h_node)
            attn_e = attn_n[node_idx] + self.att_dst(self.edge_lin(edge_embed))[edge_idx]
            attn_w = torch.softmax(
                scatter_softmax_prep(attn_e.squeeze(-1), edge_idx, n_edges),
                dim=0,
            )
            edge_embed = torch.zeros(n_edges, x.size(-1), device=device)
            edge_embed.index_add_(0, edge_idx, node_feats * attn_w.unsqueeze(-1))
        else:
            edge_embed = edge_embed / edge_deg.unsqueeze(-1).clamp(min=1.0)

        edge_embed = self.edge_lin(edge_embed)
        edge_embed = F.relu(edge_embed)
        edge_embed = F.dropout(edge_embed, p=self.dropout, training=self.training)

        # Hyperedge -> node
        node_embed = torch.zeros(n_nodes, edge_embed.size(-1), device=device)
        node_embed.index_add_(0, node_idx, edge_embed[edge_idx])
        node_embed = node_embed / node_deg.unsqueeze(-1).clamp(min=1.0)
        node_embed = self.node_lin(x) + node_embed
        return F.relu(node_embed)


def scatter_softmax_prep(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-edge softmax over incident nodes (simplified)."""
    max_per_edge = torch.full((dim_size,), float("-inf"), device=scores.device)
    max_per_edge.scatter_reduce_(0, index, scores, reduce="amax", include_self=False)
    exp = torch.exp(scores - max_per_edge[index])
    sum_exp = torch.zeros(dim_size, device=scores.device)
    sum_exp.index_add_(0, index, exp)
    return exp / sum_exp[index].clamp(min=1e-8)


class HGNN(nn.Module):
    """Multi-layer HGNN with pair prediction head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            layers.append(
                HGNNLayer(
                    dims[i],
                    dims[i + 1],
                    use_attention=use_attention and i == n_layers - 1,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.predictor = PairPredictor(out_dim)

    def encode(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        node_deg: torch.Tensor,
        edge_deg: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, H, node_deg, edge_deg)
        return x

    def forward(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        node_deg: torch.Tensor,
        edge_deg: torch.Tensor,
        drug_a: torch.Tensor,
        drug_b: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, H, node_deg, edge_deg)
        return self.predictor(z, drug_a, drug_b)
