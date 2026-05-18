"""Model factory and forward function dispatch."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.hgnn import HGNN
from src.models.mlp_baseline import MLPBaseline
from src.models.gcn import GCNBaseline
from src.models.gat import GATBaseline
from src.models.graphsage import GraphSAGEBaseline


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate model from config."""
    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "hgnn").lower()
    in_dim = int(mcfg.get("in_dim", 64))
    hidden_dim = int(mcfg.get("hidden_dim", 128))
    out_dim = int(mcfg.get("out_dim", 64))
    n_layers = int(mcfg.get("n_layers", 2))
    dropout = float(mcfg.get("dropout", 0.1))

    if name == "hgnn":
        return HGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            use_attention=bool(mcfg.get("use_attention", False)),
            dropout=dropout,
        )
    if name == "mlp":
        return MLPBaseline(in_dim, hidden_dim, out_dim, dropout)
    if name == "gcn":
        return GCNBaseline(in_dim, hidden_dim, out_dim, n_layers, dropout)
    if name == "gat":
        return GATBaseline(
            in_dim, hidden_dim, out_dim, n_layers,
            heads=int(mcfg.get("heads", 4)),
            dropout=dropout,
        )
    if name == "graphsage":
        return GraphSAGEBaseline(in_dim, hidden_dim, out_dim, n_layers, dropout)
    raise ValueError(f"Unknown model: {name}")


def get_forward_fn(
    model_name: str,
    train_extra: Dict[str, Any],
) -> Callable[..., torch.Tensor]:
    """
    Return forward function closing over static graph/hypergraph tensors.

    train_extra may contain: x, H, node_deg, edge_deg, pyg_data
    """
    name = model_name.lower()
    x = train_extra["x"]
    device = x.device

    if name == "hgnn":
        H = train_extra["H"]
        node_deg = train_extra["node_deg"]
        edge_deg = train_extra["edge_deg"]
        model = None  # set by trainer

        def forward_hgnn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return model(x, H, node_deg, edge_deg, batch["drug_a"], batch["drug_b"])

        return lambda batch, m=None: forward_hgnn(m, batch)  # noqa: E731 — patched in trainer

    if name in ("gcn", "gat", "graphsage"):
        pyg_data = train_extra["pyg_data"]

        def forward_graph(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return model(pyg_data, batch["drug_a"], batch["drug_b"])

        return lambda batch, m=None: forward_graph(m, batch)

    if name == "mlp":

        def forward_mlp(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return model(x, batch["drug_a"], batch["drug_b"])

        return lambda batch, m=None: forward_mlp(m, batch)

    raise ValueError(f"Unknown model: {name}")


def make_trainer_forward(model: nn.Module, model_name: str, train_extra: Dict[str, Any]) -> Callable:
    """Bind model into forward callable for Trainer."""
    name = model_name.lower()
    x = train_extra["x"]
    if name == "hgnn":
        H, node_deg, edge_deg = train_extra["H"], train_extra["node_deg"], train_extra["edge_deg"]

        def fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return model(x, H, node_deg, edge_deg, batch["drug_a"], batch["drug_b"])

        return fn
    if name in ("gcn", "gat", "graphsage"):
        pyg_data = train_extra["pyg_data"]

        def fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return model(pyg_data, batch["drug_a"], batch["drug_b"])

        return fn

    def fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model(x, batch["drug_a"], batch["drug_b"])

    return fn
