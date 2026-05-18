#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.baselines.factory import build_model, make_trainer_forward
from src.datasets.ddi_dataset import DDIDataset, collate_ddi_batch, build_node_features
from src.datasets.graph_dataset import build_pyg_data, hyperedges_to_pair_edges
from src.evaluation.metrics import compute_metrics, get_roc_curve_data
from src.evaluation.visualization import plot_roc_curve
from src.inference.predict import load_checkpoint
from src.preprocessing.hypergraph import build_hypergraph
from src.preprocessing.negative_sampling import build_labeled_dataset
from src.preprocessing.splits import get_train_hyperedges, load_splits
from src.preprocessing.loaders import load_processed
from src.utils.config import load_config, resolve_paths
from src.utils.paths import processed_dir, splits_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DDI checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.environ.setdefault("HYPERGRAPH_DDI_ROOT", ROOT)
    cfg = resolve_paths(load_config(args.config), project_root=ROOT)
    device = torch.device(cfg.get("device", "cpu"))

    drugs_df, _ = load_processed(processed_dir(cfg))
    splits = load_splits(splits_dir(cfg))
    n_drugs = len(drugs_df)

    x = build_node_features(
        n_drugs, int(cfg["model"]["in_dim"]), seed=int(cfg.get("seed", 42))
    ).to(device)

    train_edges = get_train_hyperedges(splits["train"])
    hg = build_hypergraph(train_edges, n_nodes=n_drugs)
    H = hg.to_torch_sparse(device)
    node_deg = torch.from_numpy(hg.node_deg).float().to(device)
    edge_deg = torch.from_numpy(hg.edge_deg).float().to(device)

    pair_edges = hyperedges_to_pair_edges(train_edges)
    pyg_data = build_pyg_data(n_drugs, pair_edges, x).to(device)

    model, ckpt = load_checkpoint(args.checkpoint, cfg)
    model = model.to(device)
    train_extra = {
        "x": x, "H": H, "node_deg": node_deg, "edge_deg": edge_deg, "pyg_data": pyg_data
    }
    forward_fn = make_trainer_forward(
        model, cfg.get("model", {}).get("name", "hgnn"), train_extra
    )

    labeled = build_labeled_dataset(
        splits["test"],
        n_neg_per_pos=int(cfg.get("data", {}).get("neg_per_pos", 1)),
        n_drugs=n_drugs,
        seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(
        DDIDataset(labeled, x.cpu()),
        batch_size=int(cfg.get("training", {}).get("batch_size", 128)),
        collate_fn=collate_ddi_batch,
    )

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = forward_fn(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    y_true = torch.cat(all_labels).numpy()
    y_logits = torch.cat(all_logits).numpy()
    metrics = compute_metrics(y_true, y_logits)
    print(json.dumps(metrics, indent=2))

    if args.output:
        out = args.output
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        fpr, tpr = get_roc_curve_data(y_true, y_logits)
        plot_roc_curve(fpr, tpr, metrics["roc_auc"], os.path.join(out, "roc_curve.png"))


if __name__ == "__main__":
    main()
