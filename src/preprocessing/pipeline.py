"""End-to-end preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.preprocessing.loaders import (
    load_drugbank,
    load_twosides,
    load_synthetic_demo,
    save_processed,
)
from src.preprocessing.splits import save_splits, split_interactions
from src.preprocessing.hypergraph import build_hypergraph
from src.preprocessing.splits import get_train_hyperedges
from src.utils.paths import processed_dir, splits_dir, raw_dir


def run_preprocessing(cfg: Dict[str, Any]) -> None:
    """Run data loading, splitting, and hypergraph build from config."""
    dcfg = cfg.get("data", {})
    source = dcfg.get("source", "synthetic_demo")
    seed = int(cfg.get("seed", 42))

    if source == "synthetic_demo":
        drugs_df, interactions = load_synthetic_demo(
            n_drugs=int(dcfg.get("n_drugs", 50)),
            n_hyperedges=int(dcfg.get("n_hyperedges", 80)),
            max_hyperedge_size=int(dcfg.get("max_hyperedge_size", 4)),
            seed=seed,
        )
    elif source == "drugbank":
        path = dcfg.get("drugbank_xml") or str(raw_dir(cfg) / "drugbank.xml")
        drugs_df, interactions = load_drugbank(path)
    elif source == "twosides":
        path = dcfg.get("twosides_csv") or str(raw_dir(cfg) / "twosides.csv")
        drugs_df, interactions = load_twosides(
            path,
            drug_a_col=dcfg.get("drug_a_col", "drug1_name"),
            drug_b_col=dcfg.get("drug_b_col", "drug2_name"),
        )
    else:
        raise ValueError(f"Unknown data source: {source}")

    proc = processed_dir(cfg)
    save_processed(proc, drugs_df, interactions, source=source)

    splits = split_interactions(
        interactions,
        train_ratio=float(dcfg.get("train_ratio", 0.7)),
        val_ratio=float(dcfg.get("val_ratio", 0.15)),
        seed=seed,
    )
    save_splits(splits_dir(cfg), splits)

    train_edges = get_train_hyperedges(splits["train"])
    hg = build_hypergraph(train_edges, n_nodes=len(drugs_df))
    hg.save(proc / "hypergraph")
