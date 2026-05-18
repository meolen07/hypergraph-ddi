#!/usr/bin/env python3
"""Verify all package imports (requires dependencies installed)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main() -> None:
    from src.models.hgnn import HGNN, HGNNLayer
    from src.models.gcn import GCNBaseline
    from src.models.gat import GATBaseline
    from src.models.graphsage import GraphSAGEBaseline
    from src.models.mlp_baseline import MLPBaseline
    from src.preprocessing import build_hypergraph, load_synthetic_demo
    from src.experiments.runner import ExperimentRunner
    from src.evaluation.metrics import compute_metrics
    print("All imports OK:", HGNN, GCNBaseline, GATBaseline, GraphSAGEBaseline, MLPBaseline)


if __name__ == "__main__":
    main()
