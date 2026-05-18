#!/usr/bin/env python3
"""Preprocess DrugBank, TWOSIDES, or synthetic demo data."""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config, resolve_paths
from src.preprocessing.pipeline import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DDI data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/demo_synthetic.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    os.environ.setdefault("HYPERGRAPH_DDI_ROOT", ROOT)
    cfg = resolve_paths(load_config(args.config), project_root=ROOT)
    run_preprocessing(cfg)
    print(f"Preprocessing complete. Output: {cfg.get('data', {}).get('root', 'data')}/processed")


if __name__ == "__main__":
    main()
