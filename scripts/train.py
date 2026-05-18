#!/usr/bin/env python3
"""Train a DDI model from config."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDI model")
    parser.add_argument("--config", type=str, default="config/demo_synthetic.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    os.environ.setdefault("HYPERGRAPH_DDI_ROOT", ROOT)
    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    runner = ExperimentRunner(config_path)
    result = runner.run(run_id=args.run_id)
    print(f"Training complete. Results: {result['output_dir']}")
    print(f"Test ROC-AUC: {result['test_metrics'].get('roc_auc', 'n/a')}")


if __name__ == "__main__":
    main()
