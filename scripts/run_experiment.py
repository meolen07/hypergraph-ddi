#!/usr/bin/env python3
"""Run multi-seed experiments from config."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiments.runner import ExperimentRunner
from src.preprocessing.pipeline import run_preprocessing
from src.utils.config import load_config, resolve_paths
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed DDI experiments")
    parser.add_argument("--config", type=str, default="config/demo_synthetic.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HYPERGRAPH_DDI_ROOT", ROOT)
    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    base_cfg = resolve_paths(load_config(config_path), project_root=ROOT)

    seeds = args.seeds or [int(base_cfg.get("seed", 42))]
    if not args.skip_preprocess:
        run_preprocessing(base_cfg)

    all_results = []
    for seed in seeds:
        cfg = dict(base_cfg)
        cfg["seed"] = seed
        set_seed(seed)
        runner = ExperimentRunner(config_path)
        runner.cfg["seed"] = seed
        result = runner.run(run_id=f"seed{seed}")
        all_results.append({"seed": seed, **result["test_metrics"]})

    out_base = Path(base_cfg["project_root"]) / base_cfg.get("experiment", {}).get("output_dir", "experiments")
    summary_path = out_base / "multi_seed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Multi-seed summary saved to {summary_path}")


if __name__ == "__main__":
    main()
