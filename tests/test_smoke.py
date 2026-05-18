"""Smoke tests for import and synthetic demo pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _chdir():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    yield


def test_imports():
    from src.models.hgnn import HGNN
    from src.preprocessing.hypergraph import build_hypergraph
    from src.evaluation.metrics import compute_metrics
    assert HGNN is not None
    assert build_hypergraph is not None
    assert compute_metrics is not None


def test_hypergraph_build():
    from src.preprocessing.hypergraph import build_hypergraph
    hg = build_hypergraph([[0, 1], [1, 2, 3]], n_nodes=4)
    assert hg.n_nodes == 4
    assert hg.n_edges == 2


def test_synthetic_preprocess_and_train():
    """End-to-end smoke test on SYNTHETIC DEMO data only."""
    env = os.environ.copy()
    env["HYPERGRAPH_DDI_ROOT"] = str(ROOT)
    subprocess.run(
        [sys.executable, "scripts/preprocess.py", "--config", "config/demo_synthetic.yaml"],
        check=True,
        cwd=ROOT,
        env=env,
    )
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", "config/demo_synthetic.yaml", "--run-id", "smoke"],
        check=True,
        cwd=ROOT,
        env=env,
        timeout=300,
    )
    exp_dirs = list((ROOT / "experiments").glob("demo_hgnn_smoke"))
    assert exp_dirs, "Expected experiment output directory"
