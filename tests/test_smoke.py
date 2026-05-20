"""Lightweight smoke tests for imports and hypergraph utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _chdir():
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
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
