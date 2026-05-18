"""Negative sampling for link / interaction prediction."""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from src.preprocessing.loaders import InteractionRecord


def _positive_set(interactions: List[InteractionRecord]) -> Set[frozenset]:
    return {frozenset(nodes) for nodes, _ in interactions}


def sample_negatives(
    n_samples: int,
    n_drugs: int,
    positive_edges: Set[frozenset],
    hyperedge_size: int = 2,
    rng: np.random.Generator | None = None,
    max_attempts: int = 100_000,
) -> List[InteractionRecord]:
    """
    Sample negative hyperedges not in the positive set.

    For pair prediction (hyperedge_size=2), samples random drug pairs.
    """
    rng = rng or np.random.default_rng()
    negatives: List[InteractionRecord] = []
    attempts = 0
    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        if hyperedge_size == 2:
            a, b = rng.integers(0, n_drugs, size=2)
            if a == b:
                continue
            nodes = sorted([int(a), int(b)])
        else:
            nodes = sorted(
                rng.choice(n_drugs, size=hyperedge_size, replace=False).tolist()
            )
        key = frozenset(nodes)
        if key in positive_edges:
            continue
        negatives.append((nodes, 0))
    if len(negatives) < n_samples:
        raise RuntimeError(
            f"Could only sample {len(negatives)}/{n_samples} negatives. "
            "Increase drug space or reduce negative ratio."
        )
    return negatives


def build_labeled_dataset(
    positives: List[InteractionRecord],
    n_neg_per_pos: int,
    n_drugs: int,
    seed: int = 42,
    exclude_positives: List[InteractionRecord] | None = None,
) -> List[InteractionRecord]:
    """
    Combine positives with sampled negatives.

    exclude_positives: additional positive hyperedges to avoid in negative sampling
    (e.g. train positives when building val/test sets).
    """
    rng = np.random.default_rng(seed)
    pos_set = _positive_set(positives)
    if exclude_positives:
        pos_set = pos_set | _positive_set(exclude_positives)
    n_neg = len(positives) * n_neg_per_pos
    negs = sample_negatives(
        n_neg, n_drugs, pos_set, hyperedge_size=2, rng=rng
    )
    combined = positives + negs
    perm = rng.permutation(len(combined))
    return [combined[i] for i in perm]
