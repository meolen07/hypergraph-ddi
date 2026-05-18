"""Data loaders for DrugBank, TWOSIDES, and synthetic demo mode."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from src.preprocessing.normalization import normalize_drug_name, build_name_to_id


InteractionRecord = Tuple[List[int], int]  # (drug_ids in hyperedge), label


def load_drugbank(
    xml_path: str | Path,
    min_drugs_per_interaction: int = 2,
) -> Tuple[pd.DataFrame, List[InteractionRecord]]:
    """
    Load drug-drug interactions from DrugBank XML export.

    Expects standard DrugBank XML with <drug> entries containing
    <name> and <drug-interaction> blocks with <name> of partner drugs.

    Parameters
    ----------
    xml_path : path to DrugBank full database XML (user-provided)
    min_drugs_per_interaction : minimum drugs per hyperedge (default 2 for pairs)

    Returns
    -------
    drugs_df : columns [drug_id, name, normalized_name]
    interactions : list of ([drug_id, ...], label=1) hyperedges
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(
            f"DrugBank XML not found: {xml_path}. "
            "Download from https://go.drugbank.com/ and place under data/raw/."
        )

    ns = {"db": "http://www.drugbank.ca"}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    raw_names: list[str] = []
    raw_interactions: list[list[str]] = []

    for drug_elem in root.findall("db:drug", ns):
        if drug_elem.get("type") != "small molecule":
            continue
        name_elem = drug_elem.find("db:name", ns)
        if name_elem is None or not name_elem.text:
            continue
        drug_name = name_elem.text.strip()
        partners: list[str] = []
        for di in drug_elem.findall(".//db:drug-interaction", ns):
            partner = di.find("db:name", ns)
            if partner is not None and partner.text:
                partners.append(partner.text.strip())
        if partners:
            raw_names.append(drug_name)
            for p in partners:
                raw_interactions.append([drug_name, p])

    if not raw_interactions:
        raise ValueError(f"No interactions parsed from {xml_path}")

    all_names = list({n for pair in raw_interactions for n in pair} | set(raw_names))
    norm_to_id = build_name_to_id(all_names)

    drugs_df = pd.DataFrame(
        [
            {"drug_id": i, "name": n, "normalized_name": normalize_drug_name(n)}
            for n, i in sorted(
                ((k, v) for k, v in norm_to_id.items()),
                key=lambda x: x[1],
            )
        ]
    )

    interactions: List[InteractionRecord] = []
    seen: set[frozenset] = set()
    for a, b in raw_interactions:
        na, nb = normalize_drug_name(a), normalize_drug_name(b)
        if na not in norm_to_id or nb not in norm_to_id:
            continue
        ids = sorted([norm_to_id[na], norm_to_id[nb]])
        key = frozenset(ids)
        if key in seen or len(key) < min_drugs_per_interaction:
            continue
        seen.add(key)
        interactions.append((list(key), 1))

    return drugs_df, interactions


def load_twosides(
    csv_path: str | Path,
    drug_a_col: str = "drug1_name",
    drug_b_col: str = "drug2_name",
    min_count: int = 1,
) -> Tuple[pd.DataFrame, List[InteractionRecord]]:
    """
    Load drug pairs from TWOSIDES-style CSV.

    Expected columns: drug names for pair (configurable).
    Aggregates unique pairs as positive hyperedges (binary DDI proxy).

    User must provide the TWOSIDES file (e.g. from Tatonetti lab releases).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"TWOSIDES CSV not found: {csv_path}. "
            "Place user-downloaded TWOSIDES file under data/raw/."
        )

    df = pd.read_csv(csv_path, low_memory=False)
    for col in (drug_a_col, drug_b_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not in {csv_path}. "
                f"Available: {list(df.columns)[:20]}..."
            )

    pairs = df[[drug_a_col, drug_b_col]].dropna()
    if min_count > 1 and "count" in df.columns:
        pairs = pairs[df["count"] >= min_count]

    all_names = pd.unique(
        pd.concat([pairs[drug_a_col], pairs[drug_b_col]], ignore_index=True)
    ).tolist()
    norm_to_id = build_name_to_id([str(n) for n in all_names])

    drugs_df = pd.DataFrame(
        [
            {"drug_id": i, "normalized_name": k}
            for k, i in sorted(norm_to_id.items(), key=lambda x: x[1])
        ]
    )

    interactions: List[InteractionRecord] = []
    seen: set[frozenset] = set()
    for _, row in pairs.iterrows():
        na = normalize_drug_name(str(row[drug_a_col]))
        nb = normalize_drug_name(str(row[drug_b_col]))
        if na not in norm_to_id or nb not in norm_to_id or na == nb:
            continue
        ids = sorted([norm_to_id[na], norm_to_id[nb]])
        key = frozenset(ids)
        if key in seen:
            continue
        seen.add(key)
        interactions.append((list(ids), 1))

    return drugs_df, interactions


def load_synthetic_demo(
    n_drugs: int = 50,
    n_hyperedges: int = 80,
    max_hyperedge_size: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[InteractionRecord]]:
    """
    **SYNTHETIC DEMO ONLY** — clearly labeled toy data for smoke tests.

    Not for publication results. Use real DrugBank/TWOSIDES for research.
    """
    rng = np.random.default_rng(seed)
    drugs_df = pd.DataFrame(
        {
            "drug_id": np.arange(n_drugs),
            "name": [f"demo_drug_{i}" for i in range(n_drugs)],
            "normalized_name": [f"demo_drug_{i}" for i in range(n_drugs)],
            "is_synthetic": True,
        }
    )
    interactions: List[InteractionRecord] = []
    seen: set[frozenset] = set()
    attempts = 0
    while len(interactions) < n_hyperedges and attempts < n_hyperedges * 20:
        attempts += 1
        size = int(rng.integers(2, max_hyperedge_size + 1))
        nodes = sorted(rng.choice(n_drugs, size=size, replace=False).tolist())
        key = frozenset(nodes)
        if key in seen:
            continue
        seen.add(key)
        interactions.append((nodes, 1))
    return drugs_df, interactions


def save_processed(
    out_dir: str | Path,
    drugs_df: pd.DataFrame,
    interactions: List[InteractionRecord],
    source: str,
) -> None:
    """Persist processed drugs and interactions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drugs_df.to_parquet(out_dir / "drugs.parquet", index=False)
    rows = [{"nodes": n, "label": l} for n, l in interactions]
    pd.DataFrame(rows).to_parquet(out_dir / "interactions.parquet", index=False)
    (out_dir / "source.txt").write_text(source, encoding="utf-8")


def load_processed(processed_dir: str | Path) -> Tuple[pd.DataFrame, List[InteractionRecord]]:
    """Load preprocessed drugs and interactions."""
    processed_dir = Path(processed_dir)
    drugs_df = pd.read_parquet(processed_dir / "drugs.parquet")
    inter_df = pd.read_parquet(processed_dir / "interactions.parquet")
    interactions = []
    for _, row in inter_df.iterrows():
        nodes = row["nodes"]
        if hasattr(nodes, "tolist"):
            nodes = nodes.tolist()
        interactions.append((list(nodes), int(row["label"])))
    return drugs_df, interactions
