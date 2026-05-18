"""Drug name normalization for cross-dataset ID mapping."""

from __future__ import annotations

import re
import unicodedata


def normalize_drug_name(name: str) -> str:
    """
    Normalize a drug name for matching across sources.

    - Unicode NFKD decomposition
    - Lowercase, strip whitespace
    - Remove punctuation except hyphens in tokens
    - Collapse multiple spaces
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_name_to_id(names: list[str]) -> dict[str, int]:
    """Map normalized names to integer IDs; duplicates get same ID."""
    name_to_id: dict[str, int] = {}
    for raw in names:
        norm = normalize_drug_name(raw)
        if not norm:
            continue
        if norm not in name_to_id:
            name_to_id[norm] = len(name_to_id)
    return name_to_id
