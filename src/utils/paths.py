"""Path helpers relative to project root."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def data_dir(cfg: Dict[str, Any], sub: str = "") -> Path:
    """Return data directory path from config."""
    base = Path(cfg.get("data", {}).get("root", "data"))
    if not base.is_absolute():
        base = Path(cfg["project_root"]) / base
    return base / sub if sub else base


def processed_dir(cfg: Dict[str, Any]) -> Path:
    return data_dir(cfg, "processed")


def splits_dir(cfg: Dict[str, Any]) -> Path:
    return data_dir(cfg, "splits")


def raw_dir(cfg: Dict[str, Any]) -> Path:
    return data_dir(cfg, "raw")
