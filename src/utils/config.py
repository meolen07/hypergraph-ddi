"""YAML configuration loading and path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return cfg


def resolve_paths(cfg: Dict[str, Any], project_root: str | Path | None = None) -> Dict[str, Any]:
    """
    Resolve relative paths in config using project root or HYPERGRAPH_DDI_ROOT env.
    """
    root = Path(
        project_root
        or os.environ.get("HYPERGRAPH_DDI_ROOT", ".")
    ).resolve()

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("./"):
            return str((root / obj[2:]).resolve())
        return obj

    cfg = _resolve(cfg)
    cfg.setdefault("project_root", str(root))
    return cfg


def get_project_root(cfg: Dict[str, Any]) -> Path:
    """Return project root from config."""
    return Path(cfg.get("project_root", ".")).resolve()
