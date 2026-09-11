"""Configuration loading and device utilities.

Configs are YAML files (see ``config/*.yaml``). They are loaded into a plain
``dict`` and lightly wrapped so nested keys can be accessed with dotted paths
(``cfg.get("training.epochs")``) while remaining ordinary dictionaries for
serialisation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Repository root = two levels up from this file (src/utils/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Relative ``path`` values are resolved against the repository root so the
    same command works regardless of the current working directory. No paths
    are hard-coded.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["_config_path"] = str(p)
    return cfg


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def get_nested(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch ``cfg['a']['b']`` via the dotted key ``'a.b'`` with a default."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def get_device(prefer: Optional[str] = None) -> "object":
    """Return a torch device, auto-detecting CUDA (and Apple MPS) when present.

    ``prefer`` may be ``"cpu"``, ``"cuda"`` or ``"mps"`` to force a device;
    otherwise the best available accelerator is chosen.
    """
    import torch

    if prefer is not None and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
