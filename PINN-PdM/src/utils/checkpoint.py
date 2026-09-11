"""Model checkpoint saving/loading with metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def save_checkpoint(
    path: str | Path,
    model: "object",
    optimizer: Optional["object"] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist model (and optionally optimizer) state plus metadata to ``path``."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
        "model_class": type(model).__name__,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    # Human-readable sidecar for quick inspection without loading torch.
    sidecar = path.with_suffix(".meta.json")
    with open(sidecar, "w") as fh:
        json.dump({"epoch": epoch, "metrics": metrics or {},
                   "model_class": type(model).__name__, "extra": extra or {}}, fh, indent=2)
    return path


def load_checkpoint(
    path: str | Path,
    model: "object",
    optimizer: Optional["object"] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint into ``model`` (and ``optimizer`` if given).

    Returns the metadata dict (epoch, metrics, extra).
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return {"epoch": ckpt.get("epoch", 0), "metrics": ckpt.get("metrics", {}),
            "extra": ckpt.get("extra", {}), "model_class": ckpt.get("model_class")}
