"""Run a trained model over a data loader and collect predictions."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def collect_predictions(model, loader, device, task: str = "both") -> Dict[str, np.ndarray]:
    """Collect logits / predicted health / labels / metadata for a loader.

    The prognosis head needs ``t`` with grad only during *training*; for
    inference we run under ``no_grad`` and pass ``t`` without grad, which is
    sufficient to read H(t).
    """
    model.eval()
    logits_all, y_all, hp_all, ht_all, hc_all, rul_all, t_all, unit_all = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            feat = batch["feat"].to(device)
            t = batch["t"].to(device) if task in ("prognosis", "both") else None
            out = model(x, feat, t)
            if out.get("logits") is not None:
                logits_all.append(out["logits"].cpu().numpy())
            if out.get("health") is not None:
                hp_all.append(out["health"].squeeze(-1).cpu().numpy())
            y_all.append(batch["y"].numpy())
            ht_all.append(batch["health"].numpy())
            hc_all.append(batch["health_clean"].numpy())
            rul_all.append(batch["rul"].numpy())
            t_all.append(batch["t"].squeeze(-1).numpy())
            unit_all.append(batch["unit_id"].numpy())
    result = {
        "y_true": np.concatenate(y_all) if y_all else np.array([]),
        "health_true": np.concatenate(ht_all) if ht_all else np.array([]),
        "health_clean": np.concatenate(hc_all) if hc_all else np.array([]),
        "rul_true": np.concatenate(rul_all) if rul_all else np.array([]),
        "t": np.concatenate(t_all) if t_all else np.array([]),
        "unit_id": np.concatenate(unit_all) if unit_all else np.array([]),
    }
    if logits_all:
        result["logits"] = np.concatenate(logits_all)
        result["y_pred"] = result["logits"].argmax(axis=1)
    if hp_all:
        result["health_pred"] = np.concatenate(hp_all)
    return result


def health_by_unit(pred: Dict[str, np.ndarray], key: str = "health_pred"
                   ) -> Dict[int, np.ndarray]:
    """Group a per-sample prediction array into time-ordered per-unit arrays."""
    units = pred["unit_id"]
    t = pred["t"]
    vals = pred[key]
    out: Dict[int, np.ndarray] = {}
    for u in np.unique(units):
        mask = units == u
        order = np.argsort(t[mask])
        out[int(u)] = vals[mask][order]
    return out


def value_by_unit(pred: Dict[str, np.ndarray], key: str) -> Dict[int, np.ndarray]:
    """Generic time-ordered per-unit grouping for any prediction key."""
    return health_by_unit(pred, key=key)
