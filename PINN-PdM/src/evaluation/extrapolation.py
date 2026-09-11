"""Cross-condition extrapolation analysis (Case Study 2, Section 6.2).

Given a model's in-distribution accuracy and its held-out (3 HP) accuracy,
compute the absolute and relative accuracy drop -- the headline metric for the
research question "does physics generalise better to an unseen load?".
"""
from __future__ import annotations

from typing import Dict, List


def extrapolation_drop(in_dist_acc: float, held_out_acc: float) -> Dict[str, float]:
    """Absolute and relative accuracy drop from in-distribution to held-out."""
    abs_drop = held_out_acc - in_dist_acc
    rel_drop = 100.0 * abs_drop / in_dist_acc if in_dist_acc > 0 else 0.0
    return {
        "in_dist_acc": in_dist_acc,
        "held_out_acc": held_out_acc,
        "abs_drop": abs_drop,          # negative = accuracy lost
        "rel_drop_pct": rel_drop,
    }


def summarise_extrapolation(rows: List[Dict]) -> List[Dict]:
    """Attach drop columns to a list of {model, in_dist_acc, held_out_acc} rows."""
    out = []
    for r in rows:
        d = extrapolation_drop(r["in_dist_acc"], r["held_out_acc"])
        out.append({"model": r["model"], **d})
    return out
