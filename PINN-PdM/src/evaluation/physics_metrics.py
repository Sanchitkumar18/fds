"""Physics-consistency metrics: monotonicity violations and ODE residual.

Monotonicity is measured per run-to-failure *sequence*: a violation is a step
where the predicted health *increases* (physically impossible). Reports:

* percentage of sequences with any violation,
* number of violating timesteps,
* mean violation magnitude,

matching Case Study requirement #10 / Table 4.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def monotonicity_report(health_by_unit: Dict[int, np.ndarray], threshold: float = 1e-3) -> Dict:
    """Analyse monotonicity of predicted health trajectories per unit.

    ``health_by_unit`` maps unit id -> predicted health array ordered in time.
    A step k violates monotonicity if H[k+1] - H[k] > ``threshold`` (health
    should not *rise*). The small threshold excludes float-noise blips so the
    metric counts meaningful physical recoveries, not numerical jitter -- the
    monotonicity penalty is designed to remove real recoveries, and its effect is
    also captured by ``mean_violation_magnitude``.
    """
    n_units = len(health_by_unit)
    units_with_violation = 0
    total_steps = 0
    violating_steps = 0
    magnitudes: List[float] = []
    for _, h in health_by_unit.items():
        h = np.asarray(h, dtype=float)
        if h.size < 2:
            continue
        diffs = np.diff(h)
        rises = diffs > threshold
        total_steps += diffs.size
        violating_steps += int(rises.sum())
        if rises.any():
            units_with_violation += 1
            magnitudes.extend(diffs[rises].tolist())
    return {
        "n_units": n_units,
        "pct_sequences_violating": 100.0 * units_with_violation / max(1, n_units),
        "n_violating_steps": violating_steps,
        "total_steps": total_steps,
        "pct_steps_violating": 100.0 * violating_steps / max(1, total_steps),
        "mean_violation_magnitude": float(np.mean(magnitudes)) if magnitudes else 0.0,
    }


def mean_physics_residual(health_by_unit: Dict[int, np.ndarray],
                          t_by_unit: Dict[int, np.ndarray], rho: float) -> Dict:
    """Mean absolute degradation-ODE residual |dH/dt + rho*H| over all units.

    ``dH/dt`` is estimated by finite differences here (this is a *diagnostic*
    evaluation on already-predicted trajectories, not the training residual,
    which uses exact autograd -- Case Study #9).
    """
    residuals: List[float] = []
    for u, h in health_by_unit.items():
        h = np.asarray(h, dtype=float)
        t = np.asarray(t_by_unit[u], dtype=float)
        if h.size < 2:
            continue
        order = np.argsort(t)
        h, t = h[order], t[order]
        dhdt = np.gradient(h, t)
        residuals.extend(np.abs(dhdt + rho * h).tolist())
    arr = np.asarray(residuals) if residuals else np.array([0.0])
    return {"mean_physics_residual": float(np.mean(arr)),
            "median_physics_residual": float(np.median(arr)),
            "residuals": arr.tolist()}
