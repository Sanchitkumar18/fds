"""Degradation physics: first-order exponential health model and RUL mapping.

Case Study 2 (Section 2.3) models the scalar health index H(t) with a
first-order ODE whose residual form is:

    r_phys(t) = dH/dt - rho * H(t) = 0,   rho > 0

The analytical solution is H(t) = H0 * exp(rho_signed * t). Because health
*decays*, the physically meaningful trajectory is H(t) = H0 * exp(-|rho| * t)
(monotonically decreasing towards failure). The PINN co-estimates ``rho`` and
enforces the residual via automatic differentiation (see ``physics_losses``).

This module provides the analytical model (for the synthetic testbed and for
sanity checks) and the documented H(t) <-> RUL conversion.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def exponential_health(t: np.ndarray, rho: float, h0: float = 1.0) -> np.ndarray:
    """Analytical decaying health index H(t) = h0 * exp(-rho * t).

    ``rho > 0`` is the degradation rate. H starts at ``h0`` (typically 1.0 =
    perfectly healthy) and decays towards 0 (failed).
    """
    if rho < 0:
        raise ValueError("rho must be >= 0 for a decaying health model")
    return h0 * np.exp(-rho * np.asarray(t, dtype=float))


def health_to_rul(health: np.ndarray, failure_threshold: float = 0.0,
                  total_life: Optional[float] = None) -> np.ndarray:
    """Convert a normalised health trajectory to Remaining Useful Life.

    Two documented conventions are supported:

    * If ``total_life`` is given, RUL is expressed on the same time scale as a
      linear function of remaining health fraction:
      ``RUL = total_life * (H - failure_threshold) / (H[0] - failure_threshold)``.
    * Otherwise RUL is returned as the *normalised remaining health fraction*
      in ``[0, 1]``: ``RUL = clip((H - failure_threshold) / (H[0] - thr), 0, 1)``.

    This keeps the assumption explicit (Case Study requirement #17): the network
    predicts H(t) first; RUL is a monotone transform of H.
    """
    health = np.asarray(health, dtype=float)
    denom = health[0] - failure_threshold
    if abs(denom) < 1e-12:
        denom = 1e-12
    frac = np.clip((health - failure_threshold) / denom, 0.0, 1.0)
    if total_life is not None:
        return total_life * frac
    return frac


def rul_from_index(step_index: np.ndarray, total_steps: int) -> np.ndarray:
    """Linear ground-truth RUL label from a step index within a sequence.

    RUL(k) = (total_steps - 1 - k), i.e. cycles remaining until the last
    observed step. Used for run-to-failure datasets where the end of the
    recording is treated as end-of-life.
    """
    step_index = np.asarray(step_index, dtype=float)
    return np.maximum(total_steps - 1 - step_index, 0.0)


def estimate_rho(t: np.ndarray, health: np.ndarray) -> float:
    """Least-squares estimate of ``rho`` from a health trajectory.

    Fits log(H) = log(H0) - rho * t. Useful as an analytical baseline and to
    initialise the PINN's learnable ``rho``.
    """
    t = np.asarray(t, dtype=float)
    health = np.asarray(health, dtype=float)
    mask = health > 1e-6
    if mask.sum() < 2:
        return 0.0
    slope, _ = np.polyfit(t[mask], np.log(health[mask]), 1)
    return float(-slope)
