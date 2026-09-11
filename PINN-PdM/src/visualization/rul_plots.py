"""RUL / health-trajectory plots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_health_trajectory(t: np.ndarray, health_true: np.ndarray, health_pred: np.ndarray,
                           out_path: str | Path, title: str = "Health trajectory",
                           health_pred_baseline: Optional[np.ndarray] = None) -> Path:
    order = np.argsort(t)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t[order], health_true[order], "o-", ms=3, color="#2c3e50", label="true H(t)")
    ax.plot(t[order], health_pred[order], "-", lw=2, color="#27ae60", label="PINN H(t)")
    if health_pred_baseline is not None:
        ax.plot(t[order], health_pred_baseline[order], "--", lw=1.5, color="#e74c3c",
                label="unconstrained H(t)")
    ax.set_xlabel("Normalised time"); ax.set_ylabel("Health index"); ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path,
                        title: str = "Predicted vs actual", unit: str = "RUL") -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y_true, y_pred, s=8, alpha=0.5, color="#2980b9")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"True {unit}"); ax.set_ylabel(f"Predicted {unit}"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path,
                   title: str = "Residuals") -> Path:
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(y_true, resid, s=8, alpha=0.5, color="#8e44ad")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("True"); ax.set_ylabel("Residual (pred - true)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_monotonicity_comparison(t: np.ndarray, unconstrained: np.ndarray,
                                 constrained: np.ndarray, out_path: str | Path) -> Path:
    """Show a health trajectory that spontaneously recovers (unconstrained) vs a
    monotone physics-constrained one."""
    order = np.argsort(t)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    axes[0].plot(t[order], unconstrained[order], "-o", ms=3, color="#e74c3c")
    axes[0].set_title("Unconstrained (non-physical recoveries)")
    axes[1].plot(t[order], constrained[order], "-o", ms=3, color="#27ae60")
    axes[1].set_title("Physics-constrained (monotone)")
    for ax in axes:
        ax.set_xlabel("Normalised time"); ax.set_ylabel("Health")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)
