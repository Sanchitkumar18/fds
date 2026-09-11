"""Training-curve plots: losses, adaptive lambdas, rho, physics residual."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(history: List[Dict], out_path: str | Path,
                     title: str = "Training losses") -> Path:
    epochs = [h["epoch"] for h in history]
    keys = [k for k in history[0] if k.startswith("train_") and k != "train_total"]
    fig, ax = plt.subplots(figsize=(7, 4))
    if any("train_total" in h for h in history):
        ax.plot(epochs, [h.get("train_total", np.nan) for h in history],
                label="total", lw=2, color="black")
    palette = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12"]
    for i, k in enumerate(keys):
        ax.plot(epochs, [h.get(k, np.nan) for h in history], label=k.replace("train_", ""),
                lw=1.3, color=palette[i % len(palette)])
    ax.plot(epochs, [h.get("val_loss", np.nan) for h in history], label="val_loss",
            lw=1.3, ls="--", color="gray")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_adaptive_weights(history: List[Dict], out_path: str | Path) -> Path:
    epochs = [h["epoch"] for h in history]
    lam_keys = [k for k in history[0] if k.startswith("lambda_")]
    fig, ax = plt.subplots(figsize=(7, 4))
    if not lam_keys:
        ax.text(0.5, 0.5, "No adaptive weights (static or baseline)",
                ha="center", va="center")
    for k in lam_keys:
        ax.plot(epochs, [h.get(k, np.nan) for h in history], label=k.replace("lambda_", "λ_"), lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Adaptive weight"); ax.set_yscale("log")
    ax.set_title("Self-adaptive physics-weight evolution"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_rho_evolution(history: List[Dict], out_path: str | Path,
                       true_rho: float = None) -> Path:
    epochs = [h["epoch"] for h in history if "rho" in h]
    rhos = [h["rho"] for h in history if "rho" in h]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if rhos:
        ax.plot(epochs, rhos, lw=1.8, color="#c0392b", label="co-estimated ρ")
    if true_rho is not None:
        ax.axhline(true_rho, ls="--", color="gray", label=f"true ρ={true_rho}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("ρ"); ax.set_title("Co-estimated degradation rate ρ")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_residual_distribution(residuals: np.ndarray, out_path: str | Path,
                               title: str = "Physics residual distribution") -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.asarray(residuals), bins=40, color="#2980b9", alpha=0.8)
    ax.set_xlabel("|dH/dt + ρH|"); ax.set_ylabel("Count"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)
