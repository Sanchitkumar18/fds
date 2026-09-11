"""Model-comparison and ablation bar charts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_bars(models: List[str], values: List[float], out_path: str | Path,
                     ylabel: str, title: str, highlight: str = "pinn_pdm",
                     lower_better: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#27ae60" if m.lower().replace("-", "_").startswith(highlight) else "#95a5a6"
              for m in models]
    bars = ax.bar(models, values, color=colors)
    ax.set_ylabel(ylabel); ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_extrapolation_comparison(rows: List[Dict], out_path: str | Path) -> Path:
    """Grouped bars: in-distribution vs held-out accuracy per model."""
    models = [r["model"] for r in rows]
    in_dist = [r["in_dist_acc"] for r in rows]
    held = [r["held_out_acc"] for r in rows]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, in_dist, w, label="in-distribution", color="#3498db")
    ax.bar(x + w / 2, held, w, label="held-out (3 HP)", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_title("Cross-condition extrapolation")
    ax.legend()
    for i, (a, b) in enumerate(zip(in_dist, held)):
        ax.text(i - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_ablation(variants: List[str], metrics: Dict[str, List[float]],
                  out_path: str | Path) -> Path:
    """Grouped bars for the ablation: several metrics across variants."""
    n = len(variants)
    keys = list(metrics.keys())
    x = np.arange(n); w = 0.8 / max(1, len(keys))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    palette = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad"]
    for i, k in enumerate(keys):
        ax.bar(x + i * w - 0.4 + w / 2, metrics[k], w, label=k, color=palette[i % len(palette)])
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_title("Ablation study"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)
