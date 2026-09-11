"""Confusion-matrix plotting."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, class_names: List[str], out_path: str | Path,
                          title: str = "Confusion matrix", normalize: bool = True) -> Path:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cmn = cm / row_sums
    else:
        cmn = cm
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1 if normalize else cmn.max())
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thresh = cmn.max() / 2.0
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            txt = f"{cmn[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cmn[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)
