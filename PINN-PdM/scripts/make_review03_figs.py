#!/usr/bin/env python3
"""Figures for the Review-03 manuscript, from real CWRU/MFPT data and the
measured run summaries in ``outputs/review03``. Writes to ``paper/figs``.

    python scripts/make_review03_figs.py            # all figures available so far
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocessing import compute_stft_map
from src.review03.data_real import (CWRU_CLASSES, SKF_6205, fault_frequency_ratios,
                                    load_cwru, spectral_views)
from src.utils.config import REPO_ROOT

OUT = Path(REPO_ROOT) / "paper" / "figs"
RES = Path(REPO_ROOT) / "outputs" / "review03"
# Reference categorical palette (validated slots 1-3) + neutral for baselines.
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _window(recs, rid, k=10):
    r = next(r for r in recs if r.rec_id == rid)
    return r, r.signal[k * 1024:k * 1024 + 2048]


def fig_envelope(recs):
    """Raw vs squared-envelope spectrum of real windows, defect lines overlaid."""
    ratios = fault_frequency_ratios(SKF_6205)
    rows = [("CWRU234", "outer", "Outer race (0.021 in, 0 hp)"),
            ("CWRU209", "inner", "Inner race (0.021 in, 0 hp)"),
            ("CWRU222", "ball", "Ball (0.021 in, 0 hp)")]
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.0, 4.4), sharex=True)
    for i, (rid, cls, title) in enumerate(rows):
        r, w = _window(recs, rid)
        raw, env, f = spectral_views(w[None], r.fs)
        f0 = ratios[cls] * r.shaft_hz
        for j, (S, name) in enumerate([(raw[0], "raw power spectrum"), (env[0], "squared-envelope spectrum")]):
            ax = axes[i, j]
            ax.plot(f, S / S.max(), color=C1, lw=1.0)
            for h in (1, 2, 3):
                ax.axvline(h * f0, color=C2, lw=0.9, ls="--", zorder=0)
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{title}: {name}", loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("normalised power")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_envelope_real.png")
    plt.close(fig)


def fig_stft(recs):
    rows = [("CWRU97", "Healthy"), ("CWRU209", "Inner race"), ("CWRU234", "Outer race"), ("CWRU222", "Ball")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, (rid, name) in zip(axes, rows):
        r, w = _window(recs, rid)
        m, _ = compute_stft_map(w, fs=r.fs, out_size=64)
        ax.imshow(m, origin="lower", aspect="auto", cmap="Blues", extent=[0, 2048 / r.fs * 1000, 0, r.fs / 2000])
        ax.set_title(name, color=INK)
        ax.set_xlabel("time (ms)")
        ax.grid(False)
    axes[0].set_ylabel("frequency (kHz)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stft_real.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Result figures (need outputs/review03/summary.json)
# ---------------------------------------------------------------------------
LABEL = {"rf": "RF", "svm": "SVM", "cnn": "CNN", "transformer": "Transformer", "cnn_lstm": "CNN-LSTM",
         "spec_v1_adapt": "+L_spec v1 (adapt.)", "spec_v2_raw_adapt": "+L_spec v2 raw (adapt.)",
         "spec_v2_env_adapt": "+L_spec v2 env (adapt.)", "spec_v2_env_fix0.1": "+v2 env, fixed 0.1",
         "spec_v2_env_fix0.5": "+v2 env, fixed 0.5", "spec_v2_env_fix1.0": "+v2 env, fixed 1.0",
         "spec_v2_env_adapt_colloc": "+v2 env + unlabelled target"}


def fig_robustness(summ):
    s = summ["xload"]
    cfgs = [("cnn_lstm", GREY, "o"), ("spec_v2_raw_adapt", C3, "^"),
            ("spec_v2_env_adapt", C1, "s"), ("spec_v2_env_adapt_colloc", C2, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.4, 1]})
    snrs = ["clean", "noise_10dB", "noise_5dB", "noise_0dB", "noise_-5dB"]
    xs = np.arange(len(snrs))
    for c, col, mk in cfgs:
        if c not in s:
            continue
        m = [s[c][("test" if k == "clean" else k) + "_acc"]["mean"] for k in snrs]
        h = [s[c][("test" if k == "clean" else k) + "_acc"]["ci95"] for k in snrs]
        axes[0].errorbar(xs, m, yerr=h, color=col, marker=mk, ms=4, lw=1.5, capsize=2, label=LABEL[c])
    axes[0].set_xticks(xs, ["clean", "10 dB", "5 dB", "0 dB", "-5 dB"])
    axes[0].set_ylabel("accuracy on held-out 3 hp")
    axes[0].set_title("(a) additive white noise (SNR)", loc="left")
    axes[0].legend(frameon=False, loc="lower left")
    sp = ["test", "speed_x0.95", "speed_x1.05", "speed_fluct5pct"]
    width = 0.8 / len(cfgs)
    for i, (c, col, _) in enumerate(cfgs):
        if c not in s:
            continue
        m = [s[c][k + "_acc"]["mean"] for k in sp]
        h = [s[c][k + "_acc"]["ci95"] for k in sp]
        axes[1].bar(np.arange(len(sp)) + (i - 1.5) * width, m, width * 0.9, yerr=h, color=col,
                    capsize=1.5, error_kw={"lw": 0.7})
    axes[1].set_xticks(np.arange(len(sp)), ["nominal", "x0.95", "x1.05", "+/-5% fluct."])
    axes[1].set_title("(b) emulated speed change", loc="left")
    axes[1].set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness.png")
    plt.close(fig)


def fig_sensitivity(summ):
    s = summ.get("sens")
    if not s:
        return
    sig, H = [2, 5, 10, 20], [1, 3, 5]
    grid = np.array([[s[f"env_sigma{a:g}_H{h}"]["test_acc"]["mean"] for h in H] for a in sig])
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    im = ax.imshow(grid, cmap="Blues", aspect="auto", vmin=grid.min() - 0.01, vmax=grid.max())
    for i in range(len(sig)):
        for j in range(len(H)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > grid.min() + 0.6 * (grid.max() - grid.min()) else INK)
    ax.set_xticks(range(len(H)), [str(h) for h in H])
    ax.set_yticks(range(len(sig)), [f"{a} Hz" for a in sig])
    ax.set_xlabel("harmonics H")
    ax.set_ylabel("band width sigma")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="3 hp accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.png")
    plt.close(fig)


def fig_confusion(summ):
    from src.review03 import train  # noqa: F401  (for parity of labels)
    runs = sorted((RES / "runs" / "xload").glob("spec_v2_env_adapt__seed*.json")) + \
        sorted((RES / "runs" / "xload").glob("cnn_lstm__seed*.json"))
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for ax, cfg, title in zip(axes, ["cnn_lstm", "spec_v2_env_adapt"], ["CNN-LSTM", "+L_spec v2 env (adapt.)"]):
        cms = [np.array(json.loads(p.read_text())["test"]["cm"]) for p in
               (RES / "runs" / "xload").glob(f"{cfg}__seed*.json")]
        cm = np.sum(cms, axis=0).astype(float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.6 else INK)
        ax.set_xticks(range(4), ["H", "IR", "OR", "B"])
        ax.set_yticks(range(4), ["H", "IR", "OR", "B"])
        ax.set_title(title)
        ax.set_xlabel("predicted")
        ax.grid(False)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion_real.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_cwru(Path(REPO_ROOT) / "data/raw/cwru")
    fig_envelope(recs)
    fig_stft(recs)
    p = RES / "summary.json"
    if p.exists():
        summ = json.loads(p.read_text())
        if "xload" in summ:
            fig_robustness(summ)
            fig_confusion(summ)
        fig_sensitivity(summ)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
