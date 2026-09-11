"""Signal-level plots: raw waveform, FFT, PSD, STFT with fault-frequency overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal


def plot_raw_signal(sig: np.ndarray, fs: float, title: str, out_path: str | Path) -> Path:
    t = np.arange(len(sig)) / fs
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, sig, lw=0.6, color="#2c3e50")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_fft(sig: np.ndarray, fs: float, title: str, out_path: str | Path,
             fault_freqs: Optional[Dict[str, float]] = None, fmax: float = 6000.0) -> Path:
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, spec, lw=0.7, color="#2980b9")
    if fault_freqs:
        colors = {"outer": "#e74c3c", "inner": "#27ae60", "ball": "#8e44ad", "cage": "#f39c12"}
        for name, f0 in fault_freqs.items():
            if f0 <= fmax:
                ax.axvline(f0, color=colors.get(name, "gray"), ls="--", lw=1.0,
                           label=f"{name.upper()} {f0:.0f}Hz")
        ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, fmax); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("|FFT|")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_psd(sig: np.ndarray, fs: float, title: str, out_path: str | Path,
             fmax: float = 6000.0) -> Path:
    f, pxx = sp_signal.welch(sig, fs=fs, nperseg=min(1024, len(sig)))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(f, pxx + 1e-12, lw=0.8, color="#16a085")
    ax.set_xlim(0, fmax); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_stft_map(stft_map: np.ndarray, title: str, out_path: str | Path,
                  freq_axis: Optional[np.ndarray] = None) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    extent = None
    if freq_axis is not None:
        extent = [0, 1, freq_axis[0], freq_axis[-1]]
    im = ax.imshow(stft_map, aspect="auto", origin="lower", cmap="magma", extent=extent)
    ax.set_title(title); ax.set_xlabel("Time (norm.)"); ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_class_distribution(labels: np.ndarray, class_names, out_path: str | Path,
                            title: str = "Class distribution") -> Path:
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(class_names, counts, color="#34495e")
    ax.set_ylabel("Windows"); ax.set_title(title)
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def plot_load_distribution(load_hp: np.ndarray, out_path: str | Path) -> Path:
    valid = load_hp[load_hp >= 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    if valid.size:
        uniq, counts = np.unique(valid, return_counts=True)
        ax.bar([f"{u} HP" for u in uniq], counts, color="#7f8c8d")
    ax.set_ylabel("Windows"); ax.set_title("Load distribution")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)
