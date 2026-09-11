"""Time/frequency statistical feature extraction.

Case Study 2 summarises each window with four statistical features -- RMS,
kurtosis, crest factor and skewness. For the classical RF/SVM baselines
(Section 4, "12 handcrafted time/frequency features") a richer 12-feature set is
also provided. All functions handle NaN/Inf defensively.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import stats

# The four core statistical features used as auxiliary network inputs.
CORE_FEATURE_NAMES: List[str] = ["rms", "kurtosis", "crest_factor", "skewness"]

# The 12 handcrafted features for classical baselines.
HANDCRAFTED_FEATURE_NAMES: List[str] = [
    "rms", "kurtosis", "crest_factor", "skewness",
    "peak", "peak_to_peak", "std", "mean_abs",
    "shape_factor", "impulse_factor", "clearance_factor", "spectral_centroid",
]


def _safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def rms(window: np.ndarray) -> float:
    w = _safe(window)
    return float(np.sqrt(np.mean(w ** 2) + 1e-12))


def crest_factor(window: np.ndarray) -> float:
    w = _safe(window)
    r = np.sqrt(np.mean(w ** 2) + 1e-12)
    return float(np.max(np.abs(w)) / (r + 1e-12))


def core_features(window: np.ndarray) -> np.ndarray:
    """Return the 4 core features [RMS, kurtosis, crest factor, skewness]."""
    w = _safe(window)
    r = np.sqrt(np.mean(w ** 2) + 1e-12)
    peak = np.max(np.abs(w))
    kurt = float(stats.kurtosis(w, fisher=True, bias=False)) if w.std() > 1e-9 else 0.0
    skew = float(stats.skew(w, bias=False)) if w.std() > 1e-9 else 0.0
    feats = np.array([r, kurt, peak / (r + 1e-12), skew], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def handcrafted_features(window: np.ndarray, fs: float = 12000.0) -> np.ndarray:
    """Return the 12-feature handcrafted vector for classical baselines."""
    w = _safe(window)
    r = np.sqrt(np.mean(w ** 2) + 1e-12)
    peak = float(np.max(np.abs(w)))
    p2p = float(np.max(w) - np.min(w))
    std = float(np.std(w))
    mean_abs = float(np.mean(np.abs(w)) + 1e-12)
    kurt = float(stats.kurtosis(w, fisher=True, bias=False)) if std > 1e-9 else 0.0
    skew = float(stats.skew(w, bias=False)) if std > 1e-9 else 0.0
    shape_factor = r / mean_abs
    impulse_factor = peak / mean_abs
    clearance_factor = peak / (np.mean(np.sqrt(np.abs(w))) ** 2 + 1e-12)
    # Spectral centroid (frequency-domain feature).
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(w.shape[0], d=1.0 / fs)
    spectral_centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    feats = np.array([
        r, kurt, peak / (r + 1e-12), skew,
        peak, p2p, std, mean_abs,
        shape_factor, impulse_factor, clearance_factor, spectral_centroid,
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def batch_core_features(windows: np.ndarray) -> np.ndarray:
    """Vectorised core features for an array of windows (N, L) -> (N, 4)."""
    return np.stack([core_features(w) for w in windows], axis=0)


def batch_handcrafted_features(windows: np.ndarray, fs: float = 12000.0) -> np.ndarray:
    """Vectorised 12-feature vectors for an array of windows (N, L) -> (N, 12)."""
    return np.stack([handcrafted_features(w, fs) for w in windows], axis=0)
