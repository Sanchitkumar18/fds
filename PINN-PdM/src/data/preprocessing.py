"""Preprocessing: STFT time-frequency maps and train-only standardisation.

Case Study 2 transforms each 2048-sample window into a 64x64 STFT magnitude map
and standardises per channel. Crucially, all scaling parameters are fit on the
*training* split only, then applied to val/test (Case Study requirement #5, #28)
to prevent data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import zoom


def compute_stft_map(
    window: np.ndarray,
    fs: float = 12000.0,
    out_size: int = 64,
    log_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a fixed-size STFT magnitude map from a 1-D window.

    Uses ``scipy.signal.stft`` (no librosa dependency required) and resamples
    the magnitude spectrogram to ``out_size`` x ``out_size`` via spline zoom.

    Returns
    -------
    (mag_map, freq_axis)
        ``mag_map``: (out_size, out_size) float32 in roughly [0, 1].
        ``freq_axis``: length-``out_size`` array of centre frequencies (Hz)
        matching the rows of ``mag_map`` (used by the spectral physics loss).
    """
    window = np.nan_to_num(np.asarray(window, dtype=np.float64).ravel())
    n = window.shape[0]
    # Choose nperseg so we get ~out_size time frames after STFT.
    nperseg = max(16, n // out_size * 2)
    nperseg = min(nperseg, n)
    noverlap = nperseg // 2
    f, t, Zxx = sp_signal.stft(window, fs=fs, nperseg=nperseg, noverlap=noverlap,
                               boundary=None, padded=True)
    mag = np.abs(Zxx).astype(np.float64)  # (F, T)
    if log_scale:
        mag = np.log1p(mag)
    # Resample to (out_size, out_size).
    if mag.shape[0] < 2 or mag.shape[1] < 2:
        mag = np.pad(mag, ((0, max(0, 2 - mag.shape[0])), (0, max(0, 2 - mag.shape[1]))))
    zoom_f = out_size / mag.shape[0]
    zoom_t = out_size / mag.shape[1]
    mag_map = zoom(mag, (zoom_f, zoom_t), order=1)
    mag_map = mag_map[:out_size, :out_size]
    if mag_map.shape != (out_size, out_size):
        fixed = np.zeros((out_size, out_size), dtype=np.float64)
        fixed[:mag_map.shape[0], :mag_map.shape[1]] = mag_map
        mag_map = fixed
    # Per-map min-max normalisation for stable conv inputs.
    mmin, mmax = mag_map.min(), mag_map.max()
    if mmax - mmin > 1e-9:
        mag_map = (mag_map - mmin) / (mmax - mmin)
    # Frequency axis resampled to out_size rows (0 .. fs/2).
    freq_axis = np.linspace(0.0, fs / 2.0, out_size)
    return mag_map.astype(np.float32), freq_axis.astype(np.float32)


def batch_stft_maps(windows: np.ndarray, fs: float = 12000.0,
                    out_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Compute STFT maps for an array of windows (N, L) -> (N, out, out)."""
    maps = []
    freq_axis = None
    for w in windows:
        m, freq_axis = compute_stft_map(w, fs=fs, out_size=out_size)
        maps.append(m)
    if not maps:
        return np.empty((0, out_size, out_size), dtype=np.float32), \
            np.linspace(0.0, fs / 2.0, out_size).astype(np.float32)
    return np.stack(maps, axis=0), freq_axis


@dataclass
class StandardScaler1D:
    """Feature-wise standardiser fit on training data only.

    Stores per-feature mean/std so the exact transform can be reapplied to
    val/test and persisted for reproducibility.
    """

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler1D":
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("scaler not fit")
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict:
        return {"mean": None if self.mean is None else self.mean.tolist(),
                "std": None if self.std is None else self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler1D":
        s = cls()
        s.mean = None if d.get("mean") is None else np.array(d["mean"])
        s.std = None if d.get("std") is None else np.array(d["std"])
        return s
