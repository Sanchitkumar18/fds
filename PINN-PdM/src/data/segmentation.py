"""Signal windowing / segmentation utilities.

Case Study 2 segments continuous vibration into fixed-length windows of 2048
samples with 50% overlap. This module implements overlap-aware segmentation and
tracks, for every emitted window, the *record id* it came from so that
downstream splitting can keep overlapping windows of one record together and
avoid train/test leakage (Case Study requirement #28).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def segment_signal(
    signal: np.ndarray,
    window_size: int = 2048,
    overlap: float = 0.5,
    drop_last: bool = True,
) -> np.ndarray:
    """Segment a 1-D signal into overlapping windows.

    Parameters
    ----------
    signal:
        1-D array.
    window_size:
        Samples per window.
    overlap:
        Fractional overlap in [0, 1). 0.5 -> 50% overlap (hop = window/2).
    drop_last:
        Drop a trailing partial window if the signal length is not a multiple
        of the hop. If ``False``, the last window is zero-padded.

    Returns
    -------
    windows: array of shape (n_windows, window_size).
    """
    signal = np.asarray(signal, dtype=np.float32).ravel()
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    hop = max(1, int(round(window_size * (1.0 - overlap))))
    n = signal.shape[0]
    if n < window_size:
        if drop_last:
            return np.empty((0, window_size), dtype=np.float32)
        padded = np.zeros(window_size, dtype=np.float32)
        padded[:n] = signal
        return padded[None, :]
    starts = list(range(0, n - window_size + 1, hop))
    if not drop_last and (starts[-1] + window_size) < n:
        starts.append(n - window_size)
    windows = np.stack([signal[s:s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)


def segment_with_metadata(
    signal: np.ndarray,
    record_id: int,
    label: int,
    load_hp: int = -1,
    window_size: int = 2048,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """Segment a signal and produce aligned metadata arrays.

    Returns ``(windows, meta)`` where ``meta`` has arrays ``record_id``,
    ``label`` and ``load_hp`` (one entry per window). ``record_id`` groups
    overlapping windows so they never straddle a train/test split.
    """
    windows = segment_signal(signal, window_size=window_size, overlap=overlap)
    m = windows.shape[0]
    meta = {
        "record_id": np.full(m, record_id, dtype=np.int64),
        "label": np.full(m, label, dtype=np.int64),
        "load_hp": np.full(m, load_hp, dtype=np.int64),
        "window_index": np.arange(m, dtype=np.int64),
    }
    return windows, meta
