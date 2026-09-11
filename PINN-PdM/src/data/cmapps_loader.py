"""NASA C-MAPSS FD001 loader (OPTIONAL / stretch RUL benchmark).

Official source
---------------
NASA Prognostics Center of Excellence, "Turbofan Engine Degradation Simulation
Data Set" (C-MAPSS):
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Place ``train_FD001.txt``, ``test_FD001.txt`` and ``RUL_FD001.txt`` under
``data/raw/cmapss/``. Each row is: unit, cycle, 3 operating settings, 21 sensors.

This loader builds sliding windows of multivariate sensor readings and a
piecewise-linear RUL target (capped at ``rul_clip``). It is intentionally
tabular/multivariate (not STFT-based) and feeds the ``pinn_pdm_mlp`` variant.
The whole module is optional -- the core CWRU/IMS/synthetic pipeline runs
without it. Falls back to a synthetic multivariate degradation stand-in when the
raw files are absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger("cmapss_loader")


def load_cmapss(cfg: Dict) -> Dict[str, np.ndarray]:
    """Load C-MAPSS FD001 windows (or a synthetic fallback).

    Returns a dict with ``X`` (N, window, C), ``rul`` (N,), ``unit_id`` (N,),
    ``t`` (N,) normalised time, and a ``split`` marker ('train'/'test').
    """
    from ..utils.config import resolve_path

    cm = cfg["cmapss"]
    subset = cfg.get("subset", "FD001")
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    train_path = Path(raw_dir) / f"train_{subset}.txt"
    test_path = Path(raw_dir) / f"test_{subset}.txt"
    rul_path = Path(raw_dir) / f"RUL_{subset}.txt"

    if train_path.exists() and test_path.exists() and rul_path.exists():
        logger.info("Loading real C-MAPSS %s from %s", subset, raw_dir)
        return _load_real_cmapss(train_path, test_path, rul_path, cm)

    if not cm.get("allow_synthetic_fallback", True):
        raise FileNotFoundError(f"No C-MAPSS {subset} files in {raw_dir}. See README.")
    logger.warning("No C-MAPSS files found -- generating SYNTHETIC multivariate stand-in.")
    return _synthetic_cmapss(cfg)


def _prep_columns(df: np.ndarray, sensors_drop) -> np.ndarray:
    """Drop near-constant sensor columns (1-indexed sensor ids)."""
    keep = [i for i in range(21) if (i + 1) not in set(sensors_drop)]
    return df[:, keep]


def _load_real_cmapss(train_path: Path, test_path: Path, rul_path: Path,
                      cm: Dict) -> Dict[str, np.ndarray]:
    rul_clip = int(cm.get("rul_clip", 125))
    window = int(cm.get("window_size", 30))
    stride = int(cm.get("window_stride", 1))
    sensors_drop = cm.get("sensors_drop", [])

    train = np.loadtxt(train_path)
    # Columns: 0 unit, 1 cycle, 2-4 settings, 5-25 sensors.
    def build(arr: np.ndarray, rul_true: Optional[np.ndarray]) -> Tuple:
        Xs, ys, uids, ts = [], [], [], []
        units = np.unique(arr[:, 0]).astype(int)
        for u in units:
            rows = arr[arr[:, 0] == u]
            sensors = rows[:, 5:26]
            sensors = _prep_columns(sensors, sensors_drop)
            n_cycles = rows.shape[0]
            if rul_true is None:
                max_rul = n_cycles
            else:
                max_rul = n_cycles + int(rul_true[u - 1])
            for start in range(0, max(1, n_cycles - window + 1), stride):
                win = sensors[start:start + window]
                if win.shape[0] < window:
                    win = np.pad(win, ((window - win.shape[0], 0), (0, 0)), mode="edge")
                end_cycle = start + window
                rul = max_rul - end_cycle
                rul = min(rul, rul_clip)
                Xs.append(win)
                ys.append(max(rul, 0))
                uids.append(u)
                ts.append(end_cycle / max(1, max_rul))
        return Xs, ys, uids, ts

    Xtr, ytr, utr, ttr = build(train, None)
    X = np.array(Xtr, dtype=np.float32)
    # Per-feature standardisation fit on train.
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std
    return {
        "X": X.astype(np.float32),
        "rul": np.array(ytr, dtype=np.float32),
        "unit_id": np.array(utr, dtype=np.int64),
        "t": np.array(ttr, dtype=np.float32),
        "_synthetic": np.array([0]),
        "_scaler_mean": mean.astype(np.float32),
        "_scaler_std": std.astype(np.float32),
    }


def _synthetic_cmapss(cfg: Dict) -> Dict[str, np.ndarray]:
    """Synthetic multivariate turbofan-like degradation."""
    cm = cfg["cmapss"]
    window = int(cm.get("window_size", 30))
    rul_clip = int(cm.get("rul_clip", 125))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    n_units = 80
    n_sensors = 14
    Xs, ys, uids, ts = [], [], [], []
    for u in range(n_units):
        life = rng.integers(130, 250)
        t = np.linspace(0, 1, life)
        # Each sensor drifts monotonically with degradation + noise.
        base = rng.normal(0, 1, size=n_sensors)
        drift = rng.uniform(0.5, 2.0, size=n_sensors) * (rng.integers(0, 2, n_sensors) * 2 - 1)
        signal = base[None, :] + t[:, None] * drift[None, :] + rng.normal(0, 0.1, (life, n_sensors))
        for start in range(0, life - window + 1):
            win = signal[start:start + window]
            end_cycle = start + window
            rul = min(life - end_cycle, rul_clip)
            Xs.append(win)
            ys.append(max(rul, 0))
            uids.append(u)
            ts.append(end_cycle / life)
    X = np.array(Xs, dtype=np.float32)
    mean = X.reshape(-1, n_sensors).mean(axis=0)
    std = X.reshape(-1, n_sensors).std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std
    return {
        "X": X.astype(np.float32),
        "rul": np.array(ys, dtype=np.float32),
        "unit_id": np.array(uids, dtype=np.int64),
        "t": np.array(ts, dtype=np.float32),
        "_synthetic": np.array([1]),
        "_scaler_mean": mean.astype(np.float32),
        "_scaler_std": std.astype(np.float32),
    }
