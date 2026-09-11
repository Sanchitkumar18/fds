"""IMS bearing run-to-failure dataset loader (RUL / degradation).

Official source
---------------
NASA Prognostics Center of Excellence Data Repository (IMS Bearings), provided
by the Center for Intelligent Maintenance Systems, University of Cincinnati:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Place the extracted test folders (e.g. ``1st_test/``, ``2nd_test/``,
``3rd_test/``) under ``data/raw/ims/``. Each test folder contains many ASCII
snapshot files, one per ~10-minute acquisition, with columns = accelerometer
channels (one column per bearing). Files sort chronologically by name.

This loader treats each bearing channel as a run-to-failure *unit*, builds a
health index from RMS of each snapshot (normalised so start=1, failure~0),
constructs RUL labels, and windows the final snapshot signals for STFT/features.
The split is BY UNIT so no bearing appears in both train and test
(Case Study #16, #28).

If no raw files are found and ``allow_synthetic_fallback`` is true, a documented
synthetic run-to-failure stand-in is generated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils.logging import get_logger
from .synthetic import SyntheticConfig, generate_dataset, synthetic_config_from_dict

logger = get_logger("ims_loader")


def load_ims_raw(cfg: Dict) -> Dict[str, np.ndarray]:
    """Load IMS (or synthetic fallback) into the raw per-window dict."""
    from ..utils.config import resolve_path

    ims_cfg = cfg["ims"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    test_dirs = [d for d in sorted(Path(raw_dir).glob("*")) if d.is_dir()] \
        if Path(raw_dir).exists() else []

    if test_dirs:
        logger.info("Found %d IMS test folders in %s", len(test_dirs), raw_dir)
        try:
            return _load_real_ims(test_dirs, cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Real IMS load failed (%s); falling back to synthetic.", exc)

    if not ims_cfg.get("allow_synthetic_fallback", True):
        raise FileNotFoundError(
            f"No IMS data in {raw_dir} and synthetic fallback disabled. See README.")

    logger.warning("No IMS raw files found -- generating documented SYNTHETIC "
                   "run-to-failure stand-in.")
    return _synthetic_ims(cfg)


def _load_real_ims(test_dirs: List[Path], cfg: Dict) -> Dict[str, np.ndarray]:
    """Load real IMS snapshots into run-to-failure units (one per bearing channel)."""
    ims_cfg = cfg["ims"]
    pre = cfg["preprocess"]
    window_size = int(pre["window_size"])
    fs = int(ims_cfg.get("sampling_rate", 20000))
    smoothing = int(ims_cfg.get("smoothing_window", 5))

    raws: List[np.ndarray] = []
    labels, health, rul, tnorm, unit_ids, rec_ids = [], [], [], [], [], []
    unit_counter = 0

    for td in test_dirs:
        files = sorted([f for f in td.iterdir() if f.is_file()])
        if not files:
            continue
        # Peek at first file to count channels (bearings).
        try:
            sample = np.loadtxt(files[0])
        except Exception:
            continue
        n_channels = 1 if sample.ndim == 1 else sample.shape[1]
        # Per channel: RMS over time -> health proxy.
        rms_series = np.zeros((len(files), n_channels), dtype=np.float64)
        last_window = [None] * n_channels
        for fi, f in enumerate(files):
            try:
                arr = np.loadtxt(f)
            except Exception:
                continue
            if arr.ndim == 1:
                arr = arr[:, None]
            for ch in range(min(n_channels, arr.shape[1])):
                col = arr[:, ch]
                rms_series[fi, ch] = np.sqrt(np.mean(col ** 2) + 1e-12)
                last_window[ch] = col[:window_size]
        # Build a unit per channel.
        for ch in range(n_channels):
            series = rms_series[:, ch]
            if smoothing > 1:
                kernel = np.ones(smoothing) / smoothing
                series = np.convolve(series, kernel, mode="same")
            # Health: inverted, normalised RMS (rising RMS => falling health).
            s_min, s_max = series.min(), series.max()
            if s_max - s_min < 1e-9:
                continue
            h = 1.0 - (series - s_min) / (s_max - s_min)
            steps = len(h)
            t_grid = np.linspace(0.0, 1.0, steps)
            # Use each snapshot's signal window as the vibration input.
            for k in range(steps):
                f = files[k]
                try:
                    arr = np.loadtxt(f)
                except Exception:
                    continue
                if arr.ndim == 1:
                    arr = arr[:, None]
                if ch >= arr.shape[1]:
                    continue
                w = arr[:window_size, ch].astype(np.float32)
                if w.shape[0] < window_size:
                    w = np.pad(w, (0, window_size - w.shape[0]))
                raws.append(w)
                labels.append(0)
                health.append(float(h[k]))
                rul.append(float(steps - 1 - k))
                tnorm.append(float(t_grid[k]))
                unit_ids.append(unit_counter)
                rec_ids.append(unit_counter)
            unit_counter += 1

    if not raws:
        raise RuntimeError("No usable IMS snapshots loaded.")
    n = len(raws)
    return {
        "raw": np.stack(raws).astype(np.float32),
        "label": np.array(labels, dtype=np.int64),
        "health": np.array(health, dtype=np.float32),
        "rul": np.array(rul, dtype=np.float32),
        "t": np.array(tnorm, dtype=np.float32),
        "unit_id": np.array(unit_ids, dtype=np.int64),
        "record_id": np.array(rec_ids, dtype=np.int64),
        "load_hp": np.full(n, -1, dtype=np.int64),
        "_synthetic": np.array([0]),
    }


def _synthetic_ims(cfg: Dict) -> Dict[str, np.ndarray]:
    """Generate a synthetic run-to-failure dataset for the RUL task."""
    ims_cfg = cfg["ims"]
    pre = cfg["preprocess"]
    scfg = SyntheticConfig(
        n_units=40,
        n_train_units=28,
        steps_per_unit=40,
        fs=int(ims_cfg.get("sampling_rate", 20000)),
        window_size=int(pre["window_size"]),
        shaft_hz=33.0,
        noise_std=0.06,
        rho=1.0,
        seed=int(cfg.get("seed", 42)),
    )
    data = generate_dataset(scfg)
    # For the pure RUL task, collapse fault labels to 0 (health-only).
    data["label"] = np.zeros_like(data["label"])
    data["_synthetic"] = np.array([1])
    return data
