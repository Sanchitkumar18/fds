"""CWRU bearing dataset loader (fault diagnosis).

Official source
---------------
Case Western Reserve University Bearing Data Center:
https://engineering.case.edu/bearingdatacenter/download-data-file

Place the drive-end ``.mat`` files under ``data/raw/cwru/``. Because the Data
Center does not offer a stable programmatic API, files must be downloaded
manually (see ``scripts/download_data.py`` for the URL list and the README).
Each ``.mat`` file contains a variable like ``X097_DE_time`` (drive-end
accelerometer). This loader:

1. scans ``raw_dir`` for ``.mat`` files,
2. reads the DE (or configured) channel,
3. infers the fault class and motor load from an optional ``catalog.csv`` or
   from the filename convention ``<class>_<load>hp_<...>.mat``,
4. segments each record into 2048-sample windows with metadata.

If no raw files are found and ``allow_synthetic_fallback`` is true, a documented
CWRU-shaped synthetic stand-in is generated across four loads (0-3 HP) and four
classes so the full pipeline -- including the critical cross-load extrapolation
experiment -- runs end-to-end. The stand-in is clearly labelled as synthetic in
the returned metadata.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..physics.bearing_frequencies import SKF_6205, cwru_shaft_hz
from ..utils.logging import get_logger
from .segmentation import segment_with_metadata
from .synthetic import SyntheticConfig, generate_vibration_window

logger = get_logger("cwru_loader")

CLASS_NAMES = ["normal", "inner", "outer", "ball"]
_CLASS_ALIASES = {
    "normal": "normal", "healthy": "normal", "n": "normal", "baseline": "normal",
    "inner": "inner", "ir": "inner", "innerrace": "inner",
    "outer": "outer", "or": "outer", "outerrace": "outer",
    "ball": "ball", "b": "ball", "roller": "ball",
}


def _infer_from_filename(name: str) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort (class_idx, load_hp) inference from a filename."""
    low = name.lower()
    cls_idx = None
    for alias, canon in _CLASS_ALIASES.items():
        if re.search(rf"(^|[^a-z]){alias}([^a-z]|$)", low):
            cls_idx = CLASS_NAMES.index(canon)
            break
    load = None
    m = re.search(r"(\d)\s*hp", low)
    if m:
        load = int(m.group(1))
    return cls_idx, load


def _read_mat_signal(path: Path, channel: str = "DE") -> Optional[np.ndarray]:
    """Read the requested accelerometer channel from a CWRU ``.mat`` file."""
    try:
        from scipy.io import loadmat
        mat = loadmat(str(path))
    except Exception as exc:  # pragma: no cover - corrupt file handling
        logger.warning("Failed to read %s: %s", path.name, exc)
        return None
    # CWRU variables look like 'X097_DE_time', 'X097_FE_time'.
    candidates = [k for k in mat.keys() if k.endswith(f"_{channel}_time")]
    if not candidates:
        candidates = [k for k in mat.keys() if k.endswith("_DE_time")]
    if not candidates:
        return None
    sig = np.asarray(mat[candidates[0]]).ravel().astype(np.float32)
    return sig if sig.size > 0 else None


def load_cwru_raw(cfg: Dict) -> Dict[str, np.ndarray]:
    """Load CWRU (or synthetic fallback) into the raw per-window dict.

    ``cfg`` is the parsed ``cwru.yaml`` dictionary.
    """
    from ..utils.config import resolve_path

    cwru_cfg = cfg["cwru"]
    pre = cfg["preprocess"]
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    window_size = int(pre["window_size"])
    overlap = float(pre["overlap"])
    channel = cwru_cfg.get("channel", "DE")

    mat_files = sorted(Path(raw_dir).glob("*.mat")) if Path(raw_dir).exists() else []
    if mat_files:
        logger.info("Found %d CWRU .mat files in %s", len(mat_files), raw_dir)
        return _load_real_cwru(mat_files, raw_dir, channel, window_size, overlap, cwru_cfg)

    if not cwru_cfg.get("allow_synthetic_fallback", True):
        raise FileNotFoundError(
            f"No CWRU .mat files in {raw_dir} and synthetic fallback disabled. "
            "Download from https://engineering.case.edu/bearingdatacenter and place them there.")

    logger.warning("No CWRU raw files found in %s -- generating documented "
                   "CWRU-shaped SYNTHETIC stand-in (4 loads x 4 classes).", raw_dir)
    return _synthetic_cwru(cfg)


def _load_real_cwru(mat_files: List[Path], raw_dir: Path, channel: str,
                    window_size: int, overlap: float, cwru_cfg: Dict) -> Dict[str, np.ndarray]:
    """Load real CWRU records using an optional catalog.csv or filename inference."""
    import csv

    catalog = {}
    catalog_path = raw_dir / "catalog.csv"
    if catalog_path.exists():
        with open(catalog_path) as fh:
            for row in csv.DictReader(fh):
                catalog[row["filename"]] = (
                    CLASS_NAMES.index(_CLASS_ALIASES.get(row["class"].lower(), row["class"].lower())),
                    int(row["load_hp"]),
                )
        logger.info("Loaded CWRU catalog with %d entries", len(catalog))

    raws, labels, loads, rec_ids = [], [], [], []
    rec = 0
    for f in mat_files:
        if f.name in catalog:
            cls_idx, load = catalog[f.name]
        else:
            cls_idx, load = _infer_from_filename(f.name)
        if cls_idx is None or load is None:
            logger.warning("Could not infer class/load for %s -- skipping. "
                           "Add it to catalog.csv.", f.name)
            continue
        sig = _read_mat_signal(f, channel)
        if sig is None or sig.size < window_size:
            logger.warning("Skipping %s (unreadable or too short)", f.name)
            continue
        windows, meta = segment_with_metadata(sig, record_id=rec, label=cls_idx,
                                               load_hp=load, window_size=window_size,
                                               overlap=overlap)
        raws.append(windows)
        labels.append(meta["label"])
        loads.append(meta["load_hp"])
        rec_ids.append(meta["record_id"])
        rec += 1
    if not raws:
        raise RuntimeError("No usable CWRU records were loaded. Check files/catalog.")
    raw = np.concatenate(raws, axis=0)
    n = raw.shape[0]
    return {
        "raw": raw,
        "label": np.concatenate(labels),
        "load_hp": np.concatenate(loads),
        "record_id": np.concatenate(rec_ids),
        "unit_id": np.concatenate(rec_ids),
        "health": np.full(n, -1.0, dtype=np.float32),
        "rul": np.full(n, -1.0, dtype=np.float32),
        "t": np.zeros(n, dtype=np.float32),
        "_synthetic": np.array([0]),
    }


def _synthetic_cwru(cfg: Dict) -> Dict[str, np.ndarray]:
    """Generate a CWRU-shaped synthetic diagnosis dataset over 4 loads.

    Each (class, load) pair yields several records; shaft speed varies with load
    (per ``rpm_by_load``) so the cross-load extrapolation to 3 HP is meaningful:
    the defect frequencies shift with speed, mimicking the real distribution
    shift the physics term is designed to be robust to.
    """
    cwru_cfg = cfg["cwru"]
    pre = cfg["preprocess"]
    window_size = int(pre["window_size"])
    fs = int(cwru_cfg.get("sampling_rate", 12000))
    loads = list(cwru_cfg.get("loads_hp", [0, 1, 2, 3]))
    classes = list(cwru_cfg.get("fault_classes", CLASS_NAMES))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    records_per_pair = 8
    windows_per_record = 12
    # Weak fault signatures + strong background noise make classification
    # non-trivial (so accuracy is not saturated at 100%) and let the small
    # per-load defect-frequency shift produce a genuine, measurable accuracy drop
    # on the held-out 3 HP load -- the phenomenon the extrapolation test probes.
    severity = float(cwru_cfg.get("synth_severity", 0.35))
    vib_noise = float(cwru_cfg.get("synth_vib_noise", 0.7))

    raws, labels, loads_arr, rec_ids = [], [], [], []
    rec = 0
    for load in loads:
        shaft_hz = cwru_shaft_hz(load)
        scfg = SyntheticConfig(fs=fs, window_size=window_size, shaft_hz=shaft_hz,
                               bearing=SKF_6205, vib_noise_std=vib_noise, seed=seed)
        for cls in classes:
            cls_idx = classes.index(cls)
            for _ in range(records_per_pair):
                # A "record" = a contiguous acquisition -> several overlapping windows.
                wins = np.stack([
                    generate_vibration_window(cls, scfg, severity=severity, rng=rng)
                    for _ in range(windows_per_record)
                ], axis=0)
                raws.append(wins)
                labels.append(np.full(windows_per_record, cls_idx, dtype=np.int64))
                loads_arr.append(np.full(windows_per_record, load, dtype=np.int64))
                rec_ids.append(np.full(windows_per_record, rec, dtype=np.int64))
                rec += 1
    raw = np.concatenate(raws, axis=0)
    n = raw.shape[0]
    return {
        "raw": raw.astype(np.float32),
        "label": np.concatenate(labels),
        "load_hp": np.concatenate(loads_arr),
        "record_id": np.concatenate(rec_ids),
        "unit_id": np.concatenate(rec_ids),
        "health": np.full(n, -1.0, dtype=np.float32),
        "rul": np.full(n, -1.0, dtype=np.float32),
        "t": np.zeros(n, dtype=np.float32),
        "_synthetic": np.array([1]),
    }
