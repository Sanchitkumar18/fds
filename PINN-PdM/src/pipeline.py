"""End-to-end orchestration: data prep, splitting, loaders and training.

Scripts stay thin by delegating to these functions. The pipeline is dataset- and
task-aware and enforces leakage-safe splitting and train-only feature scaling.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data.dataset import (PdMDataset, apply_feature_scaler, build_processed,
                           collate, load_processed, save_processed)
from .data.splits import (by_unit_split, cross_condition_split,
                          stratified_group_split)
from .models.common import count_parameters
from .models.factory import build_model
from .physics.bearing_frequencies import cwru_shaft_hz
from .training.pinn_trainer import PINNTrainer
from .training.trainer import build_baseline_trainer
from .utils.config import get_device, resolve_path
from .utils.logging import get_logger
from .utils.seed import set_seed

logger = get_logger("pipeline")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_raw(cfg: Dict) -> Dict[str, np.ndarray]:
    """Dispatch to the correct loader based on ``cfg['dataset']``."""
    ds = cfg["dataset"]
    if ds == "synthetic":
        from .data.synthetic import generate_dataset, synthetic_config_from_dict
        return generate_dataset(synthetic_config_from_dict(cfg.get("synthetic", {})))
    if ds == "cwru":
        from .data.cwru_loader import load_cwru_raw
        return load_cwru_raw(cfg)
    if ds == "ims":
        from .data.ims_loader import load_ims_raw
        return load_ims_raw(cfg)
    raise ValueError(f"load_raw does not handle dataset '{ds}' (see cmapss_loader for C-MAPSS)")


def _fs_for(cfg: Dict) -> float:
    ds = cfg["dataset"]
    if ds == "synthetic":
        return float(cfg.get("synthetic", {}).get("fs", 12000))
    if ds == "cwru":
        return float(cfg["cwru"].get("sampling_rate", 12000))
    if ds == "ims":
        return float(cfg["ims"].get("sampling_rate", 20000))
    return 12000.0


def prepare_dataset(cfg: Dict, force: bool = False) -> Tuple[Dict[str, np.ndarray], Path]:
    """Load raw -> build processed (STFT+features) -> cache to npz. Returns
    (processed, path). Reuses the cache unless ``force``."""
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    cache = processed_dir / "processed.npz"
    if cache.exists() and not force:
        logger.info("Loading cached processed dataset: %s", cache)
        return load_processed(cache), cache
    logger.info("Preparing dataset '%s' ...", cfg["dataset"])
    raw = load_raw(cfg)
    fs = _fs_for(cfg)
    processed = build_processed(raw, fs=fs, stft_size=int(cfg["preprocess"]["stft_size"]))
    processed["_synthetic"] = raw.get("_synthetic", np.array([1]))
    save_processed(processed, cache, extra_meta={"dataset": cfg["dataset"], "fs": fs})
    logger.info("Cached %d windows -> %s", processed["stft"].shape[0], cache)
    return processed, cache


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def make_splits(cfg: Dict, processed: Dict[str, np.ndarray],
                regime: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Return train/val/test index arrays honouring the dataset regime."""
    ds = cfg["dataset"]
    seed = int(cfg.get("seed", 42))
    labels = processed["label"]
    groups = processed["record_id"]
    split_cfg = cfg.get("split", {})
    regime = regime or split_cfg.get("regime", "in_distribution")

    if ds == "ims" or regime == "by_unit":
        units = processed["unit_id"]
        train_units = split_cfg.get("train_units")
        tr, te = by_unit_split(units, test_frac=float(split_cfg.get("test_frac", 0.30)),
                               seed=seed, train_units=train_units)
        # Carve a validation set from train units.
        rng = np.random.default_rng(seed)
        u_train = np.unique(units[tr])
        rng.shuffle(u_train)
        n_val = max(1, int(round(len(u_train) * 0.2)))
        val_units = set(u_train[:n_val].tolist())
        val = np.array([i for i in tr if units[i] in val_units], dtype=int)
        tr2 = np.array([i for i in tr if units[i] not in val_units], dtype=int)
        return {"train": tr2, "val": val, "test": te}

    if regime == "cross_condition":
        cc = split_cfg.get("cross_condition", {})
        tr, val, te = cross_condition_split(
            processed["load_hp"], cc.get("train_loads", [0, 1, 2]),
            cc.get("test_loads", [3]), val_frac=float(split_cfg.get("val", 0.15)),
            seed=seed, groups=groups)
        return {"train": tr, "val": val, "test": te}

    # Default: stratified group split (in-distribution 70/15/15).
    tr, val, te = stratified_group_split(
        labels, groups, train=float(split_cfg.get("train", 0.70)),
        val=float(split_cfg.get("val", 0.15)), test=float(split_cfg.get("test", 0.15)),
        seed=seed)
    return {"train": tr, "val": val, "test": te}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def make_loaders(cfg: Dict, processed: Dict[str, np.ndarray], splits: Dict[str, np.ndarray],
                 task: str) -> Dict[str, "object"]:
    """Build DataLoaders, fitting the feature scaler on train indices only."""
    import torch
    from torch.utils.data import DataLoader

    # Train-only feature standardisation (leakage-safe).
    if cfg["preprocess"].get("standardize", True) and "train" in splits and len(splits["train"]):
        apply_feature_scaler(processed, splits["train"], key="feats")

    seq_len = int(cfg["model"].get("seq_len", 1))
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"].get("num_workers", 0))
    g = torch.Generator(); g.manual_seed(int(cfg.get("seed", 42)))

    loaders = {}
    for name, idx in splits.items():
        if len(idx) == 0:
            loaders[name] = None
            continue
        dset = PdMDataset(processed, idx, seq_len=seq_len, task=task)
        loaders[name] = DataLoader(dset, batch_size=bs, shuffle=(name == "train"),
                                   num_workers=nw, collate_fn=collate,
                                   generator=g if name == "train" else None,
                                   drop_last=False)
    return loaders


def shaft_hz_for(cfg: Dict) -> float:
    ds = cfg["dataset"]
    if ds == "synthetic":
        return float(cfg.get("synthetic", {}).get("shaft_hz", 30.0))
    if ds == "cwru":
        return cwru_shaft_hz(0)
    if ds == "ims":
        return 33.0
    return 30.0


# ---------------------------------------------------------------------------
# Training one model
# ---------------------------------------------------------------------------
def estimate_rho_from_processed(processed: Dict[str, np.ndarray], train_idx: np.ndarray) -> float:
    """Least-squares estimate of the degradation rate rho from training health.

    Averages a per-unit fit of log H = log H0 - rho t over the training units.
    Used to initialise the PINN's co-estimated rho near a data-driven value (a
    standard PINN inverse-problem practice), so the ODE prior enforces the
    *correct* exponential and denoises rather than biasing predictions.
    """
    from .physics.degradation import estimate_rho
    units = processed["unit_id"][train_idx]
    t = processed["t"][train_idx]
    h = processed["health"][train_idx]
    rhos = []
    for u in np.unique(units):
        m = units == u
        if m.sum() >= 3 and np.all(h[m] > 0):
            rhos.append(estimate_rho(t[m], h[m]))
    rhos = [r for r in rhos if 0.05 < r < 20]
    return float(np.median(rhos)) if rhos else 1.0


def train_model(cfg: Dict, model_name: str, task: str, processed: Dict[str, np.ndarray],
                splits: Dict[str, np.ndarray], device=None, log_root: Optional[str] = None,
                is_pinn: Optional[bool] = None) -> Dict:
    """Train a single model and return a result dict with history, params, time."""
    set_seed(int(cfg.get("seed", 42)))
    device = device or get_device(cfg.get("device", "auto"))
    class_names = list(cfg.get(cfg["dataset"], {}).get("fault_classes",
                       cfg.get("synthetic", {}).get("fault_classes", ["normal", "inner", "outer", "ball"])))
    n_classes = int(max(processed["label"]) + 1) if processed["label"].size else len(class_names)
    n_classes = max(n_classes, len(class_names))

    # For physics-capable prognosis models, initialise the co-estimated rho from
    # a data-driven estimate on the training split (documented; see above).
    if is_pinn and task in ("prognosis", "both") and processed["health"].max() >= 0:
        import copy as _copy
        cfg = _copy.deepcopy(cfg)
        cfg.setdefault("physics", {})
        cfg["physics"]["rho_init"] = estimate_rho_from_processed(processed, splits["train"])
        logger.info("Initialised rho from training data: %.3f", cfg["physics"]["rho_init"])

    loaders = make_loaders(cfg, processed, splits, task)
    model = build_model(model_name, cfg, n_classes=n_classes,
                        n_features=int(cfg["preprocess"].get("n_features", 4)))
    n_params = count_parameters(model)

    freq_axis = processed["freq_axis"]
    shaft = shaft_hz_for(cfg)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['dataset']}_{model_name}.pt"
    log_dir = str(Path(log_root or (resolve_path(cfg["paths"]["output_dir"]) / "logs"))
                  / f"{cfg['dataset']}_{model_name}")

    # Per-load shaft frequencies for load-aware spectral masks (CWRU).
    shaft_by_load = None
    if cfg["dataset"] == "cwru":
        from .physics.bearing_frequencies import rpm_to_hz
        rpm_by_load = cfg["cwru"].get("rpm_by_load", {0: 1797, 1: 1772, 2: 1750, 3: 1730})
        shaft_by_load = {int(k): rpm_to_hz(float(v)) for k, v in rpm_by_load.items()}

    if is_pinn is None:
        is_pinn = (model_name == "pinn_pdm")
    if is_pinn:
        trainer = PINNTrainer(model, cfg, device, task, class_names, freq_axis, shaft,
                              log_dir=log_dir, model_name=model_name,
                              shaft_hz_by_load=shaft_by_load)
    else:
        trainer = build_baseline_trainer(model, cfg, device, task, class_names, freq_axis,
                                         shaft, log_dir=log_dir, model_name=model_name)

    t0 = time.time()
    fit = trainer.fit(loaders["train"], loaders["val"] or loaders["train"], ckpt_path=str(ckpt_path))
    train_time = time.time() - t0

    return {
        "model_name": model_name, "task": task, "n_params": n_params,
        "history": fit["history"], "final_weights": fit["final_weights"],
        "train_time_s": train_time, "checkpoint": str(ckpt_path),
        "class_names": class_names, "n_classes": n_classes,
        "device": str(device), "model": model, "loaders": loaders,
        "is_pinn": is_pinn,
    }
