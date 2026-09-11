"""Synthetic physics-consistent testbed generator (Case Study 2, Section 6.7).

Generates vibration windows whose fault signatures sit at the *theoretical*
bearing defect frequencies (SKF-6205 geometry) plus harmonics and noise, and
run-to-failure health trajectories H(t) = exp(-rho t) with ~5% sensor noise.

This is the DEFAULT, always-runnable data source. It lets the whole pipeline
(preprocess -> STFT -> features -> baselines -> PINN -> physics losses ->
evaluation -> plots) execute without any external download, and it is the
measured proof-of-concept that verifies the physics module recovers the
closed-form defect frequencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..physics.bearing_frequencies import (SKF_6205, BearingGeometry,
                                           all_fault_frequencies)
from ..physics.degradation import exponential_health


@dataclass
class SyntheticConfig:
    n_units: int = 160
    n_train_units: int = 120
    steps_per_unit: int = 32
    fs: int = 12000
    window_size: int = 2048
    shaft_hz: float = 30.0
    noise_std: float = 0.05           # health-trajectory sensor noise (~5%)
    vib_noise_std: float = 0.15       # background vibration noise
    rho: float = 1.2                  # ground-truth degradation rate
    fault_classes: Tuple[str, ...] = ("normal", "inner", "outer", "ball")
    n_harmonics: int = 3
    bearing: BearingGeometry = SKF_6205
    seed: int = 42


def _impact_resonance(t: np.ndarray, defect_hz: float, fs: float,
                      resonance_hz: float = 3000.0, decay: float = 800.0,
                      n_harmonics: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate a bearing defect vibration: periodic impacts exciting a
    high-frequency structural resonance, repeating at ``defect_hz``.

    This produces energy concentrated at ``defect_hz`` and its harmonics -- the
    signature the spectral physics loss expects.
    """
    rng = rng or np.random.default_rng()
    n = t.shape[0]
    sig = np.zeros(n, dtype=np.float64)
    period = fs / defect_hz
    # Impact train: decaying sinusoid bursts at each defect period.
    impact_times = np.arange(0, n, period)
    for it in impact_times:
        it = int(round(it)) + int(rng.normal(0, period * 0.01))  # small jitter
        if it < 0 or it >= n:
            continue
        tt = (np.arange(n) - it) / fs
        mask = tt >= 0
        burst = np.zeros(n)
        burst[mask] = np.exp(-decay * tt[mask]) * np.sin(2 * np.pi * resonance_hz * tt[mask])
        sig += burst
    # Add explicit harmonic tones at the defect frequency for a crisp signature.
    for h in range(1, n_harmonics + 1):
        sig += (0.5 / h) * np.sin(2 * np.pi * defect_hz * h * t)
    return sig


def generate_vibration_window(fault_class: str, cfg: SyntheticConfig,
                              severity: float = 1.0,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate one vibration window for a given fault class.

    ``severity`` in [0, 1] scales the fault amplitude (used to couple diagnosis
    with the degradation trajectory).
    """
    rng = rng or np.random.default_rng()
    n = cfg.window_size
    t = np.arange(n) / cfg.fs
    # Baseline shaft rotation + broadband noise present in every class.
    sig = 0.3 * np.sin(2 * np.pi * cfg.shaft_hz * t)
    sig += rng.normal(0, cfg.vib_noise_std, size=n)
    if fault_class != "normal":
        freqs = all_fault_frequencies(cfg.shaft_hz, cfg.bearing)
        key = {"inner": "inner", "outer": "outer", "ball": "ball", "cage": "cage"}[fault_class]
        defect_hz = freqs[key]
        amp = 1.0 + 2.0 * severity
        sig += amp * _impact_resonance(t, defect_hz, cfg.fs,
                                       n_harmonics=cfg.n_harmonics, rng=rng)
    return sig.astype(np.float32)


def generate_dataset(cfg: SyntheticConfig) -> Dict[str, np.ndarray]:
    """Generate the full synthetic dataset (diagnosis + prognosis).

    Each *unit* is a run-to-failure sequence: health decays as exp(-rho t) and
    the fault severity grows as health drops, so vibration signatures intensify
    over the trajectory. A unit is assigned one fault class (or normal). Windows
    are emitted per time-step.

    Returns a dict of aligned per-window arrays:
        raw       (N, window_size) float32   -- raw vibration windows
        label     (N,) int64                 -- fault class index
        health    (N,) float32               -- normalised health index in (0,1]
        rul       (N,) float32               -- remaining steps (RUL label)
        t         (N,) float32               -- normalised time in [0,1] within unit
        unit_id   (N,) int64                 -- run-to-failure unit id
        record_id (N,) int64                 -- grouping id (== unit for synthetic)
        load_hp   (N,) int64                 -- placeholder (0)
        is_train_unit (N,) int64             -- 1 if unit in the train partition
    plus scalars in the returned ``meta`` via ``dataset['_meta']`` (object).
    """
    rng = np.random.default_rng(cfg.seed)
    classes = list(cfg.fault_classes)
    raw, label, health_arr, rul_arr, tnorm = [], [], [], [], []
    health_clean_arr = []
    unit_id, record_id, load_hp, is_train = [], [], [], []

    steps = cfg.steps_per_unit
    t_grid = np.linspace(0.0, 1.0, steps)
    for u in range(cfg.n_units):
        cls = classes[u % len(classes)]
        cls_idx = classes.index(cls)
        # Health trajectory H(t) = exp(-rho t) with additive sensor noise.
        h_clean = exponential_health(t_grid, cfg.rho, h0=1.0)
        h_noisy = h_clean + rng.normal(0, cfg.noise_std, size=steps)
        h_noisy = np.clip(h_noisy, 1e-3, 1.2).astype(np.float32)
        for k in range(steps):
            severity = float(np.clip(1.0 - h_clean[k], 0.0, 1.0))
            w = generate_vibration_window(cls, cfg, severity=severity, rng=rng)
            raw.append(w)
            label.append(cls_idx)
            health_arr.append(h_noisy[k])
            health_clean_arr.append(float(h_clean[k]))
            rul_arr.append(float(steps - 1 - k))
            tnorm.append(float(t_grid[k]))
            unit_id.append(u)
            record_id.append(u)
            load_hp.append(0)
            is_train.append(1 if u < cfg.n_train_units else 0)

    data = {
        "raw": np.stack(raw).astype(np.float32),
        "label": np.array(label, dtype=np.int64),
        "health": np.array(health_arr, dtype=np.float32),
        "health_clean": np.array(health_clean_arr, dtype=np.float32),
        "rul": np.array(rul_arr, dtype=np.float32),
        "t": np.array(tnorm, dtype=np.float32),
        "unit_id": np.array(unit_id, dtype=np.int64),
        "record_id": np.array(record_id, dtype=np.int64),
        "load_hp": np.array(load_hp, dtype=np.int64),
        "is_train_unit": np.array(is_train, dtype=np.int64),
    }
    return data


def synthetic_config_from_dict(d: dict) -> SyntheticConfig:
    """Build a :class:`SyntheticConfig` from a parsed YAML ``synthetic`` block."""
    return SyntheticConfig(
        n_units=int(d.get("n_units", 160)),
        n_train_units=int(d.get("n_train_units", 120)),
        steps_per_unit=int(d.get("steps_per_unit", 32)),
        fs=int(d.get("fs", 12000)),
        window_size=int(d.get("window_size", 2048)),
        shaft_hz=float(d.get("shaft_hz", 30.0)),
        noise_std=float(d.get("noise_std", 0.05)),
        vib_noise_std=float(d.get("vib_noise_std", 0.15)),
        rho=float(d.get("rho", 1.2)),
        fault_classes=tuple(d.get("fault_classes", ["normal", "inner", "outer", "ball"])),
        n_harmonics=int(d.get("n_harmonics", 3)),
        seed=int(d.get("seed", 42)),
    )
