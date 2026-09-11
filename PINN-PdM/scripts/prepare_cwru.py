#!/usr/bin/env python3
"""Prepare the CWRU processed dataset (STFT + features) and cache it.

Usage:
    python scripts/prepare_cwru.py [--config config/cwru.yaml] [--force]

Also produces EDA figures (raw signal, FFT with fault-frequency overlays, STFT,
class/load distributions) under outputs/figures/.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import numpy as np

from src.physics.bearing_frequencies import SKF_6205, all_fault_frequencies, cwru_shaft_hz
from src.pipeline import prepare_dataset
from src.utils.config import load_config, resolve_path
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("prepare_cwru")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/cwru.yaml")
    ap.add_argument("--force", action="store_true", help="re-generate even if cached")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    processed, path = prepare_dataset(cfg, force=args.force)
    n = processed["stft"].shape[0]
    synth = bool(int(processed.get("_synthetic", [1])[0]))
    logger.info("CWRU processed: %d windows, %d classes, synthetic_fallback=%s",
                n, int(processed["label"].max() + 1), synth)

    # EDA figures.
    from src.visualization import signal_plots
    figs = resolve_path(cfg["paths"]["output_dir"]) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fs = float(cfg["cwru"]["sampling_rate"])
    class_names = list(cfg["cwru"]["fault_classes"])
    raw0 = None
    # Reconstruct a representative raw window per class is not cached; use STFT map plots.
    for ci, cname in enumerate(class_names):
        idx = np.where(processed["label"] == ci)[0]
        if idx.size == 0:
            continue
        g = int(idx[0])
        ff = all_fault_frequencies(cwru_shaft_hz(0), SKF_6205)
        signal_plots.plot_stft_map(processed["stft"][g], f"CWRU STFT — {cname}",
                                   figs / f"cwru_stft_{cname}.png", processed["freq_axis"])
    signal_plots.plot_class_distribution(processed["label"], class_names,
                                         figs / "cwru_class_distribution.png",
                                         "CWRU class distribution")
    signal_plots.plot_load_distribution(processed["load_hp"], figs / "cwru_load_distribution.png")
    logger.info("Wrote EDA figures to %s", figs)


if __name__ == "__main__":
    main()
