#!/usr/bin/env python3
"""Prepare the IMS processed dataset (STFT + features) and cache it.

Usage:
    python scripts/prepare_ims.py [--config config/ims.yaml] [--force]
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import numpy as np

from src.pipeline import prepare_dataset
from src.utils.config import load_config, resolve_path
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("prepare_ims")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/ims.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    processed, path = prepare_dataset(cfg, force=args.force)
    n = processed["stft"].shape[0]
    synth = bool(int(processed.get("_synthetic", [1])[0]))
    n_units = int(np.unique(processed["unit_id"]).size)
    logger.info("IMS processed: %d windows across %d units, synthetic_fallback=%s",
                n, n_units, synth)

    # Degradation-trend figure: mean health per unit over time.
    from src.visualization import rul_plots
    figs = resolve_path(cfg["paths"]["output_dir"]) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    u0 = int(np.unique(processed["unit_id"])[0])
    mask = processed["unit_id"] == u0
    t = processed["t"][mask]; h = processed["health"][mask]
    order = np.argsort(t)
    rul_plots.plot_health_trajectory(t[order], h[order], h[order],
                                     figs / "ims_degradation_trend.png",
                                     title=f"IMS degradation trend (unit {u0})")
    logger.info("Wrote IMS EDA figure to %s", figs)


if __name__ == "__main__":
    main()
