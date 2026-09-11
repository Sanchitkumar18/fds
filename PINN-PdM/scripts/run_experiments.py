#!/usr/bin/env python3
"""Run the full experiment suite and save all metrics/figures/tables.

Usage:
    python scripts/run_experiments.py --experiment all
    python scripts/run_experiments.py --experiment synthetic
    python scripts/run_experiments.py --experiment cwru --epochs 15
    python scripts/run_experiments.py --experiment ims
    python scripts/run_experiments.py --experiment ablation
    python scripts/run_experiments.py --experiment all --quick   # tiny budget smoke run
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from src.experiments import (run_ablation, run_cwru_experiment, run_ims_experiment,
                             run_synthetic_poc, verify_physics_frequencies)
from src.reporting import build_tables, generate_report
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("run_experiments")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all",
                    choices=["all", "synthetic", "cwru", "ims", "ablation", "cmapss", "verify"])
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for all runs")
    ap.add_argument("--quick", action="store_true", help="tiny budget for a smoke test")
    args = ap.parse_args()

    epochs = args.epochs
    if args.quick and epochs is None:
        epochs = 2

    syn = load_config("config/synthetic.yaml")
    cwru = load_config("config/cwru.yaml")
    ims = load_config("config/ims.yaml")
    set_seed(int(syn.get("seed", 42)))

    exp = args.experiment
    if exp in ("all", "verify"):
        v = verify_physics_frequencies(30.0)
        logger.info("Physics frequency verification: %s (match=%s)",
                    v["measured_ratios_x_fr"], v["match_within_0.01"])
        if exp == "verify":
            return

    if exp in ("all", "synthetic"):
        logger.info("=== Synthetic proof-of-concept ===")
        run_synthetic_poc(syn, epochs=epochs)
    if exp in ("all", "cwru"):
        logger.info("=== CWRU diagnosis + extrapolation ===")
        run_cwru_experiment(cwru, epochs=epochs)
    if exp in ("all", "ims"):
        logger.info("=== IMS RUL ===")
        run_ims_experiment(ims, epochs=epochs)
    if exp in ("all", "ablation"):
        logger.info("=== Ablation (synthetic testbed) ===")
        run_ablation(syn, epochs=epochs)
    if exp == "cmapss":   # optional/stretch; not part of 'all' to keep the core fast
        logger.info("=== C-MAPSS FD001 (optional) ===")
        from src.cmapss_experiment import run_cmapss_experiment
        run_cmapss_experiment(load_config("config/cmapps.yaml"), epochs=epochs)

    logger.info("Building tables + report ...")
    build_tables(syn)
    report = generate_report(syn)
    logger.info("Report written: %s", report)


if __name__ == "__main__":
    main()
