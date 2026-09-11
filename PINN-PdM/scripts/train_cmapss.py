#!/usr/bin/env python3
"""Train the optional C-MAPSS FD001 RUL experiment (stretch goal).

Usage:
    python scripts/train_cmapss.py [--epochs 20]

Trains a Random Forest reference, an unconstrained MLP baseline and a PINN-MLP
with the degradation-ODE + monotonicity prior; saves outputs/metrics/cmapss.json.
Uses a synthetic multivariate stand-in if the raw FD001 files are absent.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from src.cmapss_experiment import run_cmapss_experiment
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("train_cmapss")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/cmapps.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = run_cmapss_experiment(cfg, epochs=args.epochs)
    m = res["measured"]
    logger.info("Done. RF=%.2f baseline=%.2f PINN=%.2f (RMSE)",
                m["random_forest"]["rmse"], m["unconstrained_mlp"]["rmse"], m["pinn_mlp"]["rmse"])


if __name__ == "__main__":
    main()
