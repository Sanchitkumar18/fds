#!/usr/bin/env python3
"""Train the PINN-PdM model on a dataset.

Usage:
    python scripts/train_pinn.py --config config/cwru.yaml
    python scripts/train_pinn.py --dataset ims [--epochs 10]

Saves a checkpoint under checkpoints/<dataset>_pinn_pdm.pt and a metrics summary
under outputs/metrics/train_pinn_<dataset>.json.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from src.evaluation.predict import collect_predictions
from src.pipeline import make_splits, prepare_dataset, train_model
from src.utils.config import get_device, load_config, resolve_path
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_pinn")

_DATASET_CONFIG = {"synthetic": "config/synthetic.yaml", "cwru": "config/cwru.yaml",
                   "ims": "config/ims.yaml"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, choices=list(_DATASET_CONFIG))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--regime", default=None, help="override split regime")
    args = ap.parse_args()

    cfg_path = args.config or _DATASET_CONFIG[args.dataset or "synthetic"]
    cfg = load_config(cfg_path)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    task = cfg.get("task", "both")

    processed, _ = prepare_dataset(cfg)
    splits = make_splits(cfg, processed, regime=args.regime)
    logger.info("Training PINN-PdM on %s (task=%s) | train=%d val=%d test=%d",
                cfg["dataset"], task, len(splits["train"]), len(splits["val"]), len(splits["test"]))
    res = train_model(cfg, "pinn_pdm", task, processed, splits, device, is_pinn=True)

    summary = {"dataset": cfg["dataset"], "task": task, "n_params": res["n_params"],
               "train_time_s": res["train_time_s"], "final_weights": res["final_weights"],
               "checkpoint": res["checkpoint"], "device": res["device"]}
    if hasattr(res["model"], "rho"):
        summary["final_rho"] = float(res["model"].rho.item())
    out = resolve_path(cfg["paths"]["output_dir"]) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"train_pinn_{cfg['dataset']}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Done. Checkpoint: %s", res["checkpoint"])


if __name__ == "__main__":
    main()
