#!/usr/bin/env python3
"""Evaluate a trained model from its checkpoint and print/save metrics.

Usage:
    python scripts/evaluate.py --experiment cwru
    python scripts/evaluate.py --experiment ims

Loads the saved PINN checkpoint for the dataset, rebuilds the test split, and
recomputes metrics (diagnosis or RUL depending on task). This verifies
checkpoint save/load and provides a standalone evaluation entry point.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from src.evaluation.classification_metrics import classification_metrics
from src.evaluation.predict import collect_predictions, health_by_unit
from src.evaluation.physics_metrics import monotonicity_report
from src.evaluation.rul_metrics import rul_metrics
from src.models.factory import build_model
from src.pipeline import make_loaders, make_splits, prepare_dataset, shaft_hz_for
from src.utils.checkpoint import load_checkpoint
from src.utils.config import get_device, load_config, resolve_path
from src.utils.logging import get_logger

logger = get_logger("evaluate")

_DATASET_CONFIG = {"synthetic": "config/synthetic.yaml", "cwru": "config/cwru.yaml",
                   "ims": "config/ims.yaml"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="cwru", choices=list(_DATASET_CONFIG))
    ap.add_argument("--model", default="pinn_pdm")
    args = ap.parse_args()

    cfg = load_config(_DATASET_CONFIG[args.experiment])
    device = get_device(cfg.get("device", "auto"))
    task = cfg.get("task", "both")
    processed, _ = prepare_dataset(cfg)
    splits = make_splits(cfg, processed)
    loaders = make_loaders(cfg, processed, splits, task)

    # Match the pipeline's n_classes logic so the diagnosis head matches the
    # checkpoint (RUL datasets have a single label but a 4-class head by default).
    class_names = list(cfg.get(cfg["dataset"], {}).get("fault_classes",
                       cfg.get("synthetic", {}).get("fault_classes", ["normal", "inner", "outer", "ball"])))
    n_classes = max(int(processed["label"].max() + 1), len(class_names))
    model = build_model(args.model, cfg, n_classes=n_classes,
                        n_features=int(cfg["preprocess"].get("n_features", 4)))
    ckpt = resolve_path(cfg["paths"]["checkpoint_dir"]) / f"{cfg['dataset']}_{args.model}.pt"
    if not ckpt.exists():
        logger.error("Checkpoint not found: %s. Train first (scripts/train_pinn.py).", ckpt)
        return
    meta = load_checkpoint(ckpt, model, map_location=str(device))
    model.to(device)
    logger.info("Loaded checkpoint %s (epoch %s)", ckpt, meta.get("epoch"))

    pred = collect_predictions(model, loaders["test"], device, task)
    result = {}
    if task in ("diagnosis", "both") and "y_pred" in pred:
        cm = classification_metrics(pred["y_true"], pred["y_pred"])
        result["classification"] = {"accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"]}
        logger.info("Accuracy=%.4f macro-F1=%.4f", cm["accuracy"], cm["macro_f1"])
    if task in ("prognosis", "both") and "health_pred" in pred:
        from src.experiments import _health_to_rul_pred
        rm = rul_metrics(pred["rul_true"], _health_to_rul_pred(pred))
        mono = monotonicity_report(health_by_unit(pred))
        result["rul"] = rm
        result["monotonicity"] = mono
        logger.info("RUL RMSE=%.3f MAE=%.3f R2=%.3f PHM=%.1f | mono viol=%.1f%%",
                    rm["rmse"], rm["mae"], rm["r2"], rm["phm_score"], mono["pct_sequences_violating"])

    out = resolve_path(cfg["paths"]["output_dir"]) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"evaluate_{cfg['dataset']}_{args.model}.json", "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
