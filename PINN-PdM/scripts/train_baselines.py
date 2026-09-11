#!/usr/bin/env python3
"""Train the baseline models on a dataset.

Usage:
    python scripts/train_baselines.py --dataset cwru [--epochs 10]
    python scripts/train_baselines.py --dataset ims

For diagnosis datasets it trains RF, SVM, CNN, LSTM, CNN-LSTM and Transformer;
for prognosis datasets it trains the neural sequence baselines. Metrics are
saved under outputs/metrics/baselines_<dataset>.json.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from src.evaluation.classification_metrics import classification_metrics
from src.evaluation.predict import collect_predictions, health_by_unit, value_by_unit
from src.evaluation.physics_metrics import monotonicity_report
from src.evaluation.rul_metrics import rul_metrics
from src.models.classical import train_classical
from src.pipeline import make_splits, prepare_dataset, train_model
from src.utils.config import get_device, load_config, resolve_path
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_baselines")

_DATASET_CONFIG = {"synthetic": "config/synthetic.yaml", "cwru": "config/cwru.yaml",
                   "ims": "config/ims.yaml"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cwru", choices=list(_DATASET_CONFIG))
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config or _DATASET_CONFIG[args.dataset])
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    task = cfg.get("task", "both")
    processed, _ = prepare_dataset(cfg)
    splits = make_splits(cfg, processed)

    results = {}
    if task in ("diagnosis", "both"):
        class_names = list(cfg.get(cfg["dataset"], {}).get("fault_classes",
                           ["normal", "inner", "outer", "ball"]))
        logger.info("Training classical baselines (RF, SVM) ...")
        classical = train_classical(processed, splits, class_names, seed=int(cfg.get("seed", 42)))
        for name, m in classical.items():
            results[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    neural = ["cnn", "lstm", "cnn_lstm", "transformer"] if task != "prognosis" \
        else ["lstm", "cnn_lstm", "transformer"]
    import copy
    for name in neural:
        logger.info("Training baseline: %s", name)
        res = train_model(cfg, name, task, copy.deepcopy(processed), splits, device, is_pinn=False)
        pred = collect_predictions(res["model"], res["loaders"]["test"], device, task)
        entry = {"n_params": res["n_params"], "train_time_s": res["train_time_s"]}
        if task in ("diagnosis", "both") and "y_pred" in pred:
            cm = classification_metrics(pred["y_true"], pred["y_pred"])
            entry.update({"accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"]})
        if task in ("prognosis", "both") and "health_pred" in pred:
            entry["monotonicity_pct"] = monotonicity_report(health_by_unit(pred))["pct_sequences_violating"]
        results[name] = entry

    out = resolve_path(cfg["paths"]["output_dir"]) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"baselines_{cfg['dataset']}.json"
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved baseline metrics -> %s", path)


if __name__ == "__main__":
    main()
