"""Lightweight training/experiment logging.

Provides a console logger and a metric logger that always writes a CSV history
and, when TensorBoard is installed, also mirrors scalars to a TensorBoard run.
The CSV fallback keeps the project runnable without TensorBoard.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "pinn_pdm", level: int = logging.INFO) -> logging.Logger:
    """Return a configured console logger (singleton per name)."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    _LOGGERS[name] = logger
    return logger


class MetricLogger:
    """Append-only scalar logger writing CSV (+ optional TensorBoard)."""

    def __init__(self, log_dir: str | Path, use_tensorboard: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "history.csv"
        self._fieldnames: list[str] = []
        self._rows: list[Dict[str, Any]] = []
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.log_dir))
            except Exception:  # pragma: no cover - optional dependency
                self._tb = None

    def log(self, step: int, metrics: Dict[str, float], prefix: str = "") -> None:
        """Record a dict of scalar metrics at ``step``."""
        row: Dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            key = f"{prefix}{k}" if prefix else k
            row[key] = float(v)
            if self._tb is not None:
                self._tb.add_scalar(key, float(v), step)
        for k in row:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        self._rows.append(row)
        self._flush_csv()

    def _flush_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
