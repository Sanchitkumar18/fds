#!/usr/bin/env python3
"""Build result tables (CSV + Markdown) and the technical report from metrics.

Usage:
    python scripts/generate_report.py

Reads outputs/metrics/*.json (produced by run_experiments.py) and writes:
    outputs/tables/*.csv, outputs/report/technical_report.md, outputs/report/tables.md
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from src.reporting import build_tables, generate_report
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("generate_report")


def main() -> None:
    cfg = load_config("config/synthetic.yaml")
    tables = build_tables(cfg)
    if not tables:
        logger.warning("No metrics found. Run: python scripts/run_experiments.py --experiment all")
    for name in tables:
        logger.info("Built table: %s", name)
    report = generate_report(cfg)
    logger.info("Report: %s", report)


if __name__ == "__main__":
    main()
