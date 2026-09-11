"""Tests for classification/RUL/physics metrics and the PHM08 score."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classification_metrics import classification_metrics
from src.evaluation.physics_metrics import monotonicity_report
from src.evaluation.rul_metrics import mae, phm08_score, r2_score, rmse, rul_metrics


def test_perfect_classification():
    y = np.array([0, 1, 2, 3, 0, 1])
    m = classification_metrics(y, y, ["a", "b", "c", "d"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_rmse_mae_zero_for_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2_score(y, y) == pytest.approx(1.0)


def test_phm08_penalises_late_more_than_early():
    y_true = np.array([50.0])
    # Late prediction (over-estimates RUL): pred > true, d>0 in our convention.
    late = phm08_score(y_true, np.array([60.0]))   # d = +10 -> exp(10/10)-1
    early = phm08_score(y_true, np.array([40.0]))  # d = -10 -> exp(10/13)-1
    assert late > early  # late (dangerous) penalised more


def test_phm08_zero_for_perfect():
    y = np.array([10.0, 20.0, 30.0])
    assert phm08_score(y, y) == pytest.approx(0.0, abs=1e-9)


def test_rul_metrics_bundle_keys():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    m = rul_metrics(y_true, y_pred)
    assert set(m) == {"rmse", "mae", "r2", "phm_score"}


def test_monotonicity_report_counts_violations():
    # Unit 0 monotone decreasing (no violation); unit 1 has one increase.
    health = {0: np.array([1.0, 0.8, 0.6, 0.4]),
              1: np.array([1.0, 0.9, 0.95, 0.5])}
    rep = monotonicity_report(health)
    assert rep["n_units"] == 2
    assert rep["pct_sequences_violating"] == pytest.approx(50.0)
    assert rep["n_violating_steps"] == 1
    assert rep["mean_violation_magnitude"] == pytest.approx(0.05, abs=1e-6)
