"""RUL / health-regression metrics, including the PHM08 asymmetric score."""
from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray,
                a1: float = 13.0, a2: float = 10.0) -> float:
    """PHM08 asymmetric score (lower is better).

    d = y_pred - y_true (predicted RUL minus true RUL). Late predictions
    (d < 0, i.e. the model over-estimates remaining life -- dangerous) are
    penalised more heavily than early ones:

        s = sum( exp(-d/a1) - 1 )  if d < 0   (late, harsher: a1=13)
            sum( exp( d/a2) - 1 )  if d >= 0  (early, milder:  a2=10)

    Reference: Saxena et al. (2008), PHM08 (Case Study ref [11]).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / a1) - 1.0, np.exp(d / a2) - 1.0)
    return float(np.sum(score))


def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Bundle RMSE, MAE, R2 and the PHM08 score."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "phm_score": phm08_score(y_true, y_pred),
    }


def health_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Health-index regression metrics (RMSE/MAE/R2) for the synthetic testbed."""
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)}
