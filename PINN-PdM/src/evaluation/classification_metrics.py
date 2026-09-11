"""Fault-diagnosis (classification) metrics."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: Optional[List[str]] = None) -> Dict:
    """Compute accuracy, macro-F1, precision, recall and the confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names))) if class_names else None
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if class_names:
        metrics["per_class_f1"] = f1_score(y_true, y_pred, average=None,
                                           labels=labels, zero_division=0).tolist()
        metrics["class_names"] = list(class_names)
    return metrics


@np.errstate(all="ignore")
def predict_logits_to_labels(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argmax(axis=1)
