"""Classical baselines: Random Forest and RBF-SVM on 12 handcrafted features.

Case Study 2 (Section 4) specifies RF/SVM on handcrafted time/frequency features
for the diagnosis task. Features are standardised (fit on train only) and the
classifiers are trained/evaluated with the *same* split as the neural models.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..evaluation.classification_metrics import classification_metrics


def train_classical(processed: Dict[str, np.ndarray], splits: Dict[str, np.ndarray],
                    class_names: List[str], seed: int = 42) -> Dict[str, Dict]:
    """Train RF and RBF-SVM; return per-model metrics on the test split."""
    X = processed["handcrafted"]
    y = processed["label"]
    tr, te = splits["train"], splits["test"]
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    results: Dict[str, Dict] = {}
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, y[tr])
    results["random_forest"] = classification_metrics(y[te], rf.predict(Xte), class_names)
    results["random_forest"]["n_params"] = None

    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    svm.fit(Xtr, y[tr])
    results["svm_rbf"] = classification_metrics(y[te], svm.predict(Xte), class_names)
    results["svm_rbf"]["n_params"] = None
    return results
