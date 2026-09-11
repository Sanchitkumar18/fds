"""Train/val/test splitting with leakage prevention.

Implements the two CWRU regimes from Case Study 2 (Section 5) plus the IMS
by-unit split:

* ``stratified_group_split`` -- in-distribution 70/15/15, stratified by class,
  grouped by *record* so overlapping windows of one record never straddle the
  split (Case Study requirement #6, #28).
* ``cross_condition_split`` -- train on a set of loads, test on a held-out load
  (the critical extrapolation regime: train 0/1/2 HP, test 3 HP).
* ``by_unit_split`` -- split IMS sequences by bearing unit so no unit appears in
  both train and test.

Each function returns index arrays into the sample dimension and every function
asserts the absence of leakage.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def stratified_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split that keeps whole groups (records) together.

    Groups are assigned to splits (not individual windows). Stratification is
    approximate: within each class the groups are shuffled and partitioned by
    the requested fractions.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "fractions must sum to 1"
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    train_groups: List[int] = []
    val_groups: List[int] = []
    test_groups: List[int] = []

    # Determine dominant label per group for stratification.
    unique_groups = np.unique(groups)
    group_label = {}
    for g in unique_groups:
        gl = labels[groups == g]
        group_label[g] = np.bincount(gl).argmax()

    by_class: Dict[int, List[int]] = {}
    for g in unique_groups:
        by_class.setdefault(int(group_label[g]), []).append(int(g))

    for cls, gs in by_class.items():
        gs = np.array(gs)
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(n * train))
        n_va = int(round(n * val))
        # Guard tiny classes so no split is starved.
        n_tr = max(1, min(n_tr, n - 2)) if n >= 3 else n
        n_va = max(0, min(n_va, n - n_tr - 1)) if n >= 3 else 0
        train_groups.extend(gs[:n_tr].tolist())
        val_groups.extend(gs[n_tr:n_tr + n_va].tolist())
        test_groups.extend(gs[n_tr + n_va:].tolist())

    tr_mask = np.isin(groups, train_groups)
    va_mask = np.isin(groups, val_groups)
    te_mask = np.isin(groups, test_groups)
    idx_tr = np.where(tr_mask)[0]
    idx_va = np.where(va_mask)[0]
    idx_te = np.where(te_mask)[0]

    _assert_no_group_leakage(groups, idx_tr, idx_va, idx_te)
    return idx_tr, idx_va, idx_te


def cross_condition_split(
    load_hp: np.ndarray,
    train_loads: Sequence[int],
    test_loads: Sequence[int],
    val_frac: float = 0.15,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-condition extrapolation split by operating load.

    Windows whose load is in ``train_loads`` form train+val; those in
    ``test_loads`` form the held-out test set. A validation slice is carved from
    the training loads (grouped by record when ``groups`` is provided).
    """
    load_hp = np.asarray(load_hp)
    rng = np.random.default_rng(seed)
    train_pool = np.where(np.isin(load_hp, list(train_loads)))[0]
    test_idx = np.where(np.isin(load_hp, list(test_loads)))[0]

    if groups is not None:
        pool_groups = np.unique(groups[train_pool])
        rng.shuffle(pool_groups)
        n_val = int(round(len(pool_groups) * val_frac))
        val_groups = set(pool_groups[:n_val].tolist())
        val_idx = np.array([i for i in train_pool if groups[i] in val_groups], dtype=int)
        train_idx = np.array([i for i in train_pool if groups[i] not in val_groups], dtype=int)
    else:
        perm = rng.permutation(train_pool)
        n_val = int(round(len(perm) * val_frac))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    # Assert extrapolation integrity: no test load appears in train/val.
    assert not np.isin(load_hp[train_idx], list(test_loads)).any(), "test load leaked into train"
    assert not np.isin(load_hp[val_idx], list(test_loads)).any(), "test load leaked into val"
    return train_idx, val_idx, test_idx


def by_unit_split(
    unit_ids: np.ndarray,
    test_frac: float = 0.30,
    seed: int = 42,
    train_units: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split run-to-failure samples by bearing unit (IMS).

    Ensures no unit appears in both train and test (Case Study requirement #16,
    #28). Returns (train_idx, test_idx).
    """
    unit_ids = np.asarray(unit_ids)
    units = np.unique(unit_ids)
    rng = np.random.default_rng(seed)
    if train_units is not None:
        train_set = set(int(u) for u in train_units)
    else:
        shuffled = units.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(units) * test_frac)))
        test_set = set(shuffled[:n_test].tolist())
        train_set = set(shuffled[n_test:].tolist())
    train_idx = np.where(np.isin(unit_ids, list(train_set)))[0]
    test_idx = np.where(~np.isin(unit_ids, list(train_set)))[0]

    train_units_present = set(unit_ids[train_idx].tolist())
    test_units_present = set(unit_ids[test_idx].tolist())
    assert train_units_present.isdisjoint(test_units_present), \
        "bearing unit leaked across train/test split"
    return train_idx, test_idx


def _assert_no_group_leakage(groups: np.ndarray, *index_sets: np.ndarray) -> None:
    """Assert that no group id is shared across the provided index sets."""
    seen_groups = [set(groups[idx].tolist()) for idx in index_sets]
    for i in range(len(seen_groups)):
        for j in range(i + 1, len(seen_groups)):
            overlap = seen_groups[i] & seen_groups[j]
            assert not overlap, f"group leakage between split {i} and {j}: {overlap}"
