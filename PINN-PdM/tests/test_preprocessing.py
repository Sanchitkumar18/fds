"""Tests for segmentation, STFT dimensions, features and split leakage."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_extraction import (batch_core_features, core_features,
                                         handcrafted_features)
from src.data.preprocessing import StandardScaler1D, compute_stft_map
from src.data.segmentation import segment_signal, segment_with_metadata
from src.data.splits import by_unit_split, cross_condition_split, stratified_group_split


def test_segmentation_50pct_overlap():
    sig = np.arange(2048 * 3, dtype=np.float32)
    wins = segment_signal(sig, window_size=2048, overlap=0.5)
    # hop = 1024; windows start at 0,1024,2048,3072,4096 -> 5 full windows.
    assert wins.shape == (5, 2048)
    assert np.allclose(wins[1, 0], 1024)


def test_segmentation_short_signal_dropped():
    sig = np.zeros(1000, dtype=np.float32)
    wins = segment_signal(sig, window_size=2048, overlap=0.5)
    assert wins.shape == (0, 2048)


def test_stft_map_is_64x64():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(2048)
    m, freq_axis = compute_stft_map(sig, fs=12000, out_size=64)
    assert m.shape == (64, 64)
    assert freq_axis.shape == (64,)
    assert 0.0 <= m.min() and m.max() <= 1.0 + 1e-5


def test_core_features_shape_and_values():
    sig = np.ones(2048, dtype=np.float32)
    feats = core_features(sig)
    assert feats.shape == (4,)
    assert feats[0] == pytest.approx(1.0, abs=1e-3)  # RMS of ones == 1


def test_handcrafted_features_length_12():
    rng = np.random.default_rng(1)
    feats = handcrafted_features(rng.standard_normal(2048), fs=12000)
    assert feats.shape == (12,)
    assert np.isfinite(feats).all()


def test_features_handle_nan_inf():
    sig = np.array([np.nan, np.inf, -np.inf] + [0.0] * 2045, dtype=np.float32)
    feats = core_features(sig)
    assert np.isfinite(feats).all()


def test_scaler_fit_on_train_only():
    x = np.random.RandomState(0).randn(100, 4) * 5 + 3
    scaler = StandardScaler1D().fit(x[:70])
    xt = scaler.transform(x[:70])
    assert abs(xt.mean()) < 1e-5
    assert abs(xt.std() - 1.0) < 1e-1


def test_stratified_group_split_no_leakage():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3] * 5)
    groups = np.repeat(np.arange(40), 1)
    tr, va, te = stratified_group_split(labels, groups, seed=0)
    assert set(groups[tr]).isdisjoint(groups[va])
    assert set(groups[tr]).isdisjoint(groups[te])
    assert set(groups[va]).isdisjoint(groups[te])


def test_cross_condition_split_holds_out_load():
    load = np.array([0, 1, 2, 3] * 25)
    groups = np.arange(100)
    tr, va, te = cross_condition_split(load, [0, 1, 2], [3], groups=groups, seed=0)
    assert np.all(load[te] == 3)
    assert not np.isin(load[tr], [3]).any()
    assert not np.isin(load[va], [3]).any()


def test_by_unit_split_no_unit_leakage():
    units = np.repeat(np.arange(10), 20)
    tr, te = by_unit_split(units, test_frac=0.3, seed=0)
    assert set(units[tr]).isdisjoint(units[te])
