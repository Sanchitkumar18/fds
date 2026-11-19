"""Tests for the Review-03 additions: split-before-windowing and the L_spec audit."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.review03.data_real import (Recording, Segment, assert_no_overlap, build_arrays,
                                    spectral_views, time_block_segments)
from src.review03.spectral import build_masks, spectral_loss_v1, spectral_loss_v2, valid_mask

RATIOS = [None, 5.415, 3.585, 2.357]   # healthy, inner, outer, ball (SKF-6205)


def _rec(n=40000, fs=12000.0, label=0):
    rng = np.random.default_rng(label)
    return Recording(rec_id=f"r{label}", signal=rng.normal(size=n).astype(np.float32), fs=fs,
                     label=label, cls="x", load=0.0, shaft_hz=29.95)


def test_time_blocks_have_guard_and_no_window_leakage():
    r = _rec()
    segs = time_block_segments(r, [("train", .7), ("val", .15), ("test", .15)], guard=2048)
    for a, b in zip(segs, segs[1:]):
        assert b.start - a.stop >= 2048
    arr, _ = build_arrays(segs, 2048, 1024)
    for k, v in arr.items():
        assert (v.start >= min(s.start for s in segs if s.split == k)).all()
        assert (v.start + 2048 <= max(s.stop for s in segs if s.split == k)).all()


def test_overlap_detector_catches_leakage():
    idx = {"train": {"rec": np.array(["a"]), "start": np.array([0])},
           "test": {"rec": np.array(["a"]), "start": np.array([1024])}}
    with pytest.raises(AssertionError):
        assert_no_overlap(idx, 2048)


def test_envelope_spectrum_recovers_impact_rate():
    fs, n, rate = 12000.0, 2048 * 8, 107.4
    t = np.arange(n) / fs
    x = np.zeros(n)
    for t0 in np.arange(0, t[-1], 1 / rate):          # impacts exciting a 3 kHz resonance
        k = t >= t0
        x[k] += np.exp(-800 * (t[k] - t0)) * np.sin(2 * np.pi * 3000 * (t[k] - t0))
    w = x[:2048][None].astype(np.float32)
    _, env, f = spectral_views(w, fs)
    peak = f[1:][np.argmax(env[0, 1:])]
    assert abs(peak - rate) < 6.0                     # within one 5.9 Hz bin


def _batch(B=16, F=171):
    torch.manual_seed(0)
    spec = torch.rand(B, F)
    freqs = np.linspace(0, 1000, F).astype(np.float32)
    shaft = torch.full((B,), 29.95)
    return spec, freqs, shaft


def test_v2_is_invariant_to_healthy_logit():
    spec, freqs, shaft = _batch()
    valid = valid_mask(RATIOS, "cpu")
    m = build_masks(freqs, RATIOS, shaft, 3, 5.0, "sum")
    logits = torch.randn(spec.shape[0], 4, requires_grad=True)
    g, = torch.autograd.grad(spectral_loss_v2(spec, logits, m, valid), logits)
    assert torch.allclose(g[:, 0], torch.zeros_like(g[:, 0]))
    assert g[:, 1:].abs().sum() > 0


def test_v1_gradient_leaks_into_healthy_logit():
    spec, freqs, shaft = _batch()
    valid = valid_mask(RATIOS, "cpu")
    m = build_masks(freqs, RATIOS, shaft, 3, 5.0, "max")
    logits = torch.randn(spec.shape[0], 4, requires_grad=True)
    g, = torch.autograd.grad(spectral_loss_v1(spec, torch.softmax(logits, 1), m, valid), logits)
    assert g[:, 0].abs().sum() > 0


def test_v2_masks_have_equal_mass_and_zero_healthy_row():
    _, freqs, shaft = _batch()
    m = build_masks(freqs, RATIOS, shaft, 3, 5.0, "sum")
    assert torch.allclose(m[:, 0], torch.zeros_like(m[:, 0]))
    assert torch.allclose(m[:, 1:].sum(-1), torch.ones_like(m[:, 1:].sum(-1)), atol=1e-5)


def test_v2_descent_moves_mass_to_class_with_spectral_evidence():
    F = 171
    freqs = np.linspace(0, 1000, F).astype(np.float32)
    shaft = torch.tensor([29.95])
    bpfo = 3.585 * 29.95
    spec = torch.tensor(np.exp(-0.5 * ((freqs - bpfo) / 3.0) ** 2), dtype=torch.float32)[None]
    valid = valid_mask(RATIOS, "cpu")
    m = build_masks(freqs, RATIOS, shaft, 3, 5.0, "sum")
    logits = torch.zeros(1, 4, requires_grad=True)
    g, = torch.autograd.grad(spectral_loss_v2(spec, logits, m, valid), logits)
    assert int(torch.argmin(g[0, 1:])) + 1 == 2       # outer-race logit increases fastest
