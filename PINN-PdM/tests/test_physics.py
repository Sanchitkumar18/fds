"""Tests for bearing physics, degradation ODE, autodiff and physics losses."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.bearing_frequencies import (SKF_6205, all_fault_frequencies, bpfi,
                                             bpfo, bsf, fault_frequency_ratios, ftf,
                                             harmonics)
from src.physics.degradation import estimate_rho, exponential_health, health_to_rul
from src.physics.physics_losses import (build_frequency_masks, health_derivative,
                                        monotonicity_loss, physics_ode_loss,
                                        spectral_consistency_loss)


def test_skf6205_frequency_ratios():
    """Physics module recovers the textbook SKF-6205 ratios (Case Study 6.7)."""
    ratios = fault_frequency_ratios(SKF_6205)
    assert abs(ratios["outer"] - 3.585) < 0.01
    assert abs(ratios["inner"] - 5.415) < 0.01
    assert abs(ratios["ball"] - 2.357) < 0.01


def test_bpfo_plus_bpfi_equals_n_times_fr():
    """Identity: BPFO + BPFI = n * fr (cos terms cancel)."""
    fr = 30.0
    assert abs((bpfo(fr, SKF_6205) + bpfi(fr, SKF_6205)) - SKF_6205.n_balls * fr) < 1e-6


def test_frequencies_scale_with_shaft_speed():
    assert abs(bpfo(60.0, SKF_6205) - 2 * bpfo(30.0, SKF_6205)) < 1e-6


def test_harmonics():
    assert harmonics(100.0, 3) == [100.0, 200.0, 300.0]


def test_exponential_health_decays():
    t = np.linspace(0, 1, 20)
    h = exponential_health(t, rho=1.2)
    assert h[0] == pytest.approx(1.0)
    assert np.all(np.diff(h) < 0)  # strictly decreasing


def test_estimate_rho_recovers_rate():
    t = np.linspace(0, 2, 50)
    h = exponential_health(t, rho=0.8)
    assert estimate_rho(t, h) == pytest.approx(0.8, abs=1e-3)


def test_health_to_rul_monotone():
    t = np.linspace(0, 1, 10)
    h = exponential_health(t, rho=1.0)
    rul = health_to_rul(h)
    assert np.all(np.diff(rul) <= 1e-9)
    assert rul[0] == pytest.approx(1.0)


def test_autodiff_derivative_matches_analytic():
    """dH/dt via autograd matches the analytic derivative of H = exp(-rho t)."""
    rho = 1.3
    t = torch.linspace(0.1, 1.0, 16).unsqueeze(1).requires_grad_(True)
    H = torch.exp(-rho * t)
    dHdt = health_derivative(H, t)
    analytic = -rho * torch.exp(-rho * t)
    assert torch.allclose(dHdt, analytic, atol=1e-5)


def test_ode_residual_zero_for_true_solution():
    """For H=exp(-rho t), the residual dH/dt + rho*H is ~0."""
    rho = torch.tensor(0.9)
    t = torch.linspace(0.1, 1.0, 16).unsqueeze(1).requires_grad_(True)
    H = torch.exp(-0.9 * t)
    loss = physics_ode_loss(H, t, rho)
    assert loss.item() < 1e-6


def test_monotonicity_loss_penalises_increase():
    t = torch.linspace(0, 1, 16).unsqueeze(1).requires_grad_(True)
    H_increasing = t.clone()          # dH/dt = +1 -> penalised
    H_decreasing = -t.clone()         # dH/dt = -1 -> not penalised
    assert monotonicity_loss(H_increasing, t).item() > 0.5
    assert monotonicity_loss(H_decreasing, t).item() < 1e-6


def test_spectral_mask_shapes_and_alignment():
    freq_axis = np.linspace(0, 6000, 64)
    freqs = all_fault_frequencies(30.0, SKF_6205)
    masks = build_frequency_masks(freq_axis, [None, freqs["inner"], freqs["outer"], freqs["ball"]],
                                  n_harmonics=3, bandwidth=40.0)
    assert masks.shape == (4, 64)
    assert masks[0].sum() == 0.0  # 'normal' has no defect band

    # Energy concentrated at the outer band + high prob on outer class -> low loss.
    masks_t = torch.tensor(masks)
    valid = torch.tensor([0.0, 1.0, 1.0, 1.0])
    energy = torch.tensor(masks[2:3]).repeat(2, 1)  # (2, F) peaked at outer band
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    aligned = spectral_consistency_loss(energy, probs, masks_t, valid).item()
    # Misaligned: same energy but predict inner class -> higher loss.
    probs_wrong = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    misaligned = spectral_consistency_loss(energy, probs_wrong, masks_t, valid).item()
    assert aligned < misaligned
