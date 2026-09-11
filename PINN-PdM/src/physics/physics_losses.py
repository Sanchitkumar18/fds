"""Differentiable physics losses for PINN-PdM.

Three physics terms are implemented, matching Case Study 2:

1. ``degradation_ode_residual`` / ``physics_ode_loss`` -- the first-order
   degradation ODE residual ``dH/dt - rho*H`` evaluated by autograd
   (Section 2.3). ``L_phys = mean(r_phys^2)``.

2. ``monotonicity_loss`` -- penalises a *positive* health derivative so the
   predicted health cannot spontaneously recover: ``L_mono = mean(ReLU(dH/dt)^2)``
   (Section 2.3).

3. ``spectral_consistency_loss`` -- the diagnosis prior (Section 2.2). For each
   window it builds a soft Gaussian weighting mask around the theoretical
   characteristic frequency (and harmonics) of the *target/predicted* class and
   rewards STFT energy that concentrates in that band. Implemented with a soft
   (differentiable) class weighting rather than a hard argmax.

All functions return scalar tensors and are numerically stabilised.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Degradation ODE residual + monotonicity
# ---------------------------------------------------------------------------
def health_derivative(health, t):
    """Compute exact ``dH/dt`` via automatic differentiation.

    Parameters
    ----------
    health:
        Tensor of shape (B, 1) or (B,) -- the network's health output H(t).
    t:
        The *same* time leaf tensor (``requires_grad=True``) that produced
        ``health``. Must be shape (B, 1).

    Returns
    -------
    dHdt: tensor of shape (B, 1)
    """
    import torch

    if not t.requires_grad:
        raise ValueError("time tensor `t` must have requires_grad=True for autodiff")
    grad = torch.autograd.grad(
        outputs=health.sum(),
        inputs=t,
        create_graph=True,   # keep graph so the residual is itself differentiable
        retain_graph=True,
    )[0]
    return grad


def degradation_ode_residual(health, t, rho):
    """Return the ODE residual ``r = dH/dt + rho*H`` (decaying convention).

    Case Study 2 writes the residual as ``dH/dt - rho*H`` with the understanding
    that health decays, i.e. the physical solution is ``H = H0 exp(-rho t)`` so
    ``dH/dt = -rho H``. We therefore enforce ``dH/dt + |rho| H = 0``; ``rho`` is
    passed already constrained positive by the model (softplus). Both forms are
    algebraically equivalent up to the sign convention of ``rho`` and are
    documented here to avoid ambiguity (Case Study requirement #37).
    """
    dHdt = health_derivative(health, t)
    if health.dim() == 1:
        health = health.unsqueeze(1)
    return dHdt + rho * health


def physics_ode_loss(health, t, rho):
    """``L_phys = mean(r_phys^2)`` for the degradation ODE."""
    import torch

    r = degradation_ode_residual(health, t, rho)
    return torch.mean(r ** 2)


def monotonicity_loss(health, t):
    """``L_mono = mean(ReLU(dH/dt)^2)`` -- forbid increasing health."""
    import torch

    dHdt = health_derivative(health, t)
    return torch.mean(torch.relu(dHdt) ** 2)


def monotonicity_from_derivative(dHdt):
    """Monotonicity loss given a precomputed derivative tensor."""
    import torch

    return torch.mean(torch.relu(dHdt) ** 2)


# ---------------------------------------------------------------------------
# Spectral consistency loss (diagnosis prior)
# ---------------------------------------------------------------------------
# Mapping from fault-class name to the characteristic-frequency key used by
# ``bearing_frequencies.all_fault_frequencies``. "normal"/"healthy" have no
# characteristic defect frequency and are excluded from the spectral penalty.
CLASS_TO_FREQ_KEY = {
    "outer": "outer", "outer_race": "outer", "or": "outer",
    "inner": "inner", "inner_race": "inner", "ir": "inner",
    "ball": "ball", "roller": "ball", "b": "ball",
    "cage": "cage", "ftf": "cage",
}


def build_frequency_masks(
    freq_axis: np.ndarray,
    class_freqs: Sequence[Optional[float]],
    n_harmonics: int = 3,
    bandwidth: float = 5.0,
) -> np.ndarray:
    """Build soft Gaussian frequency masks for each class.

    Parameters
    ----------
    freq_axis:
        1-D array of STFT frequency-bin centre frequencies (Hz), length F.
    class_freqs:
        Per-class fundamental characteristic frequency (Hz), or ``None`` for
        classes without a defect frequency (e.g. healthy). Length C.
    n_harmonics:
        Number of harmonics (including fundamental) to include in the band.
    bandwidth:
        Std-dev (Hz) of the Gaussian around each harmonic.

    Returns
    -------
    masks: array of shape (C, F) with values in [0, 1]; all-zero rows for
    classes with no characteristic frequency.
    """
    freq_axis = np.asarray(freq_axis, dtype=float)
    C = len(class_freqs)
    F = freq_axis.shape[0]
    masks = np.zeros((C, F), dtype=np.float32)
    for c, f0 in enumerate(class_freqs):
        if f0 is None or f0 <= 0:
            continue
        band = np.zeros(F, dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            centre = f0 * h
            if centre > freq_axis.max():
                break
            band += np.exp(-0.5 * ((freq_axis - centre) / bandwidth) ** 2)
        m = band.max()
        if m > 0:
            band /= m
        masks[c] = band.astype(np.float32)
    return masks


def spectral_consistency_loss(
    stft_energy,
    class_probs,
    freq_masks,
    freq_valid_mask=None,
    eps: float = 1e-8,
):
    """Soft spectral-consistency penalty (diagnosis physics prior).

    For each sample the expected characteristic-frequency mask is the
    probability-weighted combination of per-class masks
    ``m_expected = sum_c p_c * mask_c`` (soft, differentiable -- no argmax).
    The spectral energy profile over frequency, ``s`` (normalised to sum 1),
    should concentrate where ``m_expected`` is high. We reward alignment via a
    normalised negative inner product:

        alignment = sum_f s_f * m_expected_f   in [0, 1]
        L_spec    = mean(1 - alignment)

    Parameters
    ----------
    stft_energy:
        Tensor (B, F) of non-negative spectral energy per frequency bin
        (already reduced over the time axis of the STFT).
    class_probs:
        Tensor (B, C) of softmax class probabilities.
    freq_masks:
        Tensor (C, F) of per-class soft frequency masks (see
        ``build_frequency_masks``).
    freq_valid_mask:
        Optional tensor (C,) marking classes that *have* a characteristic
        frequency (1.0) vs. those that don't (0.0, e.g. healthy). Samples are
        weighted by the probability mass on valid classes so healthy windows
        are not penalised.

    Returns
    -------
    scalar tensor ``L_spec``.
    """
    import torch

    # Normalise energy profile to a distribution over frequency (numerically safe).
    s = stft_energy / (stft_energy.sum(dim=1, keepdim=True) + eps)          # (B, F)
    if freq_masks.dim() == 3:
        # Per-sample (load-aware) masks: (B, C, F). m_expected[b,f] = sum_c p[b,c] mask[b,c,f].
        m_expected = torch.einsum("bc,bcf->bf", class_probs, freq_masks)
    else:
        m_expected = class_probs @ freq_masks                                # (B, F)
    # Normalise the expected mask so alignment is a proper weighted average.
    m_norm = m_expected / (m_expected.sum(dim=1, keepdim=True) + eps)        # (B, F)
    alignment = (s * m_norm).sum(dim=1)                                      # (B,)

    if freq_valid_mask is not None:
        # Weight each sample by how much probability sits on defect classes.
        sample_weight = (class_probs * freq_valid_mask.unsqueeze(0)).sum(dim=1)  # (B,)
        per_sample = (1.0 - alignment) * sample_weight
        denom = sample_weight.sum() + eps
        return per_sample.sum() / denom
    return torch.mean(1.0 - alignment)
