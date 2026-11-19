"""Spectral-consistency losses: the Review-02 formulation (v1) and its revision (v2).

Both are physics-*guided* spectral priors, not differential-equation residuals:
they compare the network's class probabilities with where the observed window
places its spectral energy relative to closed-form defect frequencies
``f_c = r_c * f_r``. Neither involves a governing ODE/PDE.

v1 (as submitted for Review-02)
-------------------------------
    m_b  = sum_c p_bc m_bc                (mixture of unit-max masks)
    m^_b = m_b / sum_f m_b                (mixture normalised to unit sum)
    a_b  = sum_f s_bf m^_bf
    L    = sum_b w_b (1 - a_b) / sum_b w_b,  w_b = sum_{c in K} p_bc  (K = fault classes)

and ``s`` was the time-summed, log-compressed, min-max-normalised 64x64 STFT
map (94 Hz bins over 0-6 kHz). The audit (``scripts/run_review03.py --exp audit``)
shows three problems:

1. *Resolution.* All CWRU defect harmonics (68-490 Hz) fall in the lowest
   five 94-Hz bins, and log compression flattens ``s``, so ``a_b`` barely
   varies between classes (the prior carries almost no information).
2. *Healthy shortcut.* ``w_b`` is differentiable, so dL/dw_b = (l_b - L)/sum w.
   Windows whose spectrum aligns worse than average are pushed to *lower*
   fault probability, i.e. toward the healthy class, independent of the label.
3. *Mask-mass coupling.* Normalising the mixture makes ``a_b`` a weighted mean
   of per-class alignments with weights ``p_bc * |m_bc|``; classes whose mask
   has more mass inside the band (more harmonics below f_max, finer bin
   coincidence) receive more pull, which is an artefact of discretisation.

v2 (revised)
------------
* ``s`` is a high-resolution (5.9 Hz) raw or squared-envelope power spectrum
  computed from the window itself (see ``data_real.spectral_views``).
* Each class mask is normalised to unit *sum* (equal mass for every class).
* Evidence ``e_bc = sum_f s_bf m^_bcf`` is rescaled per window to
  ``ebar_bc = e_bc / max_{c'} e_bc'`` in [0, 1].
* The network's fault-type distribution ``pt_bc = p_bc / sum_{K} p`` is used,
  so the term is invariant to the healthy logit, and the window weight
  ``w_b = stopgrad(sum_K p_bc)`` cannot be lowered to escape the penalty.

    L_v2 = sum_b w_b (1 - sum_{c in K} pt_bc ebar_bc) / sum_b w_b

Its gradient w.r.t. logits only redistributes probability *among fault types*
in proportion to the physical evidence; ``tests/test_review03.py`` checks
dL/d(logit_healthy) == 0 for v2 and != 0 for v1.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..physics.physics_losses import spectral_consistency_loss as _v1_loss


def build_masks(freqs: np.ndarray, ratios: Sequence[Optional[float]], shaft_hz: torch.Tensor,
                n_harmonics: int = 3, sigma: float = 5.0, norm: str = "sum") -> torch.Tensor:
    """Per-window Gaussian masks (B, C, F) around ``h * r_c * f_r`` for h=1..H.

    ``norm='max'`` reproduces v1 (unit peak), ``norm='sum'`` is used by v2.
    Classes with ratio ``None`` (healthy) get an all-zero mask.
    """
    f = torch.as_tensor(freqs, dtype=torch.float32, device=shaft_hz.device)       # (F,)
    B, C, F = shaft_hz.shape[0], len(ratios), f.shape[0]
    masks = torch.zeros(B, C, F, device=shaft_hz.device)
    for c, r in enumerate(ratios):
        if r is None:
            continue
        f0 = (r * shaft_hz)[:, None]                                               # (B, 1)
        m = torch.zeros(B, F, device=shaft_hz.device)
        for h in range(1, n_harmonics + 1):
            centre = h * f0
            inside = (centre <= f.max()).float()
            m = m + inside * torch.exp(-0.5 * ((f[None, :] - centre) / sigma) ** 2)
        denom = m.amax(dim=1, keepdim=True) if norm == "max" else m.sum(dim=1, keepdim=True)
        masks[:, c] = m / (denom + 1e-12)
    return masks


def valid_mask(ratios: Sequence[Optional[float]], device) -> torch.Tensor:
    return torch.tensor([0.0 if r is None else 1.0 for r in ratios], device=device)


def spectral_loss_v1(energy: torch.Tensor, probs: torch.Tensor, masks: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
    """Exactly the Review-02 term (delegates to the original implementation)."""
    return _v1_loss(energy, probs, masks, valid)


def fault_evidence(spec: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor,
                   eps: float = 1e-12) -> torch.Tensor:
    """Per-window relative evidence ebar (B, C) in [0, 1]; 0 for healthy."""
    s = spec / (spec.sum(dim=1, keepdim=True) + eps)
    e = torch.einsum("bf,bcf->bc", s, masks) * valid[None, :]
    return e / (e.amax(dim=1, keepdim=True) + eps)


def spectral_loss_v2(spec: torch.Tensor, logits: torch.Tensor, masks: torch.Tensor,
                     valid: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    ebar = fault_evidence(spec, masks, valid).detach()
    probs = torch.softmax(logits, dim=1)
    w = (probs * valid[None, :]).sum(dim=1).detach()
    # Fault-type distribution: softmax over fault logits only (healthy-invariant).
    fault_logits = logits.masked_fill(valid[None, :] == 0, float("-inf"))
    pt = torch.softmax(fault_logits, dim=1)
    align = (pt * ebar).sum(dim=1)
    return (w * (1.0 - align)).sum() / (w.sum() + eps)
