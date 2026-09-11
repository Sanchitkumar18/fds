"""Collocation-point sampling for evaluating the degradation ODE residual.

In PINN terminology the physics residual is enforced at *collocation points* in
the input domain. For the degradation ODE the domain is time ``t``. We sample
collocation times either on a uniform grid or at the data time-stamps, and
return them as a leaf tensor with ``requires_grad=True`` so that ``dH/dt`` can be
obtained by automatic differentiation.
"""
from __future__ import annotations

from typing import Optional


def uniform_collocation(n_points: int, t_min: float = 0.0, t_max: float = 1.0,
                        device: Optional["object"] = None):
    """Return ``n_points`` uniformly spaced collocation times of shape (n, 1).

    The tensor is a differentiable leaf (``requires_grad=True``).
    """
    import torch

    t = torch.linspace(t_min, t_max, n_points, device=device).unsqueeze(1)
    t.requires_grad_(True)
    return t


def make_differentiable_time(t):
    """Return a differentiable clone of a time tensor of shape (B, 1).

    The clone is a fresh leaf with ``requires_grad=True`` so that
    ``torch.autograd.grad(H.sum(), t)`` yields exact ``dH/dt``.
    """
    t = t.clone().detach()
    if t.dim() == 1:
        t = t.unsqueeze(1)
    t.requires_grad_(True)
    return t
