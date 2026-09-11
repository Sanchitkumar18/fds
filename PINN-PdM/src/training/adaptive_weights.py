"""Self-adaptive physics-loss weighting (Case Study 2, Section 2.5 / #13).

Data and physics losses often have very different gradient magnitudes, so a
fixed weighting lets one term dominate. We implement gradient-norm balancing in
the spirit of Wang, Teng & Perdikaris (2021, "Understanding and mitigating
gradient flow pathologies in PINNs") and SoftAdapt / self-adaptive PINNs
(McClenny & Braga-Neto 2023), both cited in the Case Study references.

Design choice (documented, Case Study #13)
------------------------------------------
The *data* loss is the anchor with weight fixed at 1. For every physics term i,
we compute the L2 norm of its gradient w.r.t. the shared backbone parameters and
set the target weight so that term's gradient magnitude matches the data term's:

    hat_lambda_i = ||grad L_data|| / (||grad L_i|| + eps)

The active weight is an exponential moving average of this target:

    lambda_i <- (1 - alpha) * lambda_i + alpha * hat_lambda_i

Weights are clamped to a sane range for stability. This keeps no single term
dominating early training, without any per-condition hand-tuning. The evolution
of every weight is logged and plotted.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch


class AdaptiveWeights:
    def __init__(self, physics_keys: Iterable[str], method: str = "gradnorm",
                 alpha: float = 0.12, init: float = 1.0, w_min: float = 1e-3,
                 w_max: float = 1e2, update_every: int = 1,
                 target_ratio: float = 0.25) -> None:
        self.keys: List[str] = list(physics_keys)
        self.method = method
        self.alpha = alpha
        self.w_min = w_min
        self.w_max = w_max
        self.update_every = max(1, update_every)
        # Physics gradient magnitude is balanced to ``target_ratio`` of the data
        # gradient (not equal to it): physics regularises without dominating, so
        # the network cannot collapse onto the trivial ODE solution H->0.
        self.target_ratio = target_ratio
        self.weights: Dict[str, float] = {k: float(init) for k in self.keys}
        self._step = 0
        self._prev_losses: Dict[str, float] = {}

    def get(self, key: str) -> float:
        return self.weights.get(key, 1.0)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.weights)

    @staticmethod
    def _grad_norm(loss: torch.Tensor, params: List[torch.Tensor]) -> float:
        """L2 norm of d(loss)/d(params), robust to unused parameters."""
        grads = torch.autograd.grad(loss, params, retain_graph=True,
                                    allow_unused=True, create_graph=False)
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.detach().pow(2).sum().item())
        return total ** 0.5

    def update(self, data_loss: torch.Tensor, physics_losses: Dict[str, torch.Tensor],
               shared_params: Iterable[torch.nn.Parameter]) -> None:
        """Update adaptive weights from current loss gradients.

        No-op when ``method == 'none'``. Only fires every ``update_every`` steps.
        Must be called *before* ``loss.backward()`` (it does its own autograd).
        """
        self._step += 1
        if self.method == "none" or (self._step % self.update_every != 0):
            return
        params = [p for p in shared_params if p.requires_grad]
        if not params:
            return

        if self.method == "gradnorm":
            eps = 1e-8
            try:
                g_data = self._grad_norm(data_loss, params)
            except RuntimeError:
                return
            for k, lk in physics_losses.items():
                if lk is None or not lk.requires_grad:
                    continue
                try:
                    g_i = self._grad_norm(lk, params)
                except RuntimeError:
                    continue
                if g_i < eps:
                    continue
                target = self.target_ratio * g_data / (g_i + eps)
                new_w = (1 - self.alpha) * self.weights.get(k, 1.0) + self.alpha * target
                self.weights[k] = float(min(max(new_w, self.w_min), self.w_max))

        elif self.method == "softadapt":
            # Weight terms by their relative rate of *decrease* (softmax of ratios).
            eps = 1e-8
            ratios = {}
            for k, lk in physics_losses.items():
                if lk is None:
                    continue
                cur = float(lk.detach().item())
                prev = self._prev_losses.get(k, cur)
                ratios[k] = cur / (prev + eps)
                self._prev_losses[k] = cur
            if ratios:
                vals = torch.tensor(list(ratios.values()))
                soft = torch.softmax(vals, dim=0)
                for (k, _), s in zip(ratios.items(), soft):
                    target = float(len(ratios) * s.item())
                    new_w = (1 - self.alpha) * self.weights.get(k, 1.0) + self.alpha * target
                    self.weights[k] = float(min(max(new_w, self.w_min), self.w_max))
