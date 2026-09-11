"""Multivariate MLP PINN variant for C-MAPSS (tabular sensor windows).

C-MAPSS provides multivariate sensor time-series rather than vibration signals,
so the STFT/CNN backbone does not apply. This lightweight model flattens a
(window x n_sensors) block, runs an MLP, and predicts a health index H(t) with
the *same* degradation-ODE + monotonicity physics prior (rho co-estimated). A
baseline is obtained by setting the physics weights to zero.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNMlp(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = (128, 128, 64),
                 dropout: float = 0.3, rho_init: float = 0.5) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for hdim in hidden:
            layers += [nn.Linear(prev, hdim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = hdim
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(prev + 1, 64), nn.ReLU(inplace=True),
                                  nn.Linear(64, 1))
        inv = torch.log(torch.expm1(torch.tensor(float(rho_init))).clamp(min=1e-4))
        self.rho_raw = nn.Parameter(inv)

    @property
    def rho(self) -> torch.Tensor:
        return F.softplus(self.rho_raw)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        z = self.body(x.flatten(1))
        if t is None:
            t = x.new_zeros(x.shape[0], 1)
        health = self.head(torch.cat([z, t], dim=1))
        return {"health": health, "embedding": z}
