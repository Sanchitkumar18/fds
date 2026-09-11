"""PINN-PdM: the proposed physics-informed predictive-maintenance network.

Shared CNN-LSTM backbone (Case Study 2, Section 2.4) with two heads:

* Diagnosis head -> fault-class logits (constrained by the spectral
  characteristic-frequency prior, ``L_spec``).
* Prognosis head -> health index H(t), differentiable in ``t`` (constrained by
  the degradation ODE ``L_phys`` and monotonicity ``L_mono``).

The degradation rate ``rho`` is *co-estimated* as a learnable parameter (an
inverse-problem parameter in the PINN sense, Section 2.3). It is stored as an
unconstrained ``rho_raw`` and mapped to a positive value via softplus so the
ODE ``dH/dt + rho*H = 0`` always has ``rho > 0`` (documented design choice,
Case Study requirement #9, #13).

Baselines reuse this class with the physics weights set to zero (Appendix A note
"Baseline models reuse the same backbone with the physics weights set to zero"),
so the architecture is identical and only the loss differs.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CNNLSTMBackbone, DiagnosisHead, PrognosisHead


class PINNPdM(nn.Module):
    def __init__(self, n_classes: int = 4, conv_filters=(16, 32, 64), input_hw: int = 64,
                 lstm_hidden: int = 128, lstm_layers: int = 2, embed_dim: int = 128,
                 head_dim: int = 64, n_features: int = 4, dropout: float = 0.3,
                 rho_init: float = 1.0) -> None:
        super().__init__()
        self.backbone = CNNLSTMBackbone(conv_filters, input_hw, lstm_hidden, lstm_layers,
                                        embed_dim, n_features, dropout)
        self.diagnosis = DiagnosisHead(embed_dim, head_dim, n_classes)
        self.prognosis = PrognosisHead(embed_dim, head_dim)
        # Co-estimated degradation rate rho > 0 via softplus(rho_raw).
        # Initialise rho_raw so that softplus(rho_raw) ~= rho_init.
        inv = torch.log(torch.expm1(torch.tensor(float(rho_init))).clamp(min=1e-4))
        self.rho_raw = nn.Parameter(inv)
        self.n_classes = n_classes

    @property
    def rho(self) -> torch.Tensor:
        """Positive degradation rate rho = softplus(rho_raw)."""
        return F.softplus(self.rho_raw)

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Return {'logits': (B,C), 'health': (B,1) or None, 'embedding': (B,E)}."""
        z = self.backbone(x, feat)
        logits = self.diagnosis(z)
        health = self.prognosis(z, t) if t is not None else None
        return {"logits": logits, "health": health, "embedding": z}


def build_pinn_pdm(cfg: Dict, n_classes: int, n_features: int = 4) -> PINNPdM:
    """Construct a :class:`PINNPdM` from a config's ``model`` + ``physics`` block."""
    m = cfg["model"]
    phys = cfg.get("physics", {})
    return PINNPdM(
        n_classes=n_classes,
        conv_filters=tuple(m.get("conv_filters", [16, 32, 64])),
        input_hw=int(cfg["preprocess"]["stft_size"]),
        lstm_hidden=int(m.get("lstm_hidden", 128)),
        lstm_layers=int(m.get("lstm_layers", 2)),
        embed_dim=int(m.get("embed_dim", 128)),
        head_dim=int(m.get("head_dim", 64)),
        n_features=n_features,
        dropout=float(m.get("dropout", 0.3)),
        rho_init=float(phys.get("rho_init", 1.0)),
    )
