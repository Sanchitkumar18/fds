"""CNN-LSTM baseline (Case Study 2, Section 4).

Architecturally identical to the PINN-PdM backbone (conv stack + LSTM + shared
embedding + two heads) but trained *without* any physics loss. This is the
strongest data-driven reference and the correct opponent for the extrapolation
test: the only difference from PINN-PdM is the physics term.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import CNNLSTMBackbone, DiagnosisHead, PrognosisHead


class CNNLSTMBaseline(nn.Module):
    def __init__(self, n_classes: int = 4, conv_filters=(16, 32, 64), input_hw: int = 64,
                 lstm_hidden: int = 128, lstm_layers: int = 2, embed_dim: int = 128,
                 head_dim: int = 64, n_features: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = CNNLSTMBackbone(conv_filters, input_hw, lstm_hidden, lstm_layers,
                                        embed_dim, n_features, dropout)
        self.diagnosis = DiagnosisHead(embed_dim, head_dim, n_classes)
        self.prognosis = PrognosisHead(embed_dim, head_dim)

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        z = self.backbone(x, feat)
        logits = self.diagnosis(z)
        health = self.prognosis(z, t) if t is not None else None
        return {"logits": logits, "health": health, "embedding": z}
