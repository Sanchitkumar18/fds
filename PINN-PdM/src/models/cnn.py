"""CNN baseline (Case Study 2, Section 4): 2D CNN over the STFT map.

A pure convolutional model with no recurrent stack. It applies the shared
ConvStack to each window, mean-pools the conv embedding across the window
sequence, fuses the statistical features, and attaches the same diagnosis /
prognosis heads. It consumes identical inputs to PINN-PdM (like-for-like).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import ConvStack, DiagnosisHead, PrognosisHead


class CNNBaseline(nn.Module):
    def __init__(self, n_classes: int = 4, conv_filters=(16, 32, 64), input_hw: int = 64,
                 embed_dim: int = 128, head_dim: int = 64, n_features: int = 4,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.conv = ConvStack(conv_filters)
        conv_dim = self.conv.output_dim(input_hw)
        self.emb = nn.Sequential(
            nn.Linear(conv_dim + n_features, embed_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout))
        self.diagnosis = DiagnosisHead(embed_dim, head_dim, n_classes)
        self.prognosis = PrognosisHead(embed_dim, head_dim)

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, s, c, h, w = x.shape
        f = self.conv(x.reshape(b * s, c, h, w)).flatten(1).reshape(b, s, -1)
        pooled = f.mean(dim=1)                         # aggregate over sequence
        if feat is None:
            feat = x.new_zeros(b, self.emb[0].in_features - pooled.shape[1])
        z = self.emb(torch.cat([pooled, feat], dim=1))
        logits = self.diagnosis(z)
        health = self.prognosis(z, t) if t is not None else None
        return {"logits": logits, "health": health, "embedding": z}
