"""LSTM / GRU baseline (Case Study 2, Section 4).

A recurrent model with no convolutional stack: each STFT window is flattened and
linearly projected to a token, then a 2-layer LSTM (or GRU) runs over the window
sequence. Sequential degradation modelling for the RUL task; also usable for
diagnosis. Same heads and inputs as the other models.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import DiagnosisHead, PrognosisHead


class LSTMBaseline(nn.Module):
    def __init__(self, n_classes: int = 4, input_hw: int = 64, proj_dim: int = 128,
                 hidden: int = 128, layers: int = 2, embed_dim: int = 128,
                 head_dim: int = 64, n_features: int = 4, dropout: float = 0.3,
                 rnn: str = "lstm") -> None:
        super().__init__()
        self.proj = nn.Linear(input_hw * input_hw, proj_dim)
        rnn_cls = nn.LSTM if rnn.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(proj_dim, hidden, num_layers=layers, batch_first=True)
        self.emb = nn.Sequential(
            nn.Linear(hidden + n_features, embed_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout))
        self.diagnosis = DiagnosisHead(embed_dim, head_dim, n_classes)
        self.prognosis = PrognosisHead(embed_dim, head_dim)

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, s, c, h, w = x.shape
        tokens = self.proj(x.reshape(b, s, h * w))
        seq_out, _ = self.rnn(tokens)
        last = seq_out[:, -1]
        if feat is None:
            feat = x.new_zeros(b, self.emb[0].in_features - last.shape[1])
        z = self.emb(torch.cat([last, feat], dim=1))
        logits = self.diagnosis(z)
        health = self.prognosis(z, t) if t is not None else None
        return {"logits": logits, "health": health, "embedding": z}
