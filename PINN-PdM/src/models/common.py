"""Shared model building blocks for PINN-PdM and the neural baselines.

The backbone follows Case Study 2 (Section 2.4): a 3-block Conv2D stack
(16/32/64 filters, 3x3 kernels, BatchNorm+ReLU+MaxPool) applied to each 64x64
STFT window, a 2-layer LSTM (128 hidden) over the sequence of window embeddings,
and a shared Dense(128)+ReLU+Dropout embedding. The 4 statistical features are
fused into the shared embedding. Two heads (diagnosis, prognosis) sit on top.

Keeping the backbone shared across models makes the comparison strictly
like-for-like: every neural model sees identical inputs (Case Study #14).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvStack(nn.Module):
    """3x [Conv2D -> BatchNorm -> ReLU -> MaxPool] spatial-spectral extractor."""

    def __init__(self, filters: List[int] = (16, 32, 64), in_ch: int = 1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_ch
        for f in filters:
            layers += [
                nn.Conv2d(prev, f, kernel_size=3, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = f
        self.net = nn.Sequential(*layers)
        self.out_channels = filters[-1]
        self.n_pools = len(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def output_dim(self, input_hw: int) -> int:
        """Flattened conv output dim for a square input of side ``input_hw``."""
        side = input_hw
        for _ in range(self.n_pools):
            side = side // 2
        return self.out_channels * side * side


class CNNLSTMBackbone(nn.Module):
    """Conv-LSTM backbone shared by PINN-PdM, CNN-LSTM and (optionally) others.

    forward(x, feat) -> shared embedding z of shape (B, embed_dim).
    ``x``: (B, S, 1, H, W) window sequence; ``feat``: (B, n_features).
    """

    def __init__(self, conv_filters: List[int] = (16, 32, 64), input_hw: int = 64,
                 lstm_hidden: int = 128, lstm_layers: int = 2, embed_dim: int = 128,
                 n_features: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv = ConvStack(conv_filters)
        self.conv_dim = self.conv.output_dim(input_hw)
        self.lstm = nn.LSTM(self.conv_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True)
        # Fuse the 4 statistical features into the shared embedding.
        self.emb = nn.Sequential(
            nn.Linear(lstm_hidden + n_features, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, c, h, w = x.shape
        f = self.conv(x.reshape(b * s, c, h, w)).flatten(1).reshape(b, s, -1)
        seq_out, _ = self.lstm(f)
        last = seq_out[:, -1]                      # (B, lstm_hidden)
        if feat is None:
            feat = x.new_zeros(b, self.emb[0].in_features - last.shape[1])
        z = self.emb(torch.cat([last, feat], dim=1))
        return z


class DiagnosisHead(nn.Module):
    """Dense(head_dim) -> ReLU -> Dense(n_classes) classification head."""

    def __init__(self, embed_dim: int = 128, head_dim: int = 64, n_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, head_dim), nn.ReLU(inplace=True),
            nn.Linear(head_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PrognosisHead(nn.Module):
    """Dense(head_dim) -> Dense(1) health head; differentiable in time ``t``.

    The time input is concatenated to the shared embedding so H = f(z, t) and
    dH/dt can be taken by autograd (Case Study 2, Appendix A).
    """

    def __init__(self, embed_dim: int = 128, head_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, head_dim), nn.ReLU(inplace=True),
            nn.Linear(head_dim, 1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, t], dim=1))
