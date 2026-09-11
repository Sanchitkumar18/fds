"""Transformer encoder baseline (T4PdM-style, Case Study 2, Section 4).

Each 64x64 STFT window is split into non-overlapping patches; each patch is
linearly embedded into a token. Learnable positional encodings are added, a
[CLS] token aggregates, and a stack of multi-head self-attention encoder layers
processes the token sequence. When a window sequence of length S>1 is given, the
per-window CLS embeddings are mean-pooled. The classification/regression head is
shared with the other models. Capacity is kept comparable to the CNN-LSTM.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import DiagnosisHead, PrognosisHead


class PatchEmbed(nn.Module):
    """Split an (1, H, W) map into patches and linearly embed them."""

    def __init__(self, input_hw: int = 64, patch: int = 8, dim: int = 128) -> None:
        super().__init__()
        assert input_hw % patch == 0, "patch size must divide input side"
        self.n_patches = (input_hw // patch) ** 2
        self.proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W) -> (N, n_patches, dim)
        x = self.proj(x)
        n, d, hp, wp = x.shape
        return x.flatten(2).transpose(1, 2)


class TransformerBaseline(nn.Module):
    def __init__(self, n_classes: int = 4, input_hw: int = 64, patch: int = 8,
                 dim: int = 128, depth: int = 3, heads: int = 4, mlp_ratio: float = 2.0,
                 embed_dim: int = 128, head_dim: int = 64, n_features: int = 4,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(input_hw, patch, dim)
        n_tokens = self.patch_embed.n_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.emb = nn.Sequential(
            nn.Linear(dim + n_features, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.diagnosis = DiagnosisHead(embed_dim, head_dim, n_classes)
        self.prognosis = PrognosisHead(embed_dim, head_dim)

    def _encode_window(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W) -> CLS embedding (N, dim)
        tok = self.patch_embed(x)
        cls = self.cls_token.expand(tok.shape[0], -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_embed
        out = self.encoder(tok)
        return self.norm(out[:, 0])

    def forward(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, s, c, h, w = x.shape
        cls = self._encode_window(x.reshape(b * s, c, h, w)).reshape(b, s, -1)
        pooled = cls.mean(dim=1)
        if feat is None:
            feat = x.new_zeros(b, self.emb[0].in_features - pooled.shape[1])
        z = self.emb(torch.cat([pooled, feat], dim=1))
        logits = self.diagnosis(z)
        health = self.prognosis(z, t) if t is not None else None
        return {"logits": logits, "health": health, "embedding": z}
