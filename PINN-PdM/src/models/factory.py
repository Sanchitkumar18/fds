"""Model factory: build any model by name from a config dict."""
from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .cnn import CNNBaseline
from .cnn_lstm import CNNLSTMBaseline
from .lstm import LSTMBaseline
from .pinn_pdm import PINNPdM
from .transformer import TransformerBaseline

# Names of models that are physics-capable (expose a prognosis head + rho).
PHYSICS_MODELS = {"pinn_pdm"}
NEURAL_MODELS = {"pinn_pdm", "cnn", "lstm", "gru", "cnn_lstm", "transformer"}


def build_model(name: str, cfg: Dict, n_classes: int, n_features: int = 4) -> nn.Module:
    """Instantiate a model by ``name`` using ``cfg['model']`` hyperparameters."""
    m = cfg["model"]
    phys = cfg.get("physics", {})
    input_hw = int(cfg["preprocess"]["stft_size"])
    conv_filters = tuple(m.get("conv_filters", [16, 32, 64]))
    lstm_hidden = int(m.get("lstm_hidden", 128))
    lstm_layers = int(m.get("lstm_layers", 2))
    embed_dim = int(m.get("embed_dim", 128))
    head_dim = int(m.get("head_dim", 64))
    dropout = float(m.get("dropout", 0.3))
    name = name.lower()

    if name == "pinn_pdm":
        return PINNPdM(n_classes, conv_filters, input_hw, lstm_hidden, lstm_layers,
                       embed_dim, head_dim, n_features, dropout,
                       rho_init=float(phys.get("rho_init", 1.0)))
    if name == "cnn":
        return CNNBaseline(n_classes, conv_filters, input_hw, embed_dim, head_dim,
                           n_features, dropout)
    if name in ("lstm", "gru"):
        return LSTMBaseline(n_classes, input_hw, embed_dim, lstm_hidden, lstm_layers,
                            embed_dim, head_dim, n_features, dropout, rnn=name)
    if name == "cnn_lstm":
        return CNNLSTMBaseline(n_classes, conv_filters, input_hw, lstm_hidden, lstm_layers,
                               embed_dim, head_dim, n_features, dropout)
    if name == "transformer":
        patch = int(m.get("patch", 8))
        depth = int(m.get("depth", 3))
        heads = int(m.get("heads", 4))
        return TransformerBaseline(n_classes, input_hw, patch, embed_dim, depth, heads,
                                   float(m.get("mlp_ratio", 2.0)), embed_dim, head_dim,
                                   n_features, dropout)
    raise ValueError(f"Unknown model name: {name}")
