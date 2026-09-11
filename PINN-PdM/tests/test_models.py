"""Tests for model forward passes, parameter counts and checkpoint I/O."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.common import count_parameters
from src.models.factory import build_model
from src.models.pinn_pdm import PINNPdM
from src.utils.checkpoint import load_checkpoint, save_checkpoint

_CFG = {
    "preprocess": {"stft_size": 64, "n_features": 4},
    "model": {"conv_filters": [16, 32, 64], "lstm_hidden": 128, "lstm_layers": 2,
              "embed_dim": 128, "head_dim": 64, "dropout": 0.3, "patch": 8, "depth": 2, "heads": 4},
    "physics": {"rho_init": 1.0},
}


def _dummy_batch(b=4, s=2):
    x = torch.randn(b, s, 1, 64, 64)
    feat = torch.randn(b, 4)
    t = torch.rand(b, 1, requires_grad=True)
    return x, feat, t


def test_all_models_forward():
    for name in ["cnn", "lstm", "cnn_lstm", "transformer", "pinn_pdm"]:
        model = build_model(name, _CFG, n_classes=4, n_features=4)
        x, feat, t = _dummy_batch()
        out = model(x, feat, t)
        assert out["logits"].shape == (4, 4)
        assert out["health"].shape == (4, 1)
        assert count_parameters(model) > 0


def test_pinn_rho_is_positive():
    model = PINNPdM(n_classes=4, rho_init=1.0)
    assert float(model.rho.item()) > 0


def test_health_is_differentiable_in_time():
    model = build_model("pinn_pdm", _CFG, n_classes=4)
    x, feat, t = _dummy_batch()
    out = model(x, feat, t)
    grad = torch.autograd.grad(out["health"].sum(), t, create_graph=True)[0]
    assert grad.shape == t.shape
    assert torch.isfinite(grad).all()


def test_batch_size_one_edge_case():
    model = build_model("pinn_pdm", _CFG, n_classes=4)
    x, feat, t = _dummy_batch(b=1, s=1)
    out = model(x, feat, t)
    assert out["logits"].shape == (1, 4)


def test_checkpoint_save_load_roundtrip():
    model = build_model("pinn_pdm", _CFG, n_classes=4)
    opt = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        save_checkpoint(path, model, opt, epoch=3, metrics={"val_loss": 0.1})
        model2 = build_model("pinn_pdm", _CFG, n_classes=4)
        meta = load_checkpoint(path, model2)
        assert meta["epoch"] == 3
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
