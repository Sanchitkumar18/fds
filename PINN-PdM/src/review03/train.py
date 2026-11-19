"""Training / evaluation loop for the Review-03 experiments.

One function, :func:`run_neural`, trains any neural model with or without the
spectral physics term, under fixed or adaptive weighting, and returns measured
metrics. All variants share inputs, splits, optimiser, schedule, epochs,
early-stopping rule and class weighting, so that differences between rows can
be attributed to the physics configuration alone.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..models.cnn import CNNBaseline
from ..models.cnn_lstm import CNNLSTMBaseline
from ..models.common import count_parameters
from ..models.transformer import TransformerBaseline
from ..training.adaptive_weights import AdaptiveWeights
from .data_real import SplitArrays
from .spectral import build_masks, spectral_loss_v1, spectral_loss_v2, valid_mask


@dataclass
class PhysicsCfg:
    loss: Optional[str] = None      # None | "v1" | "v2"
    view: str = "env"               # "env" | "raw" | "stft" (v1 uses the 64-bin STFT map)
    lam: float = 0.5                # fixed weight, or initial weight when adaptive
    adaptive: bool = False
    target_ratio: float = 0.25      # adaptive: ||grad L_spec|| = ratio * ||grad L_data||
    alpha: float = 0.12             # adaptive: EMA rate
    sigma: float = 5.0              # Gaussian band std (Hz)
    harmonics: int = 3
    # Transductive variant: also apply L_spec (no labels) to windows of the
    # target condition, the analogue of PINN collocation points. Uses the
    # *inputs* of splits["unlabeled"] only -- never their labels.
    collocation: bool = False


@dataclass
class TrainCfg:
    model: str = "cnn_lstm"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 8
    grad_clip: float = 5.0
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)


DEVICE: Optional[str] = None   # override (e.g. "cpu", "mps") set by the runner


def pick_device() -> torch.device:
    if DEVICE:
        return torch.device(DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_model(name: str, n_classes: int) -> nn.Module:
    if name == "cnn":
        return CNNBaseline(n_classes=n_classes)
    if name == "cnn_lstm":
        return CNNLSTMBaseline(n_classes=n_classes)
    if name == "transformer":
        return TransformerBaseline(n_classes=n_classes)
    raise ValueError(name)


class _Tensors:
    """Device-resident tensors for one split (data sets are small)."""

    def __init__(self, a: SplitArrays, feat_mu: np.ndarray, feat_sd: np.ndarray,
                 device: torch.device) -> None:
        t = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=device)
        self.x = t(a.x[:, None, None])                       # (N, S=1, 1, H, W)
        self.feat = t((a.feat - feat_mu) / feat_sd)
        self.y = t(a.y, torch.long)
        self.shaft = t(a.shaft_hz)
        self.spec = {"env": t(a.env_spec), "raw": t(a.raw_spec), "stft": t(a.stft_energy)}
        self.n = a.y.shape[0]


def _metrics(y: np.ndarray, pred: np.ndarray, n_classes: int) -> Dict:
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    recall = (cm.diagonal() / np.maximum(cm.sum(axis=1), 1)).tolist()
    return {"acc": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)),
                                 zero_division=0)),
            "recall": recall, "cm": cm.tolist()}


@torch.no_grad()
def predict(model: nn.Module, T: _Tensors, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, T.n, bs):
        out.append(model(T.x[i:i + bs], T.feat[i:i + bs])["logits"].argmax(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _val_loss(model, T: _Tensors, cw: torch.Tensor, bs: int = 256) -> float:
    model.eval()
    tot = 0.0
    for i in range(0, T.n, bs):
        lg = model(T.x[i:i + bs], T.feat[i:i + bs])["logits"]
        tot += float(F.cross_entropy(lg, T.y[i:i + bs], weight=cw, reduction="sum"))
    return tot / T.n


def run_neural(splits: Dict[str, SplitArrays], freqs_hi: np.ndarray, ratios: Sequence[Optional[float]],
               n_classes: int, cfg: TrainCfg, seed: int,
               extra_tests: Optional[Dict[str, SplitArrays]] = None,
               keep_model: bool = False) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()
    tr, va = splits["train"], splits["val"]
    mu, sd = tr.feat.mean(0), tr.feat.std(0) + 1e-8          # train-only scaling
    Ttr, Tva = _Tensors(tr, mu, sd, device), _Tensors(va, mu, sd, device)
    counts = np.bincount(tr.y, minlength=n_classes).astype(np.float64)
    cw = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32,
                      device=device)

    model = make_model(cfg.model, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    P = cfg.physics
    valid = valid_mask(ratios, device)
    stft_freqs = _STFT_FREQS(tr)
    all_masks = None
    if P.loss == "v1":
        all_masks = build_masks(stft_freqs, ratios, Ttr.shaft, P.harmonics, P.sigma, "max")
    elif P.loss == "v2":
        all_masks = build_masks(freqs_hi, ratios, Ttr.shaft, P.harmonics, P.sigma, "sum")
    Tun, un_masks = None, None
    if P.collocation:
        un = splits["unlabeled"]
        Tun = _Tensors(un, mu, sd, device)
        un_masks = build_masks(freqs_hi, ratios, Tun.shaft, P.harmonics, P.sigma, "sum")
    adaptive = AdaptiveWeights(["spec"], alpha=P.alpha, init=P.lam,
                               target_ratio=P.target_ratio) if (P.loss and P.adaptive) else None
    shared = list(model.backbone.parameters()) if hasattr(model, "backbone") else list(model.parameters())

    best, best_state, bad, hist = float("inf"), None, 0, []
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(Ttr.n, generator=g).to(device)
        agg = {"data": 0.0, "spec": 0.0, "lam": 0.0}
        nb = 0
        for i in range(0, Ttr.n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if idx.numel() < 2:
                continue
            out = model(Ttr.x[idx], Ttr.feat[idx])
            logits = out["logits"]
            l_data = F.cross_entropy(logits, Ttr.y[idx], weight=cw)
            total = l_data
            if P.loss:
                masks = all_masks[idx]
                if P.loss == "v1":
                    l_spec = spectral_loss_v1(Ttr.spec["stft"][idx], torch.softmax(logits, 1),
                                              masks, valid)
                else:
                    l_spec = spectral_loss_v2(Ttr.spec[P.view][idx], logits, masks, valid)
                if Tun is not None:
                    j = torch.randint(0, Tun.n, (idx.numel(),), generator=g).to(device)
                    lg_u = model(Tun.x[j], Tun.feat[j])["logits"]
                    l_spec = l_spec + spectral_loss_v2(Tun.spec[P.view][j], lg_u, un_masks[j], valid)
                if adaptive is not None:
                    adaptive.update(l_data, {"spec": l_spec}, shared)
                    lam = adaptive.get("spec")
                else:
                    lam = P.lam
                total = total + lam * l_spec
                agg["spec"] += float(l_spec.detach())
                agg["lam"] += float(lam)
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            agg["data"] += float(l_data.detach())
            nb += 1
        sched.step()
        vl = _val_loss(model, Tva, cw)
        hist.append({"epoch": ep, "val_loss": vl, **{k: v / max(nb, 1) for k, v in agg.items()}})
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)

    res = {"seed": seed, "model": cfg.model, "physics": asdict(P), "epochs_run": len(hist),
           "train_time_s": time.time() - t0, "params": count_parameters(model), "history": hist}
    for name, arr in {"val": va, **({"test": splits["test"]} if "test" in splits else {}),
                      **(extra_tests or {})}.items():
        T = _Tensors(arr, mu, sd, device)
        res[name] = _metrics(arr.y, predict(model, T), n_classes)
    if keep_model:
        res["_model"] = model
        res["_scaler"] = (mu, sd)
    return res


def _STFT_FREQS(a: SplitArrays) -> np.ndarray:
    """Row frequencies of the 64x64 STFT map (0 .. fs/2), as used by v1."""
    return np.linspace(0.0, a.fs / 2.0, a.stft_energy.shape[1]).astype(np.float32)


def evaluate_model(model: nn.Module, scaler, arr: SplitArrays, n_classes: int) -> Dict:
    T = _Tensors(arr, scaler[0], scaler[1], next(model.parameters()).device)
    return _metrics(arr.y, predict(model, T), n_classes)
