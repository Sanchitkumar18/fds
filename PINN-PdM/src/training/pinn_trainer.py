"""Composite physics-informed trainer for PINN-PdM and the neural baselines.

Implements the composite objective from Case Study 2 (Section 2.5):

    L = L_data + lambda_phys * L_phys + lambda_mono * L_mono + lambda_spec * L_spec

* ``L_data`` = cross-entropy (diagnosis) and/or MSE on the health index (prognosis)
* ``L_phys`` = degradation-ODE residual, dH/dt via autograd (prognosis)
* ``L_mono`` = monotonicity penalty ReLU(dH/dt)^2 (prognosis)
* ``L_spec`` = spectral characteristic-frequency consistency (diagnosis)

The same trainer runs baselines by setting all physics weights to zero (they then
minimise pure ``L_data``), exactly as described in Appendix A. Self-adaptive
weighting (``AdaptiveWeights``) balances the terms; every weight is logged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics.bearing_frequencies import (SKF_6205, BearingGeometry,
                                           all_fault_frequencies)
from ..physics.physics_losses import (build_frequency_masks, monotonicity_loss,
                                       physics_ode_loss, spectral_consistency_loss)
from ..utils.checkpoint import save_checkpoint
from ..utils.logging import MetricLogger, get_logger
from .adaptive_weights import AdaptiveWeights
from .callbacks import EarlyStopping

logger = get_logger("pinn_trainer")


def build_class_freq_masks(freq_axis: np.ndarray, class_names: List[str],
                           shaft_hz: float, geom: BearingGeometry = SKF_6205,
                           n_harmonics: int = 3, bandwidth: float = 40.0):
    """Build per-class soft frequency masks + a validity mask for the classes.

    Classes without a characteristic defect frequency (e.g. ``normal``) get an
    all-zero mask and validity 0. The bandwidth (Hz) is wide enough to absorb the
    small (~4%) cross-load shift in defect frequency, justifying a single nominal
    mask across CWRU loads (documented approximation).
    """
    freqs = all_fault_frequencies(shaft_hz, geom)
    key_map = {"inner": "inner", "outer": "outer", "ball": "ball", "cage": "cage"}
    class_freqs: List[Optional[float]] = []
    valid: List[float] = []
    for name in class_names:
        k = key_map.get(name.lower())
        if k is None:
            class_freqs.append(None)
            valid.append(0.0)
        else:
            class_freqs.append(freqs[k])
            valid.append(1.0)
    masks = build_frequency_masks(freq_axis, class_freqs, n_harmonics, bandwidth)
    return masks, np.array(valid, dtype=np.float32)


class PINNTrainer:
    def __init__(self, model: nn.Module, cfg: Dict, device, task: str,
                 class_names: List[str], freq_axis: Optional[np.ndarray] = None,
                 shaft_hz: float = 30.0, log_dir: Optional[str] = None,
                 model_name: str = "model",
                 shaft_hz_by_load: Optional[Dict[int, float]] = None) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.task = task
        self.class_names = class_names
        self.model_name = model_name

        phys = cfg.get("physics", {})
        self.lambda_phys = float(phys.get("lambda_phys", 0.0))
        self.lambda_mono = float(phys.get("lambda_mono", 0.0))
        self.lambda_spec = float(phys.get("lambda_spec", 0.0))
        self.use_adaptive = bool(phys.get("adaptive", False))
        self.has_physics = self.lambda_phys > 0 or self.lambda_mono > 0 or self.lambda_spec > 0

        # Spectral masks (only needed if L_spec active and freq_axis available).
        # Load-aware: a defect frequency = (load-invariant ratio) x shaft freq, so
        # we build one mask per operating load using that load's shaft speed. On a
        # held-out load the correct-ratio band is still anchored, giving the physics
        # prior a legitimate cross-load generalisation mechanism.
        self.freq_masks = None            # nominal (C, F) fallback
        self.freq_valid = None
        self._load_index = {}             # load_hp -> row in _load_masks
        self._load_masks = None           # (L, C, F)
        n_harm = int(phys.get("n_harmonics", 3))
        bw = float(phys.get("spec_bandwidth", 40.0))
        if freq_axis is not None and self.lambda_spec > 0:
            masks, valid = build_class_freq_masks(freq_axis, class_names, shaft_hz,
                                                  n_harmonics=n_harm, bandwidth=bw)
            self.freq_masks = torch.tensor(masks, dtype=torch.float32, device=device)
            self.freq_valid = torch.tensor(valid, dtype=torch.float32, device=device)
            if shaft_hz_by_load:
                per_load = []
                for i, (load, hz) in enumerate(sorted(shaft_hz_by_load.items())):
                    m, _ = build_class_freq_masks(freq_axis, class_names, hz,
                                                  n_harmonics=n_harm, bandwidth=bw)
                    per_load.append(m)
                    self._load_index[int(load)] = i
                self._load_masks = torch.tensor(np.stack(per_load), dtype=torch.float32,
                                                device=device)

        # Adaptive weighting over the *active* physics terms.
        active_keys = [k for k, w in [("phys", self.lambda_phys), ("mono", self.lambda_mono),
                                      ("spec", self.lambda_spec)] if w > 0]
        self.adaptive = AdaptiveWeights(
            active_keys, method=phys.get("adaptive_method", "gradnorm") if self.use_adaptive else "none",
            alpha=float(phys.get("adaptive_alpha", 0.12)),
            init=float(phys.get("adaptive_init", 0.1)),
            target_ratio=float(phys.get("adaptive_target_ratio", 0.25))) if active_keys else None
        # Static base weights (used directly if adaptive disabled).
        self.base_w = {"phys": self.lambda_phys, "mono": self.lambda_mono,
                       "spec": self.lambda_spec}

        tr = cfg["training"]
        self.epochs = int(tr["epochs"])
        self.grad_clip = float(tr.get("grad_clip", 0.0))
        opt_name = tr.get("optimizer", "adam").lower()
        params = self.model.parameters()
        if opt_name == "adamw":
            self.opt = torch.optim.AdamW(params, lr=float(tr["lr"]),
                                         weight_decay=float(tr.get("weight_decay", 0.0)))
        else:
            self.opt = torch.optim.Adam(params, lr=float(tr["lr"]),
                                        weight_decay=float(tr.get("weight_decay", 0.0)))
        self.scheduler = None
        if tr.get("scheduler", "none") == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs)
        self.early = EarlyStopping(patience=int(tr.get("early_stopping_patience", 8)), mode="min")
        self.logger = MetricLogger(log_dir) if log_dir else None
        # rho is available only on physics-capable models.
        self.has_rho = hasattr(self.model, "rho")

    # -- shared-parameter accessor for gradient-norm balancing ---------------
    def _shared_params(self):
        if hasattr(self.model, "backbone"):
            return list(self.model.backbone.parameters())
        return list(self.model.parameters())

    # -- loss computation ----------------------------------------------------
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        feat = batch["feat"].to(self.device)
        y = batch["y"].to(self.device)
        health_tgt = batch["health"].to(self.device)

        want_prog = self.task in ("prognosis", "both") or self.lambda_phys > 0 or self.lambda_mono > 0
        t = None
        if want_prog:
            t = batch["t"].to(self.device).clone().requires_grad_(True)

        out = self.model(x, feat, t)
        logits = out["logits"]
        health = out["health"]

        losses: Dict[str, torch.Tensor] = {}
        data_terms = []
        if self.task in ("diagnosis", "both"):
            data_terms.append(F.cross_entropy(logits, y))
        if self.task in ("prognosis", "both") and health is not None:
            valid = health_tgt >= 0
            if valid.any():
                data_terms.append(F.mse_loss(health.squeeze(-1)[valid], health_tgt[valid]))
        losses["data"] = torch.stack(data_terms).sum() if data_terms else logits.sum() * 0.0

        # Physics: degradation ODE + monotonicity (need health, t, rho).
        if (self.lambda_phys > 0 or self.lambda_mono > 0) and health is not None and t is not None:
            rho = self.model.rho if self.has_rho else torch.tensor(1.0, device=self.device)
            if self.lambda_phys > 0:
                losses["phys"] = physics_ode_loss(health, t, rho)
            if self.lambda_mono > 0:
                losses["mono"] = monotonicity_loss(health, t)

        # Spectral consistency (diagnosis prior).
        if self.lambda_spec > 0 and self.freq_masks is not None:
            probs = F.softmax(logits, dim=1)
            last_win = x[:, -1, 0]                       # (B, H, W); rows = freq
            energy = last_win.sum(dim=2)                 # (B, H) energy per freq bin
            masks = self.freq_masks
            if self._load_masks is not None and "load_hp" in batch:
                loads = batch["load_hp"].tolist()
                rows = [self._load_index.get(int(l), None) for l in loads]
                if all(r is not None for r in rows):
                    idx = torch.tensor(rows, device=self.device, dtype=torch.long)
                    masks = self._load_masks[idx]        # (B, C, F) per-sample
            losses["spec"] = spectral_consistency_loss(
                energy, probs, masks, self.freq_valid)
        return losses

    def _weighted_total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = losses["data"]
        for k in ("phys", "mono", "spec"):
            if k in losses:
                w = self.adaptive.get(k) if (self.adaptive and self.use_adaptive) else self.base_w[k]
                total = total + w * losses[k]
        return total

    # -- epoch loops ---------------------------------------------------------
    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        agg: Dict[str, float] = {}
        n = 0
        for batch in loader:
            self.opt.zero_grad()
            losses = self.compute_losses(batch)
            # Adaptive weight update (uses its own autograd; before backward).
            if self.adaptive and self.use_adaptive and self.has_physics:
                phys_losses = {k: losses[k] for k in ("phys", "mono", "spec") if k in losses}
                if phys_losses:
                    self.adaptive.update(losses["data"], phys_losses, self._shared_params())
            total = self._weighted_total(losses)
            total.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            bs = batch["x"].shape[0]
            n += bs
            agg["total"] = agg.get("total", 0.0) + float(total.item()) * bs
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.item()) * bs
        return {k: v / max(1, n) for k, v in agg.items()}

    @torch.no_grad()
    def _eval_data_loss(self, loader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            # Validation data loss only (no physics needed for model selection).
            x = batch["x"].to(self.device)
            feat = batch["feat"].to(self.device)
            y = batch["y"].to(self.device)
            health_tgt = batch["health"].to(self.device)
            t = batch["t"].to(self.device) if self.task in ("prognosis", "both") else None
            out = self.model(x, feat, t)
            terms = []
            if self.task in ("diagnosis", "both"):
                terms.append(F.cross_entropy(out["logits"], y))
            if self.task in ("prognosis", "both") and out["health"] is not None:
                valid = health_tgt >= 0
                if valid.any():
                    terms.append(F.mse_loss(out["health"].squeeze(-1)[valid], health_tgt[valid]))
            loss = torch.stack(terms).sum() if terms else torch.tensor(0.0)
            bs = x.shape[0]
            total += float(loss.item()) * bs
            n += bs
        return total / max(1, n)

    def fit(self, train_loader, val_loader, ckpt_path: Optional[str] = None) -> Dict:
        history = []
        best_val = float("inf")
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_loss = self._eval_data_loss(val_loader)
            if self.scheduler:
                self.scheduler.step()
            row = {"epoch": epoch, "val_loss": val_loss, **{f"train_{k}": v for k, v in train_metrics.items()}}
            if self.adaptive and self.use_adaptive:
                for k, w in self.adaptive.as_dict().items():
                    row[f"lambda_{k}"] = w
            if self.has_rho:
                row["rho"] = float(self.model.rho.item())
            history.append(row)
            if self.logger:
                self.logger.log(epoch, {k: v for k, v in row.items() if k != "epoch"})
            improved = self.early.step(val_loss)
            msg = (f"[{self.model_name}] epoch {epoch}/{self.epochs} "
                   f"val_loss={val_loss:.4f} train_total={train_metrics.get('total', 0):.4f}")
            if self.has_rho:
                msg += f" rho={float(self.model.rho.item()):.3f}"
            logger.info(msg)
            if improved and ckpt_path:
                best_val = val_loss
                save_checkpoint(ckpt_path, self.model, self.opt, epoch,
                                metrics={"val_loss": val_loss}, extra={"model_name": self.model_name})
            if self.early.should_stop:
                logger.info("[%s] early stopping at epoch %d", self.model_name, epoch)
                break
        if self.logger:
            self.logger.close()
        return {"history": history, "best_val": best_val,
                "final_weights": self.adaptive.as_dict() if self.adaptive else {}}
