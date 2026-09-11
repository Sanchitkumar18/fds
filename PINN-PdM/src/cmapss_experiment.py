"""Optional C-MAPSS FD001 experiment (stretch, Case Study 2 §6.6).

Self-contained tabular RUL pipeline: multivariate sensor windows -> normalised
RUL (treated as a health index H in [0,1], decreasing over the engine's life).
Trains a ``PINNMlp`` with the degradation-ODE + monotonicity prior and an
identically-sized unconstrained baseline (physics off), plus a Random Forest
reference, and reports RMSE/MAE/R2 on real RUL units. Kept separate from the
vibration pipeline because C-MAPSS is not STFT-based. Runs on a synthetic
multivariate stand-in when the raw files are absent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .data.cmapps_loader import load_cmapss
from .data.splits import by_unit_split
from .evaluation.rul_metrics import rul_metrics
from .models.pinn_mlp import PINNMlp
from .physics.physics_losses import monotonicity_loss, physics_ode_loss
from .utils.config import get_device, resolve_path
from .utils.logging import get_logger
from .utils.seed import set_seed

logger = get_logger("cmapss")


def _make_tensors(data: Dict[str, np.ndarray], idx: np.ndarray, rul_clip: float, device):
    X = torch.tensor(data["X"][idx], dtype=torch.float32, device=device)
    # Normalise RUL into [0,1] as a health index (1=healthy, 0=failed).
    H = torch.tensor(np.clip(data["rul"][idx] / rul_clip, 0, 1), dtype=torch.float32, device=device)
    t = torch.tensor(data["t"][idx], dtype=torch.float32, device=device).unsqueeze(1)
    return X, H, t


def run_cmapss_experiment(cfg: Dict, epochs: int | None = None) -> Dict:
    device = get_device(cfg.get("device", "auto"))
    set_seed(int(cfg.get("seed", 42)))
    cm = cfg["cmapss"]
    rul_clip = float(cm.get("rul_clip", 125))
    tr_cfg = cfg["training"]
    epochs = epochs or int(tr_cfg["epochs"])
    phys = cfg.get("physics", {})

    data = load_cmapss(cfg)
    synthetic = bool(int(data.get("_synthetic", [1])[0]))
    tr, te = by_unit_split(data["unit_id"], test_frac=0.30, seed=int(cfg.get("seed", 42)))
    Xtr, Htr, ttr = _make_tensors(data, tr, rul_clip, device)
    Xte, Hte, tte = _make_tensors(data, te, rul_clip, device)
    in_dim = Xtr.shape[1] * Xtr.shape[2]

    def train(use_physics: bool) -> Dict:
        model = PINNMlp(in_dim, hidden=tuple(cfg["model"].get("hidden", [128, 128, 64])),
                        dropout=float(cfg["model"].get("dropout", 0.3)),
                        rho_init=float(phys.get("rho_init", 0.5))).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["lr"]))
        bs = int(tr_cfg["batch_size"])
        n = Xtr.shape[0]
        lam_phys = float(phys.get("lambda_phys", 0.5)) if use_physics else 0.0
        lam_mono = float(phys.get("lambda_mono", 0.5)) if use_physics else 0.0
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                xb, hb = Xtr[j], Htr[j]
                tb = ttr[j].clone().requires_grad_(True)
                opt.zero_grad()
                out = model(xb, tb)
                loss = F.mse_loss(out["health"].squeeze(-1), hb)
                if use_physics:
                    loss = loss + lam_phys * physics_ode_loss(out["health"], tb, model.rho)
                    loss = loss + lam_mono * monotonicity_loss(out["health"], tb)
                loss.backward()
                opt.step()
        train_time = time.time() - t0
        model.eval()
        with torch.no_grad():
            pred_h = model(Xte, tte)["health"].squeeze(-1).cpu().numpy()
        pred_rul = np.clip(pred_h, 0, 1) * rul_clip
        true_rul = np.clip(data["rul"][te], 0, rul_clip)
        m = rul_metrics(true_rul, pred_rul)
        m["train_time_s"] = train_time
        m["final_rho"] = float(model.rho.item())
        return m

    logger.info("[cmapss] training unconstrained baseline ...")
    base = train(use_physics=False)
    logger.info("[cmapss] training PINN-MLP (ODE + monotonicity) ...")
    pinn = train(use_physics=True)

    # Random Forest reference on flattened windows.
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=int(cfg.get("seed", 42)), n_jobs=-1)
    rf.fit(data["X"][tr].reshape(len(tr), -1), np.clip(data["rul"][tr], 0, rul_clip))
    rf_pred = rf.predict(data["X"][te].reshape(len(te), -1))
    rf_m = rul_metrics(np.clip(data["rul"][te], 0, rul_clip), rf_pred)

    result = {"experiment": "cmapss_FD001", "synthetic_fallback": synthetic,
              "measured": {"random_forest": rf_m, "unconstrained_mlp": base, "pinn_mlp": pinn},
              "case_study_projected_6_6": {
                  "note": "Projected from Case Study 2 §6.6 -- NOT measured here.",
                  "random_forest_rmse": 17.5, "pinn_pdm_rmse": 13.6}}
    out = resolve_path(cfg["paths"]["output_dir"]) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cmapss.json", "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("[cmapss] RF RMSE=%.2f  baseline RMSE=%.2f  PINN RMSE=%.2f",
                rf_m["rmse"], base["rmse"], pinn["rmse"])
    return result
