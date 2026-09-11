"""Plain data-only trainer for baseline models.

Baselines minimise ``L_data`` alone. This is implemented by reusing
:class:`PINNTrainer` with all physics weights forced to zero, guaranteeing the
optimisation loop, schedule and budget are *identical* to PINN-PdM so the
comparison is like-for-like (Case Study #14). The only difference is the
presence/absence of the physics terms.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn

from .pinn_trainer import PINNTrainer


def build_baseline_trainer(model: nn.Module, cfg: Dict, device, task: str,
                           class_names: List[str], freq_axis: Optional[np.ndarray] = None,
                           shaft_hz: float = 30.0, log_dir: Optional[str] = None,
                           model_name: str = "baseline") -> PINNTrainer:
    """Return a :class:`PINNTrainer` configured with zero physics weights."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("physics", {})
    cfg["physics"]["lambda_phys"] = 0.0
    cfg["physics"]["lambda_mono"] = 0.0
    cfg["physics"]["lambda_spec"] = 0.0
    cfg["physics"]["adaptive"] = False
    return PINNTrainer(model, cfg, device, task, class_names, freq_axis, shaft_hz,
                       log_dir, model_name)
