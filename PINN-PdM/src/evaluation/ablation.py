"""Ablation-study configuration helpers (Case Study 2, Section 6.5).

Defines the four ablation variants and returns, for each, the physics-weight
overrides to apply to a base config:

    A. Backbone only ............ no physics
    B. + degradation ODE ........ L_phys only
    C. + monotonicity ........... L_phys + L_mono
    D. Full PINN-PdM ............ L_phys + L_mono (+ L_spec) + adaptive weighting
"""
from __future__ import annotations

from typing import Dict, List


def ablation_variants(base_lambda_phys: float = 1.0, base_lambda_mono: float = 1.0,
                      base_lambda_spec: float = 0.0) -> List[Dict]:
    """Return the ordered list of ablation variants with physics overrides."""
    return [
        {"name": "backbone_only", "label": "Backbone only (no physics)",
         "physics": {"lambda_phys": 0.0, "lambda_mono": 0.0, "lambda_spec": 0.0, "adaptive": False}},
        {"name": "ode", "label": "+ degradation ODE (L_phys)",
         "physics": {"lambda_phys": base_lambda_phys, "lambda_mono": 0.0,
                     "lambda_spec": 0.0, "adaptive": False}},
        {"name": "ode_mono", "label": "+ monotonicity (L_mono)",
         "physics": {"lambda_phys": base_lambda_phys, "lambda_mono": base_lambda_mono,
                     "lambda_spec": 0.0, "adaptive": False}},
        {"name": "full", "label": "Full PINN-PdM (+ adaptive w.)",
         "physics": {"lambda_phys": base_lambda_phys, "lambda_mono": base_lambda_mono,
                     "lambda_spec": base_lambda_spec, "adaptive": True}},
    ]


def apply_physics_override(cfg: Dict, override: Dict) -> Dict:
    """Return a deep-copied cfg with the physics block overridden."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("physics", {})
    cfg["physics"].update(override)
    return cfg
