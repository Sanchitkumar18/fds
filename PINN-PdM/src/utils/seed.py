"""Reproducibility helpers: global seeding and deterministic configuration."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all relevant RNGs for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed applied to ``random``, ``numpy`` and ``torch``.
    deterministic:
        If ``True``, request deterministic cuDNN/algorithm behaviour. This can
        slow training slightly but makes runs repeatable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Best-effort deterministic algorithms; ignored on unsupported ops.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - version dependent
                pass
    except ImportError:  # pragma: no cover - torch always present in this project
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible multi-worker loading."""
    import torch

    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
