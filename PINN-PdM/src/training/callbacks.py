"""Training callbacks: early stopping and best-checkpoint tracking."""
from __future__ import annotations

from typing import Optional


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Epochs with no improvement before stopping.
    mode:
        ``"min"`` (lower is better, e.g. loss) or ``"max"`` (accuracy).
    min_delta:
        Minimum change to count as an improvement.
    """

    def __init__(self, patience: int = 8, mode: str = "min", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.num_bad = 0
        self.should_stop = False
        self.improved = False

    def step(self, value: float) -> bool:
        """Register a new metric value; return True if it was the best so far."""
        self.improved = False
        if self.best is None:
            self.best = value
            self.improved = True
            return True
        better = (value < self.best - self.min_delta) if self.mode == "min" \
            else (value > self.best + self.min_delta)
        if better:
            self.best = value
            self.num_bad = 0
            self.improved = True
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.improved
