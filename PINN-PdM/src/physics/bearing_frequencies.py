"""Rolling-element bearing characteristic defect frequencies.

Implements the closed-form defect frequencies from Case Study 2 (Section 2.2).
For a shaft rotating at frequency ``fr`` (Hz), a bearing with ``n`` rolling
elements, ball diameter ``d``, pitch diameter ``D`` and contact angle ``phi``
(radians), the four characteristic defect frequencies are:

    BPFO = (n/2) * fr * (1 - (d/D) cos(phi))     # outer race
    BPFI = (n/2) * fr * (1 + (d/D) cos(phi))     # inner race
    BSF  = (D/2d) * fr * (1 - ((d/D) cos(phi))^2) # ball / rolling element
    FTF  = (1/2)  * fr * (1 - (d/D) cos(phi))     # cage (fundamental train)

These are used both for physics-informed spectral loss (``physics_losses``) and
for verification against the synthetic testbed (SKF-6205 geometry).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BearingGeometry:
    """Geometry of a rolling-element bearing.

    Attributes
    ----------
    n_balls:
        Number of rolling elements.
    ball_diameter:
        Ball / rolling-element diameter ``d`` (consistent length units).
    pitch_diameter:
        Pitch diameter ``D`` (same units as ``ball_diameter``).
    contact_angle:
        Contact angle ``phi`` in radians (0 for deep-groove ball bearings).
    name:
        Optional identifier (e.g. ``"SKF-6205"``).
    """

    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.n_balls <= 0:
            raise ValueError("n_balls must be positive")
        if self.ball_diameter <= 0 or self.pitch_diameter <= 0:
            raise ValueError("diameters must be positive")
        if self.ball_diameter >= self.pitch_diameter:
            raise ValueError("ball_diameter must be smaller than pitch_diameter")


# Standard SKF-6205 drive-end bearing used by CWRU (and the synthetic testbed).
# Geometry (inches): 9 balls, d=0.3126, D=1.537, phi=0. These yield the
# textbook ratios BPFO=3.585 fr, BPFI=5.415 fr, BSF=2.357 fr, FTF=0.398 fr.
SKF_6205 = BearingGeometry(
    n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537, contact_angle=0.0, name="SKF-6205"
)

# SKF-6203 fan-end bearing (CWRU fan-end records).
SKF_6203 = BearingGeometry(
    n_balls=9, ball_diameter=0.2656, pitch_diameter=1.122, contact_angle=0.0, name="SKF-6203"
)


def bpfo(fr: float, geom: BearingGeometry) -> float:
    """Ball-Pass Frequency Outer race (Hz)."""
    c = (geom.ball_diameter / geom.pitch_diameter) * math.cos(geom.contact_angle)
    return 0.5 * geom.n_balls * fr * (1.0 - c)


def bpfi(fr: float, geom: BearingGeometry) -> float:
    """Ball-Pass Frequency Inner race (Hz)."""
    c = (geom.ball_diameter / geom.pitch_diameter) * math.cos(geom.contact_angle)
    return 0.5 * geom.n_balls * fr * (1.0 + c)


def bsf(fr: float, geom: BearingGeometry) -> float:
    """Ball Spin Frequency (Hz)."""
    c = (geom.ball_diameter / geom.pitch_diameter) * math.cos(geom.contact_angle)
    return (geom.pitch_diameter / (2.0 * geom.ball_diameter)) * fr * (1.0 - c ** 2)


def ftf(fr: float, geom: BearingGeometry) -> float:
    """Fundamental Train Frequency / cage frequency (Hz)."""
    c = (geom.ball_diameter / geom.pitch_diameter) * math.cos(geom.contact_angle)
    return 0.5 * fr * (1.0 - c)


def all_fault_frequencies(fr: float, geom: BearingGeometry) -> Dict[str, float]:
    """Return all four characteristic frequencies keyed by fault type.

    Keys: ``"outer"`` (BPFO), ``"inner"`` (BPFI), ``"ball"`` (BSF),
    ``"cage"`` (FTF).
    """
    return {
        "outer": bpfo(fr, geom),
        "inner": bpfi(fr, geom),
        "ball": bsf(fr, geom),
        "cage": ftf(fr, geom),
    }


def fault_frequency_ratios(geom: BearingGeometry) -> Dict[str, float]:
    """Return characteristic frequencies as multiples of shaft speed (fr=1)."""
    return all_fault_frequencies(1.0, geom)


def harmonics(freq: float, n_harmonics: int = 3) -> List[float]:
    """Return ``[freq, 2*freq, ..., n_harmonics*freq]``."""
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    return [freq * (k + 1) for k in range(n_harmonics)]


def rpm_to_hz(rpm: float) -> float:
    """Convert shaft speed in RPM to rotational frequency in Hz."""
    return rpm / 60.0


# Approximate CWRU shaft speeds (RPM) by motor load in horsepower.
CWRU_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def cwru_shaft_hz(load_hp: int) -> float:
    """Return the CWRU drive-end shaft frequency (Hz) for a motor load (HP)."""
    if load_hp not in CWRU_RPM_BY_LOAD:
        raise KeyError(f"Unknown CWRU load {load_hp}; expected one of {list(CWRU_RPM_BY_LOAD)}")
    return rpm_to_hz(CWRU_RPM_BY_LOAD[load_hp])
