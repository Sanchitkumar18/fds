"""Real-recording loaders with leakage-free, split-before-windowing protocols.

Leakage policy (Review-03 requirement)
--------------------------------------
Overlapping windows are only ever cut *inside* one split. Each recording is
first assigned to a split as a whole (cross-load protocols), or cut into
contiguous, non-overlapping time blocks (in-distribution protocol) separated by
a guard gap of one window length; windowing then runs inside each block. No two
windows that share a single sample can land in different splits, which
:func:`assert_no_overlap` verifies for every split that is built.

Datasets
--------
* CWRU drive-end, 12 kHz, SKF-6205: Normal (97-100), and inner-race, ball and
  outer-race (@6:00) faults of 0.007/0.014/0.021 in at 0-3 hp (40 recordings).
  The normal baseline recordings were acquired at 48 kHz, so they are
  anti-alias filtered and decimated by 4 to 12 kHz (Smith & Randall, 2015).
* MFPT (Society for Machinery Failure Prevention Technology) test rig, NICE
  bearing, 25 Hz shaft: baseline (270 lbs) and inner/outer-race faults under
  variable load. All records are decimated to 48828/4 = 12207 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.io import loadmat

from ..data.feature_extraction import batch_core_features
from ..data.preprocessing import batch_stft_maps
from ..physics.bearing_frequencies import BearingGeometry, SKF_6205, fault_frequency_ratios

# NICE bearing of the MFPT rig: 8 rollers, d = 0.235 in, D = 1.245 in, phi = 0.
MFPT_BEARING = BearingGeometry(n_balls=8, ball_diameter=0.235, pitch_diameter=1.245,
                               contact_angle=0.0, name="MFPT-NICE")

CWRU_CLASSES = ["normal", "inner", "outer", "ball"]
MFPT_CLASSES = ["normal", "inner", "outer"]

# CWRU file id -> (class, fault diameter [in], load [hp]).
CWRU_CATALOG: Dict[int, Tuple[str, float, int]] = {
    97: ("normal", 0.0, 0), 98: ("normal", 0.0, 1), 99: ("normal", 0.0, 2), 100: ("normal", 0.0, 3),
}
for _base, _cls, _dia in [(105, "inner", 0.007), (118, "ball", 0.007), (130, "outer", 0.007),
                          (169, "inner", 0.014), (185, "ball", 0.014), (197, "outer", 0.014),
                          (209, "inner", 0.021), (222, "ball", 0.021), (234, "outer", 0.021)]:
    for _load in range(4):
        CWRU_CATALOG[_base + _load] = (_cls, _dia, _load)

CWRU_NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# MFPT file -> (class, load [lbs]). vload_k follows the MFPT read-me load table.
MFPT_CATALOG: Dict[str, Tuple[str, int]] = {
    "baseline_1": ("normal", 270),
    "OuterRaceFault_1": ("outer", 270),
    "InnerRaceFault_vload_1": ("inner", 0), "InnerRaceFault_vload_2": ("inner", 50),
    "InnerRaceFault_vload_4": ("inner", 150), "InnerRaceFault_vload_7": ("inner", 300),
    "OuterRaceFault_vload_1": ("outer", 25), "OuterRaceFault_vload_2": ("outer", 50),
    "OuterRaceFault_vload_4": ("outer", 150), "OuterRaceFault_vload_7": ("outer", 300),
}


@dataclass
class Recording:
    """One continuous acquisition run."""
    rec_id: str
    signal: np.ndarray            # 1-D float32 at ``fs``
    fs: float
    label: int
    cls: str
    load: float                   # hp (CWRU) or lbs (MFPT)
    shaft_hz: float               # measured where available, else nominal
    severity: float = 0.0
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_cwru(raw_dir: str | Path) -> List[Recording]:
    raw_dir = Path(raw_dir)
    recs: List[Recording] = []
    for fid, (cls, dia, load) in sorted(CWRU_CATALOG.items()):
        path = raw_dir / f"{fid}.mat"
        if not path.exists():
            raise FileNotFoundError(f"missing CWRU recording {path}")
        mat = loadmat(str(path))
        # 99.mat also carries a copy of X098_*; always select the file's own id.
        key = f"X{fid:03d}_DE_time"
        sig = np.asarray(mat[key], dtype=np.float64).ravel()
        rpm_key = f"X{fid:03d}RPM"
        rpm = float(np.asarray(mat[rpm_key]).ravel()[0]) if rpm_key in mat else float(CWRU_NOMINAL_RPM[load])
        if cls == "normal":
            # Baseline acquired at 48 kHz -> decimate to 12 kHz (FIR anti-alias).
            sig = sp_signal.resample_poly(sig, up=1, down=4)
        recs.append(Recording(rec_id=f"CWRU{fid}", signal=sig.astype(np.float32), fs=12000.0,
                              label=CWRU_CLASSES.index(cls), cls=cls, load=float(load),
                              shaft_hz=rpm / 60.0, severity=dia, meta={"rpm": rpm}))
    return recs


def load_mfpt(raw_dir: str | Path, decimate: int = 4) -> List[Recording]:
    raw_dir = Path(raw_dir)
    recs: List[Recording] = []
    target_fs = 48828.0 / decimate
    for name, (cls, load) in MFPT_CATALOG.items():
        path = raw_dir / f"{name}.mat"
        if not path.exists():
            continue
        b = loadmat(str(path), squeeze_me=True, struct_as_record=False)["bearing"]
        sig = np.asarray(b.gs, dtype=np.float64).ravel()
        sr = float(b.sr)
        factor = int(round(sr / target_fs))
        sig = sp_signal.resample_poly(sig, up=1, down=factor)
        recs.append(Recording(rec_id=name, signal=sig.astype(np.float32), fs=target_fs,
                              label=MFPT_CLASSES.index(cls), cls=cls, load=float(load),
                              shaft_hz=float(b.rate)))
    return recs


# ---------------------------------------------------------------------------
# Split-before-windowing
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A contiguous [start, stop) block of one recording assigned to a split."""
    rec: Recording
    start: int
    stop: int
    split: str


def time_block_segments(rec: Recording, fractions: Sequence[Tuple[str, float]],
                        guard: int) -> List[Segment]:
    """Cut one recording into consecutive blocks (e.g. 70/15/15) with guard gaps."""
    n = rec.signal.shape[0]
    usable = n - guard * (len(fractions) - 1)
    segs, pos = [], 0
    for i, (name, frac) in enumerate(fractions):
        length = int(round(usable * frac)) if i < len(fractions) - 1 else n - pos
        segs.append(Segment(rec, pos, min(n, pos + length), name))
        pos = pos + length + guard
    return segs


def window_segment(seg: Segment, window: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (windows (N, L), absolute start indices) cut strictly inside ``seg``."""
    starts = np.arange(seg.start, seg.stop - window + 1, hop)
    if starts.size == 0:
        return np.empty((0, window), np.float32), starts
    idx = starts[:, None] + np.arange(window)[None, :]
    return seg.rec.signal[idx], starts


def assert_no_overlap(index: Dict[str, Dict[str, np.ndarray]], window: int) -> None:
    """Fail loudly if any two windows from different splits share a sample."""
    names = list(index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = index[a]["rec"], index[b]["rec"]
            for rec in np.intersect1d(ra, rb):
                sa = np.sort(index[a]["start"][ra == rec])
                sb = np.sort(index[b]["start"][rb == rec])
                # Two windows overlap iff |start_a - start_b| < window.
                pos = np.searchsorted(sb, sa)
                for k, s in zip(pos, sa):
                    for j in (k - 1, k):
                        if 0 <= j < sb.size and abs(int(sb[j]) - int(s)) < window:
                            raise AssertionError(f"window leakage between {a} and {b} in {rec}")


# ---------------------------------------------------------------------------
# Spectral views used by the physics term
# ---------------------------------------------------------------------------
def spectral_views(windows: np.ndarray, fs: float, f_max: float = 1000.0,
                   env_band: Tuple[float, float] = (1000.0, 5500.0)
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """High-resolution raw and squared-envelope power spectra up to ``f_max``.

    The physics term must see the defect frequencies, which lie at 60-500 Hz
    for CWRU. The 64x64 STFT used as network input has a 94 Hz bin width, so
    it cannot resolve them. Here we use a 2048-point Hann FFT (5.9 Hz bins).

    * raw: |FFT(x)|^2 of the window.
    * envelope: |FFT(|hilbert(bandpass(x))|^2 - mean)|^2 -- the squared-envelope
      spectrum of Randall & Antoni (2011), demodulating the structural
      resonance band so that the impact *repetition rate* appears directly.
    """
    n = windows.shape[1]
    hann = np.hanning(n)[None, :]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    keep = freqs <= f_max
    x = windows - windows.mean(axis=1, keepdims=True)
    raw = np.abs(np.fft.rfft(x * hann, axis=1)) ** 2
    hi = min(env_band[1], 0.45 * fs)
    sos = sp_signal.butter(4, [env_band[0], hi], btype="bandpass", fs=fs, output="sos")
    xb = sp_signal.sosfiltfilt(sos, x, axis=1)
    env = np.abs(sp_signal.hilbert(xb, axis=1)) ** 2
    env = env - env.mean(axis=1, keepdims=True)
    envs = np.abs(np.fft.rfft(env * hann, axis=1)) ** 2
    return raw[:, keep].astype(np.float32), envs[:, keep].astype(np.float32), freqs[keep].astype(np.float32)


# ---------------------------------------------------------------------------
# Building arrays for a protocol
# ---------------------------------------------------------------------------
@dataclass
class SplitArrays:
    x: np.ndarray           # (N, 64, 64) STFT maps (network input)
    feat: np.ndarray        # (N, 4) RMS, kurtosis, crest, skewness
    raw_spec: np.ndarray    # (N, F) raw power spectrum (physics view)
    env_spec: np.ndarray    # (N, F) squared-envelope spectrum (physics view)
    stft_energy: np.ndarray  # (N, 64) time-summed 64x64 map (Review-02 physics view)
    y: np.ndarray
    shaft_hz: np.ndarray
    load: np.ndarray
    rec: np.ndarray
    start: np.ndarray
    windows: np.ndarray     # raw windows, kept for robustness re-featurisation
    fs: float = 12000.0


def featurise(windows: np.ndarray, fs: float, stft_size: int = 64) -> Dict[str, np.ndarray]:
    maps, _ = batch_stft_maps(windows, fs=fs, out_size=stft_size)
    raw, env, freqs = spectral_views(windows, fs)
    return {"x": maps, "feat": batch_core_features(windows).astype(np.float32),
            "raw_spec": raw, "env_spec": env, "stft_energy": maps.sum(axis=2), "freqs": freqs}


def build_arrays(segments: List[Segment], window: int, hop: int,
                 stft_size: int = 64) -> Tuple[Dict[str, SplitArrays], np.ndarray]:
    by_split: Dict[str, List] = {}
    for seg in segments:
        w, starts = window_segment(seg, window, hop)
        if w.shape[0] == 0:
            continue
        by_split.setdefault(seg.split, []).append((seg, w, starts))
    out: Dict[str, SplitArrays] = {}
    freqs = None
    for split, items in by_split.items():
        windows = np.concatenate([w for _, w, _ in items]).astype(np.float32)
        fs = items[0][0].rec.fs
        f = featurise(windows, fs, stft_size)
        freqs = f["freqs"]
        rep = lambda attr: np.concatenate([np.full(w.shape[0], attr(s)) for s, w, _ in items])
        out[split] = SplitArrays(
            x=f["x"], feat=f["feat"], raw_spec=f["raw_spec"], env_spec=f["env_spec"],
            stft_energy=f["stft_energy"],
            y=rep(lambda s: s.rec.label).astype(np.int64),
            shaft_hz=rep(lambda s: s.rec.shaft_hz).astype(np.float32),
            load=rep(lambda s: s.rec.load).astype(np.float32),
            rec=rep(lambda s: s.rec.rec_id).astype(str),
            start=np.concatenate([st for _, _, st in items]).astype(np.int64),
            windows=windows, fs=fs)
    assert_no_overlap({k: {"rec": v.rec, "start": v.start} for k, v in out.items()}, window)
    return out, freqs


def cwru_protocol(recs: List[Recording], regime: str, window: int = 2048,
                  overlap: float = 0.5) -> List[Segment]:
    """Segments for ``in_distribution`` (70/15/15 time blocks of every recording)
    or ``cross_load`` (train/val = 85/15 time blocks of 0-2 hp, test = all of 3 hp)."""
    segs: List[Segment] = []
    for r in recs:
        if regime == "in_distribution":
            segs += time_block_segments(r, [("train", .70), ("val", .15), ("test", .15)], window)
        elif regime == "cross_load":
            if int(r.load) == 3:
                segs.append(Segment(r, 0, r.signal.shape[0], "test"))
            else:
                segs += time_block_segments(r, [("train", .85), ("val", .15)], window)
        else:
            raise ValueError(regime)
    return segs


def mfpt_protocol(recs: List[Recording], window: int = 2048) -> List[Segment]:
    """MFPT held-out-load protocol: faults train on 0-150 lbs, test on 300 lbs.

    The rig provides healthy data at a single load (270 lbs), so the normal
    baseline is time-blocked 70/15/15; the 270-lbs outer-race record is an
    additional unseen-load test record.
    """
    segs: List[Segment] = []
    for r in recs:
        if r.cls == "normal":
            segs += time_block_segments(r, [("train", .70), ("val", .15), ("test", .15)], window)
        elif r.load >= 270:
            segs.append(Segment(r, 0, r.signal.shape[0], "test"))
        else:
            segs += time_block_segments(r, [("train", .85), ("val", .15)], window)
    return segs


def class_ratios(geom: BearingGeometry, class_names: Sequence[str]) -> List[Optional[float]]:
    """Characteristic-frequency ratio (x f_r) per class; ``None`` for healthy."""
    r = fault_frequency_ratios(geom)
    key = {"inner": "inner", "outer": "outer", "ball": "ball"}
    return [r[key[c]] if c in key else None for c in class_names]
