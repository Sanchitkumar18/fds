#!/usr/bin/env python3
"""Physics residuals on the real CWRU data (no training).

1. Frequency residual per fault recording: r_f = f_obs - f_theory, where f_theory
   is BPFO/BPFI/BSF from the recorded shaft speed and f_obs is the strongest
   squared-envelope peak within +/-5 % of f_theory (full ~10 s recording,
   ~0.1 Hz resolution, parabolic peak interpolation). Line prominence = peak /
   median envelope power in the +/-5 % search band.
2. Spectral residual per window (from CWRU-PG): r_s = 1 - ebar_{b,y}, the shortfall
   of the true fault type's evidence relative to the strongest fault evidence.

    python scripts/physics_residuals.py   # -> outputs/review03/physics_residuals.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import numpy as np
from scipy import signal as sp

from src.physics.bearing_frequencies import all_fault_frequencies
from src.review03.data_real import CWRU_CLASSES, SKF_6205, load_cwru
from src.utils.config import REPO_ROOT

KEY = {"inner": "inner", "outer": "outer", "ball": "ball"}


def envelope_spectrum(x, fs, band=(1000.0, 5500.0)):
    x = x - x.mean()
    sos = sp.butter(4, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(sp.hilbert(sp.sosfiltfilt(sos, x))) ** 2
    env -= env.mean()
    S = np.abs(np.fft.rfft(env * np.hanning(env.size))) ** 2
    return S, np.fft.rfftfreq(env.size, 1.0 / fs)


def peak_near(S, f, f0, rel=0.05):
    sel = np.where((f >= f0 * (1 - rel)) & (f <= f0 * (1 + rel)))[0]
    k = sel[np.argmax(S[sel])]
    # parabolic interpolation on log power for sub-bin accuracy
    a, b, c = np.log(S[k - 1] + 1e-30), np.log(S[k] + 1e-30), np.log(S[k + 1] + 1e-30)
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    f_obs = f[k] + d * (f[1] - f[0])
    prom = S[k] / (np.median(S[sel]) + 1e-30)
    return float(f_obs), float(prom)


def main():
    out = {"recordings": []}
    for r in load_cwru(Path(REPO_ROOT) / "data/raw/cwru"):
        if r.cls == "normal":
            continue
        f_th = all_fault_frequencies(r.shaft_hz, SKF_6205)[KEY[r.cls]]
        S, f = envelope_spectrum(r.signal.astype(np.float64), r.fs)
        f_obs, prom = peak_near(S, f, f_th)
        out["recordings"].append({"rec": r.rec_id, "class": r.cls, "severity_in": r.severity,
                                  "load_hp": int(r.load), "f_theory": f_th, "f_obs": f_obs,
                                  "residual_hz": f_obs - f_th,
                                  "residual_pct": 100 * (f_obs - f_th) / f_th,
                                  "prominence": prom})
    R = out["recordings"]
    clear = lambda x: x["prominence"] >= 10.0      # a distinct spectral line
    def agg(rows):
        rp = np.array([x["residual_pct"] for x in rows])
        rc = np.array([x["residual_pct"] for x in rows if clear(x)])
        return {"n": len(rows), "n_clear_line": int(len(rc)),
                "mean_pct_clear": float(rc.mean()) if rc.size else None,
                "sd_pct_clear": float(rc.std(ddof=1)) if rc.size > 1 else None,
                "mean_abs_pct_all": float(np.abs(rp).mean()),
                "median_prominence": float(np.median([x["prominence"] for x in rows]))}
    out["by_class"] = {c: agg([x for x in R if x["class"] == c]) for c in ("inner", "outer", "ball")}
    out["by_load"] = {str(l): agg([x for x in R if x["load_hp"] == l]) for l in range(4)}
    out["by_severity"] = {str(s): agg([x for x in R if x["severity_in"] == s]) for s in (0.007, 0.014, 0.021)}

    D = np.load(Path(REPO_ROOT) / "data/derived/cwru_pg/cwru_pg.npz")
    y, ev, load = D["label"], D["evidence"], D["load_hp"]
    f = y > 0
    rs = 1.0 - ev[np.arange(y.size), y]
    def sres(m):
        v = rs[m]
        return {"n": int(m.sum()), "mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "frac_zero": float((v < 1e-6).mean())}
    out["spectral_residual"] = {"all_fault": sres(f),
                                **{CWRU_CLASSES[c]: sres(y == c) for c in (1, 2, 3)},
                                **{f"{l}hp": sres(f & (load == l)) for l in range(4)}}
    (Path(REPO_ROOT) / "outputs/review03/physics_residuals.json").write_text(json.dumps(out, indent=1))
    for x in R:
        print(f"{x['rec']:8s} {x['class']:5s} {x['severity_in']:.3f} {x['load_hp']}hp "
              f"th={x['f_theory']:7.2f} obs={x['f_obs']:7.2f} r={x['residual_pct']:+6.2f}% prom={x['prominence']:7.1f}")
    print(json.dumps({k: out[k] for k in ("by_class", "by_load", "by_severity", "spectral_residual")}, indent=1))


if __name__ == "__main__":
    main()
