#!/usr/bin/env python3
"""Review-03 experiment suite on the ORIGINAL CWRU and MFPT recordings.

Usage
-----
    python scripts/run_review03.py --exp audit        # L_spec audit (no training)
    python scripts/run_review03.py --exp xload        # P1: 0/1/2 -> 3 hp, ablations, weighting, robustness
    python scripts/run_review03.py --exp indist       # in-distribution (time-blocked) split
    python scripts/run_review03.py --exp single       # harder 0 hp -> 3 hp single-source protocol
    python scripts/run_review03.py --exp sens         # P2: bandwidth / harmonic sensitivity
    python scripts/run_review03.py --exp mfpt         # P2: second bearing dataset
    python scripts/run_review03.py --exp all
    python scripts/run_review03.py --exp summary      # aggregate -> outputs/review03/summary.json + tables

Every run is cached as ``outputs/review03/runs/<exp>/<config>__seed<k>.json``,
so the suite is resumable and each table cell traces back to its run files.
Data: ``data/raw/cwru/<id>.mat`` (40 files) and ``data/raw/mfpt/*.mat``.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy import signal as sp_signal
from scipy import stats

from src.data.feature_extraction import batch_handcrafted_features
from src.review03 import train as T
from src.review03.data_real import (CWRU_CLASSES, MFPT_BEARING, MFPT_CLASSES, SKF_6205, Recording,
                                    Segment, build_arrays, class_ratios, cwru_protocol,
                                    load_cwru, load_mfpt, mfpt_protocol, spectral_views)
from src.review03.spectral import (build_masks, fault_evidence, spectral_loss_v1,
                                   spectral_loss_v2, valid_mask)
from src.utils.config import REPO_ROOT

OUT = Path(REPO_ROOT) / "outputs" / "review03"
SEEDS = [0, 1, 2, 3, 4]
WIN, HOP = 2048, 1024

P = T.PhysicsCfg
CONFIGS: Dict[str, T.TrainCfg] = {
    "cnn": T.TrainCfg(model="cnn"),
    "transformer": T.TrainCfg(model="transformer"),
    "cnn_lstm": T.TrainCfg(model="cnn_lstm"),
    "spec_v1_adapt": T.TrainCfg(physics=P(loss="v1", view="stft", sigma=40.0, adaptive=True)),
    "spec_v2_raw_adapt": T.TrainCfg(physics=P(loss="v2", view="raw", adaptive=True)),
    "spec_v2_env_adapt": T.TrainCfg(physics=P(loss="v2", view="env", adaptive=True)),
    "spec_v2_env_fix0.1": T.TrainCfg(physics=P(loss="v2", view="env", lam=0.1)),
    "spec_v2_env_fix0.5": T.TrainCfg(physics=P(loss="v2", view="env", lam=0.5)),
    "spec_v2_env_fix1.0": T.TrainCfg(physics=P(loss="v2", view="env", lam=1.0)),
    "spec_v2_env_adapt_colloc": T.TrainCfg(physics=P(loss="v2", view="env", adaptive=True,
                                                      collocation=True)),
}
XLOAD_CONFIGS = list(CONFIGS)
INDIST_CONFIGS = ["cnn", "transformer", "cnn_lstm", "spec_v1_adapt", "spec_v2_raw_adapt",
                  "spec_v2_env_adapt"]
SINGLE_CONFIGS = ["cnn_lstm", "spec_v2_raw_adapt", "spec_v2_env_adapt", "spec_v2_env_adapt_colloc"]
MFPT_CONFIGS = ["cnn", "cnn_lstm", "spec_v2_raw_adapt", "spec_v2_env_adapt",
                "spec_v2_env_fix0.5", "spec_v2_env_adapt_colloc"]
SENS_GRID = [(s, h) for s in (2.0, 5.0, 10.0, 20.0) for h in (1, 3, 5)]
SENS_SEEDS = [0]


# ---------------------------------------------------------------------------
# Robustness transforms (applied to TEST recordings only, before windowing)
# ---------------------------------------------------------------------------
def add_noise(rec: Recording, snr_db: float, seed: int) -> Recording:
    rng = np.random.default_rng(seed)
    p = float(np.mean(rec.signal.astype(np.float64) ** 2))
    noise = rng.normal(0.0, np.sqrt(p / 10 ** (snr_db / 10.0)), rec.signal.shape)
    return replace(rec, signal=(rec.signal + noise).astype(np.float32))


def speed_warp(rec: Recording, kappa: float = 1.0, mod_depth: float = 0.0,
               mod_hz: float = 0.5) -> Recording:
    """Emulate a shaft-speed change: time-warp x(t) -> x(phi(t)).

    phi'(t) = kappa * (1 + mod_depth * sin(2 pi mod_hz t)). A constant
    ``kappa`` scales *every* spectral component (defect frequencies and the
    structural resonances alike); ``mod_depth`` adds a slow speed fluctuation.
    This is a stress test, not a substitute for genuinely variable-speed data.
    """
    n = rec.signal.shape[0]
    t = np.arange(n) / rec.fs
    phase = kappa * (t - mod_depth * np.cos(2 * np.pi * mod_hz * t) / (2 * np.pi * mod_hz)
                     + mod_depth / (2 * np.pi * mod_hz))
    src = phase * rec.fs
    src = src[src <= n - 1]
    sig = np.interp(src, np.arange(n), rec.signal.astype(np.float64))
    return replace(rec, signal=sig.astype(np.float32), shaft_hz=rec.shaft_hz * kappa)


def robustness_sets(test_recs: List[Recording]) -> Dict[str, object]:
    variants = {f"noise_{s}dB": (lambda r, s=s: add_noise(r, s, seed=1234))
                for s in (10, 5, 0, -5)}
    variants.update({"speed_x0.95": lambda r: speed_warp(r, 0.95),
                     "speed_x1.05": lambda r: speed_warp(r, 1.05),
                     "speed_fluct5pct": lambda r: speed_warp(r, 1.0, 0.05, 0.5)})
    out = {}
    for name, fn in variants.items():
        segs = [Segment(fn(r), 0, 0, "test") for r in test_recs]
        segs = [replace(s, stop=s.rec.signal.shape[0]) for s in segs]
        arr, _ = build_arrays(segs, WIN, HOP)
        out[name] = arr["test"]
    return out


# ---------------------------------------------------------------------------
# Protocol data
# ---------------------------------------------------------------------------
_CACHE: Dict[str, object] = {}


def get_protocol(name: str):
    if name in _CACHE:
        return _CACHE[name]
    if name.startswith("mfpt"):
        recs = load_mfpt(Path(REPO_ROOT) / "data/raw/mfpt")
        segs = mfpt_protocol(recs, WIN)
        classes, geom = MFPT_CLASSES, MFPT_BEARING
    else:
        recs = load_cwru(Path(REPO_ROOT) / "data/raw/cwru")
        classes, geom = CWRU_CLASSES, SKF_6205
        if name == "single":
            recs = [r for r in recs if int(r.load) in (0, 3)]
            segs = cwru_protocol(recs, "cross_load")
        else:
            segs = cwru_protocol(recs, "cross_load" if name == "xload" else "in_distribution")
    arr, freqs = build_arrays(segs, WIN, HOP)
    if "test" in arr:
        arr["unlabeled"] = arr["test"]            # inputs only; labels never read in training
    extra = {}
    if name == "xload":
        # Robustness variants of the held-out 3 hp recordings (all test recordings
        # are whole recordings in this protocol).
        test_recs = sorted({s.rec.rec_id: s.rec for s in segs if s.split == "test"}.values(),
                           key=lambda r: r.rec_id)
        extra = robustness_sets(test_recs)
    out = {"arr": arr, "freqs": freqs, "classes": classes, "ratios": class_ratios(geom, classes),
           "extra": extra, "segs": segs}
    _CACHE[name] = out
    return out


def describe(name: str) -> Dict:
    d = get_protocol(name)
    info = {}
    for k, a in d["arr"].items():
        if k == "unlabeled":
            continue
        info[k] = {"windows": int(a.y.shape[0]), "per_class": np.bincount(a.y, minlength=len(d["classes"])).tolist(),
                   "recordings": int(np.unique(a.rec).size), "loads": sorted(set(map(float, a.load)))}
    return info


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _run_path(exp: str, cfg: str, seed: int) -> Path:
    return OUT / "runs" / exp / f"{cfg}__seed{seed}.json"


def run_cfg(exp: str, protocol: str, cfg_name: str, cfg: T.TrainCfg, seed: int) -> Dict:
    path = _run_path(exp, cfg_name, seed)
    if path.exists():
        return json.loads(path.read_text())
    d = get_protocol(protocol)
    if cfg.physics.collocation and "unlabeled" not in d["arr"]:
        raise RuntimeError("collocation requires an unlabeled split")
    t0 = time.time()
    res = T.run_neural(d["arr"], d["freqs"], d["ratios"], len(d["classes"]), cfg, seed,
                       extra_tests=d["extra"])
    res["config"] = cfg_name
    res["protocol"] = protocol
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(f"[{exp}] {cfg_name:28s} seed={seed} test_acc={res['test']['acc']:.4f} "
          f"f1={res['test']['f1']:.4f} epochs={res['epochs_run']} ({time.time() - t0:.0f}s)", flush=True)
    return res


def run_classical(exp: str, protocol: str) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    d = get_protocol(protocol)
    arr = d["arr"]
    fs = arr["train"].fs
    feats = {k: batch_handcrafted_features(arr[k].windows, fs=fs) for k in ("train", "test")}
    for k, a in d["extra"].items():
        feats[k] = batch_handcrafted_features(a.windows, fs=fs)
    for name in ("rf", "svm"):
        for seed in SEEDS:
            path = _run_path(exp, name, seed)
            if path.exists():
                continue
            if name == "rf":
                clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
            else:
                clf = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", class_weight="balanced",
                                                          random_state=seed))
            clf.fit(feats["train"], arr["train"].y)
            res = {"seed": seed, "config": name, "protocol": protocol}
            for k in feats:
                if k == "train":
                    continue
                y = arr["test"].y if k == "test" else d["extra"][k].y
                p = clf.predict(feats[k])
                res[k] = {"acc": float(accuracy_score(y, p)),
                          "f1": float(f1_score(y, p, average="macro", zero_division=0))}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, indent=1))
            print(f"[{exp}] {name} seed={seed} test_acc={res['test']['acc']:.4f}", flush=True)
            if name == "svm":
                break  # SVM is deterministic; one run suffices
        if name == "svm":
            # replicate the deterministic result for every seed for uniform aggregation
            base = json.loads(_run_path(exp, "svm", SEEDS[0]).read_text())
            for seed in SEEDS[1:]:
                p = _run_path(exp, "svm", seed)
                if not p.exists():
                    p.write_text(json.dumps({**base, "seed": seed}, indent=1))


def exp_xload():
    (OUT / "protocols").mkdir(parents=True, exist_ok=True)
    (OUT / "protocols" / "xload.json").write_text(json.dumps(describe("xload"), indent=1))
    run_classical("xload", "xload")
    for seed in SEEDS:
        for c in XLOAD_CONFIGS:
            run_cfg("xload", "xload", c, CONFIGS[c], seed)


def exp_indist():
    (OUT / "protocols").mkdir(parents=True, exist_ok=True)
    (OUT / "protocols" / "indist.json").write_text(json.dumps(describe("indist"), indent=1))
    run_classical("indist", "indist")
    for seed in SEEDS:
        for c in INDIST_CONFIGS:
            run_cfg("indist", "indist", c, CONFIGS[c], seed)


def exp_single():
    (OUT / "protocols").mkdir(parents=True, exist_ok=True)
    (OUT / "protocols" / "single.json").write_text(json.dumps(describe("single"), indent=1))
    for seed in SEEDS:
        for c in SINGLE_CONFIGS:
            run_cfg("single", "single", c, CONFIGS[c], seed)


def exp_mfpt():
    (OUT / "protocols").mkdir(parents=True, exist_ok=True)
    (OUT / "protocols" / "mfpt.json").write_text(json.dumps(describe("mfpt"), indent=1))
    run_classical("mfpt", "mfpt")
    for seed in SEEDS:
        for c in MFPT_CONFIGS:
            run_cfg("mfpt", "mfpt", c, CONFIGS[c], seed)


def exp_sens():
    base = CONFIGS["spec_v2_env_adapt"]
    for seed in SENS_SEEDS:
        for sigma, h in SENS_GRID:
            cfg = copy.deepcopy(base)
            cfg.physics.sigma, cfg.physics.harmonics = sigma, h
            run_cfg("sens", "xload", f"env_sigma{sigma:g}_H{h}", cfg, seed)


# ---------------------------------------------------------------------------
# L_spec audit (Reviewer point 3): no training, deterministic
# ---------------------------------------------------------------------------
def exp_audit():
    d = get_protocol("xload")
    tr = d["arr"]["train"]
    ratios, C = d["ratios"], len(d["classes"])
    valid = valid_mask(ratios, "cpu")
    shaft = torch.tensor(tr.shaft_hz)
    y = torch.tensor(tr.y)
    out: Dict[str, object] = {}

    # (a) Informativeness: physics-only fault-type accuracy (argmax evidence) per view.
    stft_f = np.linspace(0, tr.fs / 2, tr.stft_energy.shape[1]).astype(np.float32)
    views = {"stft64_v1": (torch.tensor(tr.stft_energy), stft_f, 40.0, "max"),
             "raw_hires": (torch.tensor(tr.raw_spec), d["freqs"], 5.0, "sum"),
             "env_hires": (torch.tensor(tr.env_spec), d["freqs"], 5.0, "sum")}
    fault = y > 0
    info = {}
    for k, (S, f, sig, norm) in views.items():
        m = build_masks(f, ratios, shaft, 3, sig, norm)
        ev = fault_evidence(S, m, valid)
        pred = ev[:, 1:].argmax(1) + 1
        per = {d["classes"][c]: float((pred[y == c] == c).float().mean()) for c in range(1, C)}
        # Spread of v1-style alignment across candidate classes (how much the prior can discriminate).
        s = S / S.sum(1, keepdim=True)
        mm = m / (m.sum(2, keepdim=True) + 1e-12)
        a = torch.einsum("bf,bcf->bc", s, mm)[:, 1:]
        info[k] = {"physics_only_fault_type_acc": float((pred[fault] == y[fault]).float().mean()),
                   "per_class": per,
                   "mean_alignment": float(a.mean()),
                   "mean_alignment_spread_max_minus_min": float((a.max(1).values - a.min(1).values).mean()),
                   "bin_width_hz": float(f[1] - f[0])}
    out["informativeness"] = info

    # (b) Defect bands vs STFT bins.
    f_max_defect = max(r for r in ratios if r) * 3 * float(tr.shaft_hz.max())
    out["v1_bins_covering_all_defect_harmonics"] = int(np.sum(stft_f <= f_max_defect + 40.0))
    out["v1_max_defect_harmonic_hz"] = f_max_defect

    # (c) Mask-mass coupling in v1 (unit-max masks): total mass per class.
    m1 = build_masks(stft_f, ratios, shaft[:1], 3, 40.0, "max")[0]
    out["v1_mask_mass_per_class"] = {d["classes"][c]: float(m1[c].sum()) for c in range(C)}

    # (d) Gradient behaviour w.r.t. the healthy logit.
    torch.manual_seed(0)
    idx = torch.arange(0, tr.y.shape[0], max(1, tr.y.shape[0] // 512))
    logits = torch.randn(idx.numel(), C, requires_grad=True)
    m_v1 = build_masks(stft_f, ratios, shaft[idx], 3, 40.0, "max")
    l1 = spectral_loss_v1(torch.tensor(tr.stft_energy)[idx], torch.softmax(logits, 1), m_v1, valid)
    g1, = torch.autograd.grad(l1, logits)
    logits2 = logits.detach().clone().requires_grad_(True)
    m_v2 = build_masks(d["freqs"], ratios, shaft[idx], 3, 5.0, "sum")
    l2 = spectral_loss_v2(torch.tensor(tr.env_spec)[idx], logits2, m_v2, valid)
    g2, = torch.autograd.grad(l2, logits2)
    yy = y[idx]
    out["gradient"] = {
        "v1_mean_abs_grad_healthy_logit": float(g1[:, 0].abs().mean()),
        "v1_frac_fault_windows_pushed_toward_healthy": float((g1[yy > 0, 0] < 0).float().mean()),
        "v1_mean_abs_grad_fault_logits": float(g1[:, 1:].abs().mean()),
        "v2_mean_abs_grad_healthy_logit": float(g2[:, 0].abs().mean()),
        "v2_mean_abs_grad_fault_logits": float(g2[:, 1:].abs().mean()),
        "v2_frac_true_class_logit_increased_by_descent": float(
            (-g2[yy > 0].gather(1, yy[yy > 0, None]).squeeze(1) > 0).float().mean()),
    }
    # (e) Envelope evidence and the healthy class: no healthy mask exists, so the
    #     term must not reward/penalise predicting healthy -- verified by (d).
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def ci95(v: List[float]):
    v = np.asarray(v, dtype=float)
    m = float(v.mean())
    if v.size < 2:
        return m, 0.0, 0.0
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.975, v.size - 1) * sd / np.sqrt(v.size))
    return m, sd, h


def summarise(exp: str, ref: str = "cnn_lstm") -> Dict:
    runs: Dict[str, Dict[int, Dict]] = {}
    for p in sorted((OUT / "runs" / exp).glob("*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["config"], {})[r["seed"]] = r
    out = {}
    for cfg, by_seed in runs.items():
        seeds = sorted(by_seed)
        entry = {"n": len(seeds), "seeds": seeds}
        keys = [k for k in by_seed[seeds[0]] if isinstance(by_seed[seeds[0]][k], dict) and "acc" in by_seed[seeds[0]][k]]
        for k in keys:
            for metric in ("acc", "f1"):
                m, sd, h = ci95([by_seed[s][k][metric] for s in seeds])
                entry[f"{k}_{metric}"] = {"mean": m, "sd": sd, "ci95": h}
            if "recall" in by_seed[seeds[0]][k]:
                entry[f"{k}_recall_mean"] = np.mean([by_seed[s][k]["recall"] for s in seeds], 0).tolist()
        if "val" in keys and "test" in keys:
            m, sd, h = ci95([by_seed[s]["val"]["acc"] - by_seed[s]["test"]["acc"] for s in seeds])
            entry["drop_val_to_test"] = {"mean": m, "sd": sd, "ci95": h}
        if cfg != ref and ref in runs:
            common = [s for s in seeds if s in runs[ref]]
            if len(common) >= 2:
                diff = [by_seed[s]["test"]["acc"] - runs[ref][s]["test"]["acc"] for s in common]
                m, sd, h = ci95(diff)
                tp = float(stats.ttest_rel([by_seed[s]["test"]["acc"] for s in common],
                                           [runs[ref][s]["test"]["acc"] for s in common]).pvalue) \
                    if np.std(diff) > 0 else (1.0 if np.all(np.asarray(diff) == 0) else None)
                try:
                    wp = float(stats.wilcoxon(diff).pvalue) if np.any(np.asarray(diff) != 0) else 1.0
                except ValueError:
                    wp = 1.0
                entry["paired_vs_ref"] = {"ref": ref, "mean_diff": m, "ci95": h, "t_p": tp, "wilcoxon_p": wp}
                # Same paired comparison on every evaluation set (robustness variants).
                allp = {}
                for k in keys:
                    if k in ("val", "test") or k not in runs[ref][common[0]]:
                        continue
                    a = [by_seed[s_][k]["acc"] for s_ in common]
                    b = [runs[ref][s_][k]["acc"] for s_ in common]
                    dd = np.subtract(a, b)
                    allp[k] = {"mean_diff": float(dd.mean()),
                               "t_p": float(stats.ttest_rel(a, b).pvalue) if dd.std() > 0
                               else (1.0 if np.all(dd == 0) else None)}
                entry["paired_vs_ref_all"] = allp
        if "history" in by_seed[seeds[0]]:
            lam = [np.mean([h["lam"] for h in by_seed[s]["history"]]) for s in seeds]
            entry["mean_lambda"] = float(np.mean(lam))
            entry["epochs_run_mean"] = float(np.mean([by_seed[s]["epochs_run"] for s in seeds]))
            entry["params"] = by_seed[seeds[0]]["params"]
        out[cfg] = entry
    return out


def exp_summary():
    summ = {e: summarise(e) for e in ("xload", "indist", "single", "mfpt", "sens")
            if (OUT / "runs" / e).exists()}
    for e in ("xload", "indist", "single", "mfpt"):
        p = OUT / "protocols" / f"{e}.json"
        if p.exists():
            summ.setdefault("_protocols", {})[e] = json.loads(p.read_text())
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    for e, s in summ.items():
        if e.startswith("_"):
            continue
        print(f"\n== {e} ==")
        for cfg, v in s.items():
            t = v.get("test_acc", {})
            pv = v.get("paired_vs_ref", {})
            print(f"  {cfg:30s} n={v['n']} test_acc={t.get('mean', float('nan')):.4f}±{t.get('ci95', 0):.4f} "
                  f"f1={v.get('test_f1', {}).get('mean', float('nan')):.4f} "
                  f"diff={pv.get('mean_diff', float('nan')):+.4f} p_t={pv.get('t_p')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["audit", "xload", "indist", "single", "sens", "mfpt", "summary", "all"])
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    args = ap.parse_args()
    if args.device:
        T.DEVICE = args.device
    elif torch.backends.mps.is_available():
        T.DEVICE = "mps"
    order = ["audit", "xload", "indist", "single", "mfpt", "sens", "summary"] if args.exp == "all" else [args.exp]
    for e in order:
        globals()[f"exp_{e}"]()


if __name__ == "__main__":
    main()
