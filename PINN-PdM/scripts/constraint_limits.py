#!/usr/bin/env python3
"""Constraint values and limits from CWRU-PG (no training).

* defect-frequency constraint ranges from the recorded shaft speeds;
* per-class ranges (5th-95th percentile) of RMS, kurtosis, crest factor;
* healthy-vs-faulty alarm limits on each statistic, chosen on the 0-2 hp
  training windows and checked on the unseen 3 hp windows;
* the evidence-margin threshold above which the physics label is reliable.

    python scripts/constraint_limits.py   # -> outputs/review03/constraint_limits.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import numpy as np

from src.utils.config import REPO_ROOT

D = np.load(Path(REPO_ROOT) / "data/derived/cwru_pg/cwru_pg.npz", allow_pickle=False)
y, st, cf, ev, pred = D["label"], D["stats4"], D["char_freqs_bpfo_bpfi_bsf"], D["evidence"], D["evidence_pred"]
split = D["split_xload"]
names = [str(c) for c in D["class_names"]]
tr, te = split == "train", split == "test"
out = {}

out["defect_freq_hz"] = {k: [float(cf[:, i].min()), float(cf[:, i].max())]
                         for i, k in enumerate(["BPFO", "BPFI", "BSF"])}
out["shaft_hz"] = [float(D["shaft_hz"].min()), float(D["shaft_hz"].max())]

stat_names = ["rms", "kurtosis", "crest", "skewness"]
out["class_ranges_p5_p50_p95"] = {
    names[c]: {s: [float(np.percentile(st[y == c, j], q)) for q in (5, 50, 95)]
               for j, s in enumerate(stat_names[:3])} for c in range(4)}

# Healthy-vs-faulty alarm limit per statistic: threshold maximising balanced
# accuracy on the training windows, reported on the unseen 3 hp windows.
alarm = {}
fault = y > 0
for j, s in enumerate(stat_names[:3]):
    x = st[:, j]
    cands = np.unique(np.percentile(x[tr], np.linspace(1, 99, 197)))
    def bacc(mask, t):
        p = x[mask] > t
        return 0.5 * (p[fault[mask]].mean() + (~p[~fault[mask]]).mean())
    t = max(cands, key=lambda t: bacc(tr, t))
    p_te = x[te] > t
    alarm[s] = {"limit": float(t), "train_bal_acc": float(bacc(tr, t)), "test_bal_acc": float(bacc(te, t)),
                "test_fault_detect": float(p_te[fault[te]].mean()),
                "test_false_alarm": float(p_te[~fault[te]].mean())}
out["healthy_vs_fault_alarm"] = alarm

# Evidence margin = best minus second-best fault evidence (best is 1 by construction).
e = np.sort(ev[:, 1:], axis=1)
margin = e[:, -1] - e[:, -2]
ok = pred == y
rel = {}
for target in (0.90, 0.95):
    m_tr = fault & tr
    ts = np.linspace(0, 0.99, 100)
    good = [t for t in ts if (margin[m_tr] >= t).sum() >= 20 and ok[m_tr][margin[m_tr] >= t].mean() >= target]
    t = float(min(good)) if good else None
    m_te = fault & te
    sel = margin[m_te] >= t if t is not None else np.zeros(m_te.sum(), bool)
    rel[f"{int(target*100)}pct"] = {
        "margin_threshold": t,
        "train_coverage": float((margin[m_tr] >= t).mean()) if t is not None else 0.0,
        "test_accuracy": float(ok[m_te][sel].mean()) if sel.any() else None,
        "test_coverage": float(sel.mean()),
        "test_accuracy_below": float(ok[m_te][~sel].mean()) if (~sel).any() else None}
out["evidence_reliability"] = rel
out["evidence_margin_quartiles_fault"] = [float(np.percentile(margin[fault], q)) for q in (25, 50, 75)]

(Path(REPO_ROOT) / "outputs/review03/constraint_limits.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
