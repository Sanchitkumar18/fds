#!/usr/bin/env python3
"""Build CWRU-PG, a physics-guided dataset derived from the original CWRU recordings.

Every window is cut from a real recording (no synthesis). For each window the
dataset stores:

* provenance: recording id, class, fault diameter, motor load, recorded shaft
  speed, start sample, and the split it belongs to under each leakage-free
  protocol (in-distribution time blocks, 0/1/2->3 hp, 0->3 hp);
* network inputs: 64x64 STFT map (float16) and 4 waveform statistics;
* 12 handcrafted features (classical baselines);
* physics views: raw and squared-envelope power spectra, 0-1 kHz, 5.9 Hz bins;
* physics annotations: characteristic frequencies (BPFO/BPFI/BSF) of the
  window, per-fault-type evidence ebar under the default masks, and the
  evidence-only fault-type prediction;
* stress-test variants of the 3 hp recordings: additive white noise
  (10/5/0/-5 dB) and emulated speed change (x0.95, x1.05, +/-5 %), with the same
  features, spectra and annotations.

Output: data/derived/cwru_pg/{cwru_pg.npz, cwru_pg_index.csv, stress_*.npz,
DATASHEET.md, summary.json}. Raw windows are not stored (they are regenerated
from the CWRU files and the stored start indices), which keeps the dataset small.

    python scripts/build_derived_dataset.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.data.feature_extraction import batch_handcrafted_features
from src.physics.bearing_frequencies import all_fault_frequencies
from src.review03.data_real import (CWRU_CLASSES, SKF_6205, Segment, build_arrays,
                                    class_ratios, cwru_protocol, load_cwru)
from src.review03.spectral import build_masks, fault_evidence, valid_mask
from src.utils.config import REPO_ROOT

from run_review03 import add_noise, speed_warp  # same transforms as the experiments

OUT = Path(REPO_ROOT) / "data" / "derived" / "cwru_pg"
WIN, HOP = 2048, 1024
SIGMA, H = 5.0, 3


def annotate(arr, freqs, ratios):
    """Physics annotations: evidence per class and evidence-only prediction."""
    m = build_masks(freqs, ratios, torch.tensor(arr.shaft_hz), H, SIGMA, "sum")
    ev = fault_evidence(torch.tensor(arr.env_spec), m, valid_mask(ratios, "cpu")).numpy()
    pred = ev[:, 1:].argmax(1) + 1
    return ev.astype(np.float32), pred.astype(np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_cwru(Path(REPO_ROOT) / "data/raw/cwru")
    ratios = class_ratios(SKF_6205, CWRU_CLASSES)
    meta = {r.rec_id: r for r in recs}

    # One full-coverage windowing per recording (the in-distribution blocks are a
    # subset); split membership under each protocol is recorded as a column.
    full = [Segment(r, 0, r.signal.shape[0], "all") for r in recs]
    arr, freqs = build_arrays(full, WIN, HOP)
    a = arr["all"]

    def membership(protocol_segs):
        lab = np.full(a.y.shape[0], "unused", dtype=object)
        for s in protocol_segs:
            sel = (a.rec == s.rec.rec_id) & (a.start >= s.start) & (a.start + WIN <= s.stop)
            lab[sel] = s.split
        return lab

    split_indist = membership(cwru_protocol(recs, "in_distribution"))
    split_xload = membership(cwru_protocol(recs, "cross_load"))
    split_single = membership(cwru_protocol([r for r in recs if int(r.load) in (0, 3)], "cross_load"))

    ev, ev_pred = annotate(a, freqs, ratios)
    hand = batch_handcrafted_features(a.windows, fs=a.fs).astype(np.float32)
    cf = np.array([[all_fault_frequencies(f, SKF_6205)[k] for k in ("outer", "inner", "ball")]
                   for f in a.shaft_hz], dtype=np.float32)
    sev = np.array([meta[r].severity for r in a.rec], dtype=np.float32)

    np.savez_compressed(
        OUT / "cwru_pg.npz",
        stft=a.x.astype(np.float16), stats4=a.feat, handcrafted12=hand,
        raw_spec=a.raw_spec.astype(np.float16), env_spec=a.env_spec.astype(np.float16),
        spec_freqs=freqs, label=a.y, class_names=np.array(CWRU_CLASSES),
        load_hp=a.load, shaft_hz=a.shaft_hz, severity_in=sev, recording=a.rec, start=a.start,
        char_freqs_bpfo_bpfi_bsf=cf, evidence=ev, evidence_pred=ev_pred,
        split_indist=split_indist.astype(str), split_xload=split_xload.astype(str),
        split_single=split_single.astype(str))

    with open(OUT / "cwru_pg_index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "recording", "start", "class", "severity_in", "load_hp", "shaft_hz",
                    "bpfo_hz", "bpfi_hz", "bsf_hz", "split_indist", "split_xload", "split_single",
                    "rms", "kurtosis", "crest", "skewness",
                    "ev_inner", "ev_outer", "ev_ball", "evidence_pred"])
        for i in range(a.y.shape[0]):
            w.writerow([i, a.rec[i], int(a.start[i]), CWRU_CLASSES[a.y[i]], sev[i], int(a.load[i]),
                        round(float(a.shaft_hz[i]), 3), *[round(float(x), 2) for x in cf[i]],
                        split_indist[i], split_xload[i], split_single[i],
                        *[round(float(x), 5) for x in a.feat[i]],
                        *[round(float(x), 4) for x in ev[i, 1:]], CWRU_CLASSES[ev_pred[i]]])

    # Stress-test variants of the held-out 3 hp recordings.
    test_recs = [r for r in recs if int(r.load) == 3]
    variants = {f"noise_{s}dB": (lambda r, s=s: add_noise(r, s, seed=1234)) for s in (10, 5, 0, -5)}
    variants.update({"speed_x0.95": lambda r: speed_warp(r, 0.95),
                     "speed_x1.05": lambda r: speed_warp(r, 1.05),
                     "speed_fluct5pct": lambda r: speed_warp(r, 1.0, 0.05, 0.5)})
    stress = {}
    for name, fn in variants.items():
        segs = [Segment(fn(r), 0, 0, "test") for r in test_recs]
        segs = [replace(s, stop=s.rec.signal.shape[0]) for s in segs]
        v = build_arrays(segs, WIN, HOP)[0]["test"]
        vev, vpred = annotate(v, freqs, ratios)
        np.savez_compressed(OUT / f"stress_{name}.npz", stft=v.x.astype(np.float16), stats4=v.feat,
                            env_spec=v.env_spec.astype(np.float16), raw_spec=v.raw_spec.astype(np.float16),
                            label=v.y, recording=v.rec, start=v.start, shaft_hz=v.shaft_hz,
                            evidence=vev, evidence_pred=vpred)
        f = v.y > 0
        stress[name] = {"windows": int(v.y.shape[0]),
                        "evidence_acc_fault": float((vpred[f] == v.y[f]).mean())}

    f = a.y > 0
    summary = {
        "windows": int(a.y.shape[0]), "recordings": len(recs),
        "per_class": {c: int((a.y == i).sum()) for i, c in enumerate(CWRU_CLASSES)},
        "per_load": {str(l): int((a.load == l).sum()) for l in (0, 1, 2, 3)},
        "evidence_acc_fault_all": float((ev_pred[f] == a.y[f]).mean()),
        "evidence_acc_by_class": {CWRU_CLASSES[c]: float((ev_pred[a.y == c] == c).mean()) for c in (1, 2, 3)},
        "evidence_acc_by_load": {str(l): float((ev_pred[f & (a.load == l)] == a.y[f & (a.load == l)]).mean())
                                 for l in (0, 1, 2, 3)},
        "evidence_acc_by_severity": {str(s): float((ev_pred[f & (sev == s)] == a.y[f & (sev == s)]).mean())
                                     for s in (0.007, 0.014, 0.021)},
        "stress": stress,
        "unused_windows": {"indist": int((split_indist == "unused").sum()),
                           "xload": int((split_xload == "unused").sum())},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
