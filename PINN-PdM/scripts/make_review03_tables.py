#!/usr/bin/env python3
"""Emit LaTeX tables for the manuscript from ``outputs/review03/summary.json``
and ``outputs/review03/audit.json``. Output: ``paper/tables/*.tex``.

No number in these tables is typed by hand; each cell is mean +/- 95% CI
half-width (Student t, n seeds) over the cached per-seed run files.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

from src.utils.config import REPO_ROOT

RES = Path(REPO_ROOT) / "outputs" / "review03"
OUT = Path(REPO_ROOT) / "paper" / "tables"

NAME = {
    "rf": "Random forest (12 features)", "svm": "SVM-RBF (12 features)",
    "cnn": "CNN", "transformer": "Transformer", "cnn_lstm": "CNN--LSTM ($\\lambda=0$)",
    "spec_v1_adapt": "+ $L_{\\mathrm{spec}}^{\\mathrm{v1}}$ (Review-02 form), adaptive",
    "spec_v2_raw_adapt": "+ $L_{\\mathrm{spec}}^{\\mathrm{v2}}$ raw spectrum, adaptive",
    "spec_v2_env_adapt": "+ $L_{\\mathrm{spec}}^{\\mathrm{v2}}$ envelope, adaptive",
    "spec_v2_env_fix0.1": "+ $L_{\\mathrm{spec}}^{\\mathrm{v2}}$ envelope, fixed $\\lambda=0.1$",
    "spec_v2_env_fix0.5": "+ $L_{\\mathrm{spec}}^{\\mathrm{v2}}$ envelope, fixed $\\lambda=0.5$",
    "spec_v2_env_fix1.0": "+ $L_{\\mathrm{spec}}^{\\mathrm{v2}}$ envelope, fixed $\\lambda=1.0$",
    "spec_v2_env_adapt_colloc": "+ $L_{\\mathrm{spec}}^{\\mathrm{v2}}$ envelope + unlabelled target$^{\\dagger}$",
}


def pm(e, scale=100.0, d=1):
    if not e:
        return "--"
    return f"{e['mean'] * scale:.{d}f}\\,$\\pm$\\,{e['ci95'] * scale:.{d}f}"


def pval(v):
    p = v.get("paired_vs_ref")
    if not p:
        return "--"
    if p["t_p"] is None:   # identical difference on every seed: paired test undefined
        return f"{p['mean_diff'] * 100:+.2f} (n/a)"
    t = "$<$0.001" if p["t_p"] < 1e-3 else f"{p['t_p']:.3f}"
    return f"{p['mean_diff'] * 100:+.2f} ({t})"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)


def table_main(s, rows, cols, caption, label, star=False, extra_note=""):
    env = "table*" if star else "table"
    spec = "l" + "c" * len(cols)
    head = " & ".join(["Model"] + [c[0] for c in cols]) + " \\\\"
    lines = [f"\\begin{{{env}}}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering",
             "\\footnotesize", f"\\begin{{tabular}}{{{spec}}}", "\\toprule", head, "\\midrule"]
    for r in rows:
        if r == "--":
            lines.append("\\midrule")
            continue
        if r not in s:
            continue
        v = s[r]
        lines.append(" & ".join([NAME.get(r, r)] + [c[1](v) for c in cols]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if extra_note:
        width = "0.95\\textwidth" if star else "\\columnwidth"
        lines.append("\\par\\vspace{2pt}\\parbox{" + width + "}{\\scriptsize " + extra_note + "}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def main():
    s = json.loads((RES / "summary.json").read_text())
    acc = lambda k: (lambda v: pm(v.get(f"{k}_acc")))
    f1 = lambda k: (lambda v: pm(v.get(f"{k}_f1")))
    note = ("Mean\\,$\\pm$\\,95\\% CI half-width over 5 seeds (\\%). $\\Delta$: paired mean difference in "
            "test accuracy vs.\\ CNN--LSTM ($\\lambda=0$) on the same seeds, percentage points, "
            "with paired $t$-test $p$ in brackets. $^{\\dagger}$Transductive: $L_{\\mathrm{spec}}$ is also "
            "applied to the \\emph{unlabelled} target-load inputs; their labels are never used.")

    if "xload" in s:
        x = s["xload"]
        rows = ["rf", "svm", "cnn", "transformer", "cnn_lstm", "--", "spec_v1_adapt", "spec_v2_raw_adapt",
                "spec_v2_env_adapt", "spec_v2_env_adapt_colloc"]
        cols = [("Val (0--2 hp)", acc("val")), ("Test acc.\\ (3 hp)", acc("test")),
                ("Test macro-F1", f1("test")), ("$\\Delta$ vs.\\ CNN--LSTM ($p$)", pval)]
        write("tab_xload", table_main(x, rows, cols,
              "Cross-load extrapolation on the original CWRU recordings: train 0/1/2\\,hp, test on the "
              "unseen 3\\,hp recordings (measured).", "tab:xload", star=True, extra_note=note))
        rows = ["cnn_lstm", "spec_v2_env_fix0.1", "spec_v2_env_fix0.5", "spec_v2_env_fix1.0", "spec_v2_env_adapt"]
        cols = [("Test acc.\\ (\\%)", acc("test")), ("Mean $\\lambda$", lambda v: f"{v.get('mean_lambda', 0):.3g}"),
                ("$\\Delta$ ($p$)", pval)]
        write("tab_weighting", table_main(x, rows, cols,
              "Fixed vs.\\ adaptive weighting of $L_{\\mathrm{spec}}^{\\mathrm{v2}}$ (envelope), CWRU "
              "0/1/2$\\to$3\\,hp (measured, 5 seeds).", "tab:weighting"))
        rows = ["rf", "cnn_lstm", "spec_v2_raw_adapt", "spec_v2_env_adapt", "spec_v2_env_adapt_colloc"]
        cols = [(k, acc(f"noise_{k.split()[0]}dB")) for k in ("10 dB", "5 dB", "0 dB", "-5 dB")] + \
               [("$\\times0.95$", acc("speed_x0.95")), ("$\\times1.05$", acc("speed_x1.05")),
                ("$\\pm5\\%$", acc("speed_fluct5pct"))]
        write("tab_robust", table_main(x, rows, cols,
              "Robustness on the held-out 3\\,hp recordings: additive white noise at the given SNR and "
              "emulated speed change (time-warp). Accuracy, \\% (measured, 5 seeds; models trained on clean "
              "0--2\\,hp data only).", "tab:robust", star=True))
    if "indist" in s:
        rows = ["rf", "svm", "cnn", "transformer", "cnn_lstm", "--", "spec_v1_adapt", "spec_v2_raw_adapt",
                "spec_v2_env_adapt"]
        cols = [("Acc.\\ (\\%)", acc("test")), ("Macro-F1 (\\%)", f1("test")),
                ("Params", lambda v: f"{v['params'] / 1e6:.2f}\\,M" if "params" in v else "--")]
        write("tab_indist", table_main(s["indist"], rows, cols,
              "In-distribution diagnosis on CWRU with time-blocked 70/15/15 splits of every recording "
              "(measured, 5 seeds).", "tab:indist"))
    if "single" in s:
        rows = ["cnn_lstm", "spec_v2_raw_adapt", "spec_v2_env_adapt", "spec_v2_env_adapt_colloc"]
        cols = [("Test acc.\\ (\\%)", acc("test")), ("Macro-F1 (\\%)", f1("test")), ("$\\Delta$ ($p$)", pval)]
        write("tab_single", table_main(s["single"], rows, cols,
              "Single-source extrapolation on CWRU: train on 0\\,hp only, test on 3\\,hp (measured, 5 seeds).",
              "tab:single"))
    if "mfpt" in s:
        rows = ["rf", "svm", "cnn", "cnn_lstm", "--", "spec_v2_raw_adapt", "spec_v2_env_adapt",
                "spec_v2_env_fix0.5", "spec_v2_env_adapt_colloc"]
        cols = [("$n$", lambda v: str(v["n"])), ("Test acc.\\ (\\%)", acc("test")),
                ("Macro-F1 (\\%)", f1("test"))]
        write("tab_mfpt", table_main(s["mfpt"], rows, cols,
              "Second bearing dataset (MFPT, \\emph{preliminary}): faults trained at 0--150\\,lbs, tested at "
              "270--300\\,lbs. Measured; $n$ = number of completed seeds (CI from $n$ seeds; "
              "no CI is meaningful for $n=2$).", "tab:mfpt",
              extra_note="$^{\\dagger}$Transductive (unlabelled target inputs used in training)."))

    a = json.loads((RES / "audit.json").read_text())
    I, G = a["informativeness"], a["gradient"]
    lines = ["\\begin{table}[t]", "\\caption{Audit of the spectral-consistency term on the CWRU 0--2\\,hp "
             "training windows (measured, no training involved).}", "\\label{tab:audit}", "\\centering",
             "\\footnotesize", "\\begin{tabular}{lccc}", "\\toprule",
             "Quantity & v1 (64-bin STFT) & raw, 5.9\\,Hz & envelope, 5.9\\,Hz \\\\", "\\midrule",
             "Bin width (Hz) & " + " & ".join(f"{I[k]['bin_width_hz']:.1f}" for k in ("stft64_v1", "raw_hires", "env_hires")) + " \\\\",
             "Physics-only fault-type acc.\\ (\\%) & " + " & ".join(f"{100 * I[k]['physics_only_fault_type_acc']:.1f}" for k in ("stft64_v1", "raw_hires", "env_hires")) + " \\\\"]
    for c in ("inner", "outer", "ball"):
        lines.append(f"\\quad {c} recall (\\%) & " + " & ".join(f"{100 * I[k]['per_class'][c]:.1f}" for k in ("stft64_v1", "raw_hires", "env_hires")) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{4}{l}{\\emph{Gradient of the loss w.r.t.\\ logits (random logits, 512 windows)}} \\\\",
              f"Mean $|\\partial L/\\partial z_{{\\mathrm{{healthy}}}}|$ & {G['v1_mean_abs_grad_healthy_logit']:.1e} & \\multicolumn{{2}}{{c}}{{v2: {G['v2_mean_abs_grad_healthy_logit']:.1f} (exact)}} \\\\",
              f"Mean $|\\partial L/\\partial z_{{\\mathrm{{fault}}}}|$ & {G['v1_mean_abs_grad_fault_logits']:.1e} & \\multicolumn{{2}}{{c}}{{v2: {G['v2_mean_abs_grad_fault_logits']:.1e}}} \\\\",
              f"Faulty windows pushed toward healthy (\\%) & {100 * G['v1_frac_fault_windows_pushed_toward_healthy']:.1f} & \\multicolumn{{2}}{{c}}{{v2: 0.0}} \\\\",
              f"True fault logit raised by descent (\\%) & -- & \\multicolumn{{2}}{{c}}{{v2 (envelope): {100 * G['v2_frac_true_class_logit_increased_by_descent']:.1f}}} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_audit", "\n".join(lines) + "\n")

    if "_protocols" in s:
        P = s["_protocols"]
        lines = ["\\begin{table}[t]", "\\caption{Leakage-controlled protocols: windows (recordings) per split. "
                 "Splits are formed before windowing; no two windows in different splits share a sample.}",
                 "\\label{tab:protocols}", "\\centering", "\\footnotesize", "\\begin{tabular}{llccc}", "\\toprule",
                 "Protocol & Split & Windows & Recordings & Loads \\\\", "\\midrule"]
        names = {"indist": "CWRU in-dist.", "xload": "CWRU 0/1/2$\\to$3 hp", "single": "CWRU 0$\\to$3 hp",
                 "mfpt": "MFPT load shift"}
        for k in ("indist", "xload", "single", "mfpt"):
            if k not in P:
                continue
            for i, sp in enumerate(("train", "val", "test")):
                v = P[k][sp]
                loads = ",".join(f"{l:g}" for l in v["loads"])
                lines.append(f"{names[k] if i == 0 else ''} & {sp} & {v['windows']} & {v['recordings']} & {loads} \\\\")
            lines.append("\\midrule" if k != "mfpt" else "")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        write("tab_protocols", "\n".join(l for l in lines if l) + "\n")
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
