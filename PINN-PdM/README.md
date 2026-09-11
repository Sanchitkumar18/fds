# PINN-PdM — Predictive Maintenance of Industrial Machinery using Physics-Informed Neural Networks

A complete, runnable research codebase implementing **PINN-PdM**, a physics-informed
predictive-maintenance system for rotating machinery, as specified in *Case Study 1*
(problem/data/literature) and *Case Study 2* (model & innovation).

It embeds two governing physics priors into a shared CNN-LSTM backbone:

1. **Bearing characteristic-frequency consistency** (diagnosis) — a differentiable
   spectral loss that anchors each fault-class decision to the theoretical
   defect frequency (BPFO/BPFI/BSF/FTF) of that class.
2. **Degradation ODE + monotonicity** (prognosis) — `dH/dt + ρ·H = 0` enforced by
   automatic differentiation, plus a penalty forbidding health from spontaneously
   increasing.

and benchmarks it against RF/SVM, CNN, LSTM, CNN-LSTM and Transformer baselines,
with a controlled **cross-load extrapolation** protocol (train 0/1/2 HP → test 3 HP).

---

## 1. Project overview

Purely data-driven fault-diagnosis / RUL models saturate in-distribution but degrade
on operating conditions unseen in training and can violate physics (e.g. predicting a
*recovering* bearing). PINN-PdM injects known bearing physics into the loss so
predictions stay physically consistent and generalise better under distribution shift.

## 2. Research question

> Can embedding known physical degradation and vibration-dynamics equations into a
> neural network's loss function improve fault diagnosis and RUL prediction compared
> with purely data-driven baselines, **particularly when evaluating on operating
> conditions not seen during training**?

## 3. Motivation

Case Study 1's survey of 18 papers found that PINNs' extrapolation advantage is
**asserted more often than measured** for discrete rotating machinery. This project
turns that qualitative claim into a controlled, measured experiment.

## 4. Architecture

```mermaid
flowchart LR
    A[Raw vibration window<br/>2048 samples] --> B[STFT 64x64]
    A --> C[4 stat features<br/>RMS, kurtosis,<br/>crest, skew]
    B --> D[Conv stack<br/>16→32→64, 3x3]
    D --> E[2-layer LSTM<br/>128 hidden]
    C --> F
    E --> F[Dense 128 + ReLU + Dropout]
    F --> G[Diagnosis head<br/>Dense 64 → softmax]
    F --> H[Prognosis head<br/>Dense 64 → H t]
    G -. L_spec .-> P[(Physics losses)]
    H -. L_phys + L_mono .-> P
```

Composite objective (Case Study 2, §2.5):

```
L = L_data + λ_phys·L_phys + λ_mono·L_mono + λ_spec·L_spec
```

with **self-adaptive** `λ` (gradient-norm balancing, Wang et al. 2021 / SoftAdapt).

## 5. Physics equations

Bearing characteristic defect frequencies (shaft speed `fr`, `n` balls, ball dia `d`,
pitch dia `D`, contact angle `φ`):

```
BPFO = (n/2)·fr·(1 − (d/D)cosφ)      # outer race
BPFI = (n/2)·fr·(1 + (d/D)cosφ)      # inner race
BSF  = (D/2d)·fr·(1 − ((d/D)cosφ)²)  # ball
FTF  = (1/2)·fr·(1 − (d/D)cosφ)      # cage
```

Degradation ODE residual (autodiff in `t`) and monotonicity:

```
r_phys(t) = dH/dt + ρ·H(t) = 0,  ρ>0        L_phys = mean(r_phys²)
L_mono = mean(ReLU(dH/dt)²)
```

`ρ` is **co-estimated** (a learnable, softplus-positive parameter), initialised from a
least-squares fit on the training health trajectories.

## 6. Dataset requirements

| Dataset | Source | Use | Placement |
|---|---|---|---|
| **CWRU** | https://engineering.case.edu/bearingdatacenter | Fault diagnosis | `data/raw/cwru/*.mat` |
| **IMS** | NASA PCoE / U. Cincinnati | RUL / degradation | `data/raw/ims/<test folders>` |
| **C-MAPSS FD001** (optional) | NASA PCoE | RUL stretch | `data/raw/cmapss/*_FD001.txt` |

Datasets are **not** auto-downloaded (they require accepting terms). Run
`python scripts/download_data.py` to print the official sources and create the folders.
**If a dataset is absent, the pipeline uses a documented, physics-consistent synthetic
stand-in** so everything runs end-to-end; such runs are flagged `synthetic_fallback: true`
in the metrics and must not be read as real-data benchmarks.

## 7. Installation

```bash
# Python 3.10+ recommended (code also runs on 3.9)
pip install -r requirements.txt
# or:  conda env create -f environment.yml && conda activate pinn-pdm
```

CUDA / Apple-MPS GPUs are auto-detected; otherwise CPU is used.

## 8. Dataset preparation

```bash
python scripts/download_data.py          # print sources, make folders
python scripts/prepare_cwru.py           # build STFT+feature cache (+ EDA figures)
python scripts/prepare_ims.py
```

## 9. Training

```bash
python scripts/train_baselines.py --dataset cwru      # RF, SVM, CNN, LSTM, CNN-LSTM, Transformer
python scripts/train_pinn.py       --dataset cwru      # PINN-PdM (diagnosis)
python scripts/train_pinn.py       --dataset ims       # PINN-PdM (RUL)
python scripts/train_pinn.py       --dataset synthetic # PINN-PdM (proof-of-concept)
```

## 10. Evaluation

```bash
python scripts/evaluate.py --experiment cwru
python scripts/evaluate.py --experiment ims
```

## 11. Extrapolation experiment (the headline)

`run_experiments.py` trains every model twice — in-distribution (70/15/15) and
cross-condition (train 0/1/2 HP, test **3 HP**) — and reports the accuracy drop:

```bash
python scripts/run_experiments.py --experiment cwru
```

## 12. Ablation study

```bash
python scripts/run_experiments.py --experiment ablation
```
Variants: backbone-only → +ODE → +monotonicity → full PINN (+adaptive weighting).

## 13. Results

Run the full suite, then read the generated report and tables:

```bash
python scripts/run_experiments.py --experiment all      # add --quick for a smoke run
python scripts/generate_report.py
```

Outputs land in `outputs/`:
- `outputs/metrics/*.json` — measured metrics (with `case_study_projected` blocks kept separate)
- `outputs/tables/*.csv` + `outputs/report/tables.md` — Tables 1–6
- `outputs/figures/*.png` — all plots
- `outputs/report/technical_report.md` — full technical report

> **Honesty policy.** Every number in the tables is freshly measured by the code.
> The Case Study's projected targets (e.g. CWRU 3 HP: PINN 95.8%) are shown only in
> clearly-labelled `CS projected` columns and are **never** presented as our
> measurements. If a baseline beats PINN-PdM in a run, the tables show it.

## 14. Folder structure

```
PINN-PdM/
├── config/            # cwru.yaml, ims.yaml, cmapps.yaml, synthetic.yaml
├── src/
│   ├── data/          # loaders, segmentation, preprocessing (STFT), features, splits, synthetic
│   ├── physics/       # bearing_frequencies, degradation, physics_losses, collocation
│   ├── models/        # cnn, lstm, cnn_lstm, transformer, pinn_pdm, common, factory, classical
│   ├── training/      # trainer, pinn_trainer, adaptive_weights, callbacks
│   ├── evaluation/    # classification/rul/physics metrics, extrapolation, ablation, predict
│   ├── visualization/ # signal/training/rul/comparison plots, confusion matrix
│   ├── utils/         # seed, logging, checkpoint, config
│   ├── pipeline.py    # data→split→loader→train orchestration
│   ├── experiments.py # experiment runners
│   └── reporting.py   # tables + technical report
├── scripts/           # CLI entry points
├── tests/             # pytest suite
├── notebooks/         # EDA + results notebooks
└── outputs/           # figures, metrics, tables, report (generated)
```

## 15. Reproducibility

Fixed seeds (`src/utils/seed.py`), deterministic flags, config-driven experiments,
saved checkpoints (`checkpoints/`), metrics JSON, CSV training histories and (optional)
TensorBoard logs. Re-run any experiment with a single command.

## 16. Limitations

- Without the real CWRU/IMS files the pipeline runs on a **synthetic stand-in**; those
  numbers validate the mechanism, not real-data performance.
- The degradation prior is first-order exponential; multi-stage/Paris-law dynamics are
  future work.
- Reported numbers use a modest default training budget; raise `training.epochs` in the
  configs for publication-grade results.

## 17. Future work

Multi-stage degradation priors; validation on long-duration industrial streams with
sensor drift; coupling the physics-residual head with generative models for data-scarce
fault synthesis; a fuller sensitivity study of the self-adaptive weighting.

## 18. References

Key references (full list in the Case Studies): Raissi et al. (2019); Wang, Teng &
Perdikaris (2021); McClenny & Braga-Neto (2023); de Oliveira da Costa et al. (2022,
T4PdM); Saxena et al. (2008, PHM08); Yucesan & Viana (2021); CWRU Bearing Data Center;
NASA IMS / C-MAPSS repositories.

---

*Authors: Vaishvi Verma (23BCE0016), Tanishq Garg (23BCE0094), Sanchit Kumar (23BCE0043).
Guide: Prof. Kalyanaraman P, SCOPE, VIT Vellore.*
