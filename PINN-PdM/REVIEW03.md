# Revised method and reproducibility

This implementation reconstructs the revised bearing diagnosis experiments from the [supplied manuscript](papers/pinn4-2.pdf), especially Sections IV–V and Appendix A. It reuses the repository's STFT preprocessing, handcrafted features, classifier architectures, and original spectral loss. The manuscript's original Review 03 source and run archive was unavailable. Use the manuscript for its published experiment tables and use the commands below to generate results for this implementation.

## Prepare the environment and dataset

Run these commands from `PINN-PdM/`:

```bash
python -m pip install -r requirements.txt
python scripts/fetch_review03_data.py all
python scripts/build_derived_dataset.py
python scripts/build_derived_dataset.py --dataset mfpt
```

See the [dataset documentation](data/README.md) for provenance, sample rates, split labels and the feature schema. For a quick clean-data run, pass `--clean-only` to the CWRU builder. Generated feature archives can be placed under `data/processed/review03/` to skip data preparation.

## Run the experiments

```bash
# One-seed check of the matched neural baseline
python scripts/run_review03.py --exp cross_load --model cnn_lstm --prior none --epochs 1 --seeds 0

# All four protocols, models and spectral-prior variants
python scripts/run_review03.py --exp all
python scripts/make_review03_tables.py
python scripts/make_review03_figs.py
```

The runner uses 30 epochs with early stopping and five seeds for CWRU or two seeds for MFPT. It evaluates random forest, RBF-SVM, CNN, Transformer and CNN–LSTM. CNN–LSTM variants cover the original coarse STFT prior, high-resolution raw and squared-envelope priors, adaptive and fixed weights, and an explicitly transductive variant. Only that variant uses held-out inputs during training, and it uses them without class labels. The CNN–LSTM uses one STFT map per example through the existing backbone.

The scripts write per-run accuracy and macro F1, an evidence sensitivity sweep, mean and 95% Student-t intervals, and paired tests against the CNN–LSTM baseline where enough paired seeds are available. Cross-load runs also report accuracy on any prepared stress variants. The scripts generate their own CSV and figure from those results. The original manuscript's figure layouts, auxiliary audit and residual tables, and exact historical runs require the authors' original archive.

## Checks

```bash
python -m pytest tests/
```
