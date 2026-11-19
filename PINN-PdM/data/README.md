# Evaluation datasets

The [manuscript](../papers/pinn4-2.pdf) uses 40 CWRU drive-end recordings and 10 MFPT recordings. The manifests list the exact files selected for this reconstruction, their labels and source URLs, and the SHA-256 of each verified download.

| Dataset | Source and selection | Repository data |
| --- | --- | --- |
| CWRU | Case Western Reserve University Bearing Data Center. Four [normal baselines](https://engineering.case.edu/bearingdatacenter/normal-baseline-data) and 36 [12 kHz drive-end fault recordings](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data), including centered outer-race faults. | [Recording manifest](manifests/cwru.csv) · [Window index](manifests/cwru-window-index.csv) |
| MFPT | MFPT Society bearing fault recordings. [MathWorks mirror at the pinned revision](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/tree/d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d). The selected baseline and fixed-load outer-race recordings are the first replicates. | [Recording manifest](manifests/mfpt.csv) · [Window index](manifests/mfpt-window-index.csv) |
| CWRU-PG | Generated from CWRU vibration recordings, using the procedure in Sections IV and V and Appendix A. | [Generator](../scripts/build_derived_dataset.py) |

From `PINN-PdM/`, run:

```bash
python scripts/fetch_review03_data.py all
python scripts/build_derived_dataset.py
python scripts/build_derived_dataset.py --dataset mfpt
```

The downloader verifies the manifests and stores MATLAB files under `data/raw/cwru/` and `data/raw/mfpt/`. For manual placement, download the files listed in each manifest using the same filenames. Raw recordings and generated feature archives are kept out of Git. For offline use, extract the separately packaged evaluation-data ZIP inside `PINN-PdM/` to restore both the MATLAB files and the prepared feature archives.

## Data dictionary

The generated datasets live in `data/processed/review03/{cwru,mfpt}/`. Each condition has a compressed NPZ and a matching CSV index. Row order matches across the index and NPZ arrays.

- `record` identifies the raw MATLAB file. `start` is a zero-based offset in the processed, decimated recording. Windows contain 2,048 consecutive samples with a hop of 1,024 samples within each segment.
- `y` is the integer class label (0 healthy, 1 inner, 2 outer, 3 ball). MFPT uses labels 0–2. `load` is in horsepower for CWRU and pounds for MFPT. `diameter_in` is the CWRU defect diameter in inches.
- `shaft_hz` and `sample_rate` describe each window. CWRU normal data are decimated from 48 kHz to 12 kHz. Files 98 and 99 lack embedded RPM, so their fallback values come from the official baseline table. MFPT data are decimated to 12,207 Hz.
- `stft`, `core` and `handcrafted` hold the 64×64 model input, the four waveform statistics, and the 12 classical classifier features. Training fits feature standardization separately for each protocol.
- `raw` and `envelope` are raw and squared-envelope power spectra through 1 kHz. `defect_hz` and `evidence` contain kinematic frequencies and normalized class evidence. The fault frequency order is inner, outer, ball for CWRU and inner, outer for MFPT. `evidence_prediction` chooses among fault types only.
- Protocol columns assign `train`, `val`, `test` or `unused`. The clean index is the union of whole-recording and protocol-specific window grids, so its total can exceed the manuscript's whole-recording grid count. Guard gaps and overlapping train/test sample intervals are checked for each protocol.
- The seven CWRU stress archives cover 3 hp test windows under four noise levels and three time warps. Time-warp starts are measured in the warped trace. Windows beyond a sped-up recording's usable endpoint are omitted.
