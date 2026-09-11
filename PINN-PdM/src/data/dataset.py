"""Unified processed-dataset assembly and the PyTorch ``PdMDataset``.

A *raw* dataset is a dict of aligned per-window arrays (see ``synthetic.py`` and
the loaders): ``raw``, ``label``, ``health``, ``rul``, ``t``, ``unit_id``,
``record_id``, ``load_hp``. ``build_processed`` turns raw windows into the
64x64 STFT maps and 4 statistical features and returns a self-describing
processed dict that is cached to ``.npz``.

``PdMDataset`` serves samples to the models. Each item is a *sequence* of
``seq_len`` consecutive windows within one unit (padded at the start), matching
the CNN-LSTM backbone that runs an LSTM over a window sequence. The label / RUL
/ health / time correspond to the last window in the sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .feature_extraction import batch_core_features, batch_handcrafted_features
from .preprocessing import StandardScaler1D, batch_stft_maps


def build_processed(raw_data: Dict[str, np.ndarray], fs: float, stft_size: int = 64,
                    compute_handcrafted: bool = True) -> Dict[str, np.ndarray]:
    """Compute STFT maps + statistical features from raw windows.

    Returns a processed dict containing everything needed downstream. Handles an
    empty input gracefully.
    """
    raw = raw_data["raw"]
    stft, freq_axis = batch_stft_maps(raw, fs=fs, out_size=stft_size)
    feats = batch_core_features(raw)
    processed: Dict[str, np.ndarray] = {
        "stft": stft.astype(np.float32),                 # (N, S, S)
        "feats": feats.astype(np.float32),               # (N, 4)
        "freq_axis": freq_axis.astype(np.float32),       # (S,)
        "label": raw_data["label"].astype(np.int64),
        "health": raw_data.get("health", np.full(len(raw), -1.0)).astype(np.float32),
        "health_clean": raw_data.get("health_clean",
                                     raw_data.get("health", np.full(len(raw), -1.0))).astype(np.float32),
        "rul": raw_data.get("rul", np.full(len(raw), -1.0)).astype(np.float32),
        "t": raw_data.get("t", np.zeros(len(raw))).astype(np.float32),
        "unit_id": raw_data.get("unit_id", np.zeros(len(raw))).astype(np.int64),
        "record_id": raw_data.get("record_id", np.arange(len(raw))).astype(np.int64),
        "load_hp": raw_data.get("load_hp", np.full(len(raw), -1)).astype(np.int64),
    }
    if compute_handcrafted:
        processed["handcrafted"] = batch_handcrafted_features(raw, fs=fs).astype(np.float32)
    return processed


def save_processed(processed: Dict[str, np.ndarray], path: str | Path,
                   extra_meta: Optional[dict] = None) -> Path:
    """Cache a processed dataset to a ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in processed.items()}
    if extra_meta:
        payload["_meta"] = np.array([str(extra_meta)], dtype=object)
    np.savez_compressed(path, **payload)
    return path


def load_processed(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a cached processed ``.npz`` dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}. Run the prepare script.")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def apply_feature_scaler(processed: Dict[str, np.ndarray], train_idx: np.ndarray,
                         key: str = "feats") -> StandardScaler1D:
    """Fit a StandardScaler on ``processed[key][train_idx]`` and apply in-place.

    Fitting on training indices only prevents leakage (Case Study #5, #28).
    Returns the fitted scaler so its parameters can be persisted.
    """
    scaler = StandardScaler1D().fit(processed[key][train_idx])
    processed[key] = scaler.transform(processed[key])
    return scaler


class PdMDataset:
    """PyTorch-style dataset yielding window sequences for the CNN-LSTM backbone.

    Parameters
    ----------
    processed:
        Processed dict from :func:`build_processed`.
    indices:
        Sample indices (into the N dimension) this split covers.
    seq_len:
        Number of consecutive windows per sample (LSTM sequence length). The
        preceding ``seq_len-1`` windows are taken from the *same unit*; the start
        of a unit is padded by repeating the first window.
    task:
        ``"diagnosis"``, ``"prognosis"`` or ``"both"``.
    """

    def __init__(self, processed: Dict[str, np.ndarray], indices: np.ndarray,
                 seq_len: int = 1, task: str = "both") -> None:
        import torch  # local import so importing the module doesn't require torch

        self.torch = torch
        self.p = processed
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len = max(1, int(seq_len))
        self.task = task
        self.stft = processed["stft"]
        self.feats = processed["feats"]
        self.unit_id = processed["unit_id"]
        # Precompute, per global index, the position within its unit and the
        # ordered list of that unit's indices (sorted by time) for sequence building.
        self._unit_positions: Dict[int, np.ndarray] = {}
        order = np.argsort(processed["t"], kind="stable")
        for u in np.unique(self.unit_id):
            u_idx = np.where(self.unit_id == u)[0]
            u_idx = u_idx[np.argsort(processed["t"][u_idx], kind="stable")]
            self._unit_positions[int(u)] = u_idx
        self._pos_in_unit = {int(g): int(np.where(self._unit_positions[int(self.unit_id[g])] == g)[0][0])
                             for g in self.indices}

    def __len__(self) -> int:
        return len(self.indices)

    def _sequence_indices(self, g: int) -> List[int]:
        """Return the global indices of the seq_len-window sequence ending at g."""
        u = int(self.unit_id[g])
        u_order = self._unit_positions[u]
        pos = int(np.where(u_order == g)[0][0])
        start = pos - self.seq_len + 1
        seq = []
        for j in range(start, pos + 1):
            seq.append(int(u_order[max(0, j)]))  # pad by repeating the first window
        return seq

    def __getitem__(self, i: int) -> dict:
        torch = self.torch
        g = int(self.indices[i])
        seq_idx = self._sequence_indices(g)
        x = self.stft[seq_idx]                       # (S, H, W)
        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).float()  # (S,1,H,W)
        feat = torch.from_numpy(np.ascontiguousarray(self.feats[g])).float()  # (4,)
        sample = {
            "x": x,
            "feat": feat,
            "y": torch.tensor(int(self.p["label"][g]), dtype=torch.long),
            "t": torch.tensor([float(self.p["t"][g])], dtype=torch.float32),
            "rul": torch.tensor(float(self.p["rul"][g]), dtype=torch.float32),
            "health": torch.tensor(float(self.p["health"][g]), dtype=torch.float32),
            "health_clean": torch.tensor(float(self.p["health_clean"][g]), dtype=torch.float32),
            "unit_id": torch.tensor(int(self.unit_id[g]), dtype=torch.long),
            "load_hp": torch.tensor(int(self.p["load_hp"][g]), dtype=torch.long),
            "index": torch.tensor(g, dtype=torch.long),
        }
        return sample


def collate(batch: List[dict]) -> dict:
    """Default collate for :class:`PdMDataset` items."""
    import torch

    out = {}
    out["x"] = torch.stack([b["x"] for b in batch], dim=0)        # (B,S,1,H,W)
    out["feat"] = torch.stack([b["feat"] for b in batch], dim=0)  # (B,4)
    out["y"] = torch.stack([b["y"] for b in batch], dim=0)
    out["t"] = torch.stack([b["t"] for b in batch], dim=0)        # (B,1)
    out["rul"] = torch.stack([b["rul"] for b in batch], dim=0)
    out["health"] = torch.stack([b["health"] for b in batch], dim=0)
    out["health_clean"] = torch.stack([b["health_clean"] for b in batch], dim=0)
    out["unit_id"] = torch.stack([b["unit_id"] for b in batch], dim=0)
    out["load_hp"] = torch.stack([b["load_hp"] for b in batch], dim=0)
    out["index"] = torch.stack([b["index"] for b in batch], dim=0)
    return out
