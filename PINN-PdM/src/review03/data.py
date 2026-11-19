"""Real recording ingestion and leakage-controlled window construction."""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import signal, io

from src.data.feature_extraction import batch_handcrafted_features
from src.data.preprocessing import batch_stft_maps
from src.physics.bearing_frequencies import SKF_6205, BearingGeometry, fault_frequency_ratios

ROOT = Path(__file__).resolve().parents[2]
WIN, HOP = 2048, 1024
CLASSES = ("healthy", "inner", "outer", "ball")
NICE = BearingGeometry(8, .235, 1.245)
PROTOCOLS = {"cwru": ("in_distribution", "cross_load", "single_source"),
             "mfpt": ("load_shift",)}
STRESSES = ("noise_10", "noise_5", "noise_0", "noise_-5",
            "speed_095", "speed_105", "speed_sine")


@dataclass(frozen=True)
class Recording:
    name: str
    vibration: np.ndarray
    fs: float
    shaft_hz: float
    label: int
    load: int
    diameter_in: str


def read_manifest(dataset):
    with (ROOT / "data" / "manifests" / f"{dataset}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_recordings(dataset, raw_dir=None):
    raw_dir = Path(raw_dir) if raw_dir else ROOT / "data" / "raw" / dataset
    rows = read_manifest(dataset)
    missing = [row["file"] for row in rows if not (raw_dir / row["file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {dataset} recordings at {raw_dir}: {', '.join(missing)}")
    for row in rows:
        path = raw_dir / row["file"]
        mat = io.loadmat(path, simplify_cells=True)
        fs = int(row["sample_rate"])
        load = int(row["load"])
        if dataset == "cwru":
            prefix = f"X{int(path.stem):03}"
            vibration = mat[prefix + "_DE_time"]
            shaft_hz = float(mat.get(prefix + "RPM", row["fallback_rpm"])) / 60
            decimation = fs // 12000
        else:
            bearing = mat["bearing"]
            vibration = bearing["gs"]
            shaft_hz = float(bearing["rate"])
            if int(bearing["sr"]) != fs or shaft_hz != 25:
                raise ValueError(f"Unexpected MFPT sampling or shaft frequency: {path}")
            if "load" in bearing and int(bearing["load"]) != load:
                raise ValueError(f"Unexpected MFPT load: {path}")
            decimation = 8 if fs == 97656 else 4
        vibration = np.asarray(vibration, dtype=np.float64).ravel()
        if not np.isfinite(vibration).all() or not np.isfinite(shaft_hz) or shaft_hz <= 0:
            raise ValueError(f"Invalid vibration or shaft speed: {path}")
        if decimation > 1:
            vibration = signal.resample_poly(vibration, 1, decimation)
        yield Recording(path.stem, vibration, fs / decimation, shaft_hz,
                        CLASSES.index(row["class"]), load, row.get("diameter_in", ""))


def time_blocks(length, names, fractions):
    """Allocate usable samples to contiguous blocks separated by one window."""
    usable = length - WIN * (len(names) - 1)
    if usable <= 0:
        raise ValueError(f"Recording too short for {len(names)} guarded blocks")
    edges = np.rint(np.cumsum((0, *fractions)) * usable).astype(int)
    return [(name, edges[i] + i * WIN, edges[i + 1] + i * WIN)
            for i, name in enumerate(names)]


def split_blocks(record, protocol):
    length = len(record.vibration)
    if protocol == "in_distribution" or (protocol == "load_shift" and record.label == 0):
        return time_blocks(length, ("train", "val", "test"), (.70, .15, .15))
    if ((protocol == "load_shift" and record.load >= 270)
            or (protocol in ("cross_load", "single_source") and record.load == 3)):
        return [("test", 0, length)]
    if protocol == "single_source" and record.load != 0:
        return []
    return time_blocks(length, ("train", "val"), (.85, .15))


def window_index(record, protocols):
    grids = {protocol: {
        int(start): split
        for split, lo, hi in split_blocks(record, protocol)
        for start in range(lo, hi - WIN + 1, HOP)
    } for protocol in protocols}
    full_grid = range(0, len(record.vibration) - WIN + 1, HOP)
    starts = np.array(sorted(set(full_grid).union(*(grid for grid in grids.values()))))
    labels = {protocol: np.array([grid.get(int(start), "unused") for start in starts])
              for protocol, grid in grids.items()}
    for protocol, split in labels.items():
        used = split != "unused"
        different = split[used][1:] != split[used][:-1]
        if np.any(np.diff(starts[used])[different] < WIN):
            raise ValueError(f"Overlapping splits: {record.name}, {protocol}")
    return starts, labels


def stress_record(record, variant, rng):
    vibration = record.vibration
    if variant.startswith("noise_"):
        snr = float(variant.split("_")[1])
        noise = rng.standard_normal(len(vibration))
        noise *= np.sqrt(np.mean(vibration ** 2) / np.mean(noise ** 2) / 10 ** (snr / 10))
        return replace(record, vibration=vibration + noise)
    time = np.arange(len(vibration)) / record.fs
    if variant == "speed_sine":
        phase = 2 * np.pi * .5 * time
        warped_time = time + .05 * (1 - np.cos(phase)) / (2 * np.pi * .5)
        rate = 1 + .05 * np.sin(phase)
    else:
        rate = {"speed_095": .95, "speed_105": 1.05}[variant]
        warped_time = time * rate
    # A constant speed-up shortens the usable recording. Avoid padding its tail.
    valid = warped_time <= time[-1]
    shaft_hz = record.shaft_hz * np.broadcast_to(rate, time.shape)[valid]
    return replace(record, vibration=np.interp(warped_time[valid], time, vibration)), shaft_hz


def spectral_views(windows, fs):
    frequencies = np.fft.rfftfreq(WIN, 1 / fs)
    keep = frequencies <= 1000
    centered = windows - windows.mean(axis=1, keepdims=True)
    hann = np.hanning(WIN)
    raw = np.abs(np.fft.rfft(centered * hann)) ** 2
    band = signal.butter(4, [1000, min(5500, .45 * fs)], "bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(band, centered, axis=1)
    envelope = np.abs(signal.hilbert(filtered, axis=1)) ** 2
    envelope -= envelope.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(envelope * hann)) ** 2
    return raw[:, keep], power[:, keep], frequencies[keep]


def ratios_for(dataset):
    geometry = SKF_6205 if dataset == "cwru" else NICE
    ratios = fault_frequency_ratios(geometry)
    labels = CLASSES if dataset == "cwru" else CLASSES[:3]
    return np.array([ratios.get(label, 0) for label in labels])


def spectral_evidence(spectrum, frequencies, shaft_hz, ratios, sigma=5, harmonics=3):
    centers = shaft_hz[:, None, None, None] * ratios[None, :, None, None]
    centers = centers * np.arange(1, harmonics + 1)[None, None, :, None]
    masks = np.exp(-.5 * ((frequencies - centers) / sigma) ** 2)
    masks *= (centers > 0) & (centers <= frequencies[-1])
    masks = masks.sum(axis=2)
    masks /= masks.sum(axis=-1, keepdims=True) + 1e-12
    spectrum = spectrum / (spectrum.sum(axis=1, keepdims=True) + 1e-12)
    evidence = np.einsum("bf,bcf->bc", spectrum, masks)
    return (evidence / (evidence.max(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def build_features(recordings, dataset, variant="clean", seed=0):
    protocols = PROTOCOLS[dataset]
    ratios = ratios_for(dataset)
    rng = np.random.default_rng(seed)
    batches = []
    for record in recordings:
        if variant != "clean" and record.load != 3:
            continue
        speed = None
        if variant.startswith("speed_"):
            record, speed = stress_record(record, variant, rng)
        elif variant.startswith("noise_"):
            record = stress_record(record, variant, rng)
        starts, splits = window_index(record, protocols)
        indexes = starts[:, None] + np.arange(WIN)
        windows = record.vibration[indexes]
        shaft = np.full(len(starts), record.shaft_hz) if speed is None else speed[indexes].mean(axis=1)
        stft, _ = batch_stft_maps(windows, fs=record.fs)
        features = batch_handcrafted_features(windows, fs=record.fs)
        raw, envelope, frequencies = spectral_views(windows, record.fs)
        evidence = spectral_evidence(envelope, frequencies, shaft, ratios)
        batch = {
            "record": np.repeat(record.name, len(starts)),
            "start": starts,
            "y": np.full(len(starts), record.label),
            "load": np.full(len(starts), record.load),
            "diameter_in": np.repeat(record.diameter_in, len(starts)),
            "shaft_hz": shaft,
            "sample_rate": np.full(len(starts), record.fs),
            "stft": stft, "core": features[:, :4], "handcrafted": features,
            "raw": raw.astype(np.float32), "envelope": envelope.astype(np.float32),
            "defect_hz": shaft[:, None] * ratios[1:],
            "evidence": evidence,
            "evidence_prediction": 1 + evidence[:, 1:].argmax(axis=1),
            **splits,
        }
        batches.append(batch)
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
