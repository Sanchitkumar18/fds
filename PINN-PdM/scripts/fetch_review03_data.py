#!/usr/bin/env python3
"""Download the public benchmark files listed in the versioned manifests."""
import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]


def fetch(row, target):
    expected = row["sha256"]
    dest = target / row["file"]
    if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
        return
    request = Request(row["url"], headers={"User-Agent": "PINN-PdM benchmark downloader"})
    with urlopen(request, timeout=90) as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {dest.name}; see data/README.md")
    with NamedTemporaryFile(dir=target, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cwru", "mfpt", "all"))
    args = parser.parse_args()
    datasets = ("cwru", "mfpt") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        target = ROOT / "data" / "raw" / dataset
        target.mkdir(parents=True, exist_ok=True)
        with (ROOT / "data" / "manifests" / f"{dataset}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda row: fetch(row, target), rows))
        print(f"{dataset}: verified {len(rows)} recordings in {target}")


if __name__ == "__main__":
    main()
