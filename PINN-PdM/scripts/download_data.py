#!/usr/bin/env python3
"""Document dataset sources and prepare raw directories.

This project does NOT silently download datasets from arbitrary websites
(Case Study requirement #4). CWRU and IMS require manual download/acceptance of
terms; this script prints the official sources and creates the target folders.
Run ``scripts/prepare_*.py`` afterwards -- if a dataset is absent, a documented
physics-consistent synthetic stand-in is used so the pipeline still runs.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from pathlib import Path

from src.utils.config import REPO_ROOT

SOURCES = {
    "CWRU": {
        "url": "https://engineering.case.edu/bearingdatacenter/download-data-file",
        "place": "data/raw/cwru/",
        "notes": "Download drive-end (DE) .mat files for Normal, Inner-race, Outer-race and "
                 "Ball faults at 0-3 HP. Optionally add a catalog.csv with columns "
                 "filename,class,load_hp to label files explicitly.",
    },
    "IMS": {
        "url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/"
               "pcoe/pcoe-data-set-repository/",
        "place": "data/raw/ims/",
        "notes": "Download the IMS Bearings dataset and extract the test folders "
                 "(1st_test/, 2nd_test/, 3rd_test/) here.",
    },
    "C-MAPSS": {
        "url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/"
               "pcoe/pcoe-data-set-repository/",
        "place": "data/raw/cmapss/",
        "notes": "Optional/stretch. Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt here.",
    },
}


def main() -> None:
    print("=" * 74)
    print("PINN-PdM dataset sources (manual download required; no silent fetching)")
    print("=" * 74)
    for name, info in SOURCES.items():
        target = Path(REPO_ROOT) / info["place"]
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name}]")
        print(f"  Official source : {info['url']}")
        print(f"  Place files in  : {target}")
        print(f"  Notes           : {info['notes']}")
    print("\nAfter placing files, run:")
    print("  python scripts/prepare_cwru.py")
    print("  python scripts/prepare_ims.py")
    print("If a dataset is missing, a documented SYNTHETIC stand-in is used automatically.")


if __name__ == "__main__":
    main()
