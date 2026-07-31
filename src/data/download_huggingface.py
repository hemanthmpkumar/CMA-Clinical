#!/usr/bin/env python3
"""
src/data/download_huggingface.py

Download the requested MIMIC subsets directly from Hugging Face using the
Hugging Face Hub API. This avoids assumptions about dataset builder formats
and works for CSV tables, WebDataset-style shards, images, and reports.

Supported HF dataset repositories:
  • MIMIC-III: ntphuc149/MIMIC-III-Clinical-Database
  • MIMIC-IV:  lucky9-cyou/mimic-iv-aligned-ppg-ecg
  • MIMIC-CXR: Yamini-1628/MIMIC-CXR-RRG

Set HF_TOKEN as an environment variable. The token is intentionally NOT read
from command-line arguments so it is not exposed in shell history.
"""

import argparse
import fnmatch
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

# DATASETS = {
#     "mimiciii": {
#         "id": "ntphuc149/MIMIC-III-Clinical-Database",
#         "repo_type": "dataset",
#         "note": "Clinical tables derived from MIMIC-III",
#         "default_include": ["*.csv"],
#         "default_exclude": [],
#     },
#     "mimiciv": {
#         "id": "lucky9-cyou/mimic-iv-aligned-ppg-ecg",
#         "repo_type": "dataset",
#         "note": "Aligned PPG/ECG subset derived from MIMIC-IV",
#         "default_include": ["*"],
#         "default_exclude": [],
#     },
#     "mimiciv-cxr": {
#         "id": "Yamini-1628/MIMIC-CXR-RRG",
#         "repo_type": "dataset",
#         "note": "MIMIC-CXR radiology report generation subset",
#         "default_include": ["*"],
#         "default_exclude": [],
#     },
# }

DATASETS = {

    "mimiciii": {
        "id": "ntphuc149/MIMIC-III-Clinical-Database",
        "repo_type": "dataset",
        "note": "Full MIMIC-III clinical tables including notes",
        "default_include": [
            "*.csv",
            "*.csv.gz"
        ],
        "default_exclude": [],
    },


    "mimiciv-note": {
        "id": "mimic-capstone/mimic-iv-note",
        "repo_type": "dataset",
        "note": "MIMIC-IV clinical notes (discharge, radiology, etc.)",
        "default_include": [
            "*.parquet",
            "*.json",
            "*.csv"
        ],
        "default_exclude": [],
    },


    "mimiciv-ppg-ecg": {
        "id": "lucky9-cyou/mimic-iv-aligned-ppg-ecg",
        "repo_type": "dataset",
        "note": "MIMIC-IV aligned ECG/PPG signals",
        "default_include": [
            "*"
        ],
        "default_exclude": [],
    },


    "mimic-cxr-rrg": {
        "id": "Yamini-1628/MIMIC-CXR-RRG",
        "repo_type": "dataset",
        "note": "MIMIC-CXR radiology report generation dataset",
        "default_include": [
            "*"
        ],
        "default_exclude": [],
    },

}


def get_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Set HF_TOKEN environment variable before running this script.",
              file=sys.stderr)
        sys.exit(1)
    return token


def _matches_filters(fname: str, include: list[str], exclude: list[str]) -> bool:
    if include:
        if not any(fnmatch.fnmatch(fname, pat) for pat in include):
            return False
    if exclude:
        if any(fnmatch.fnmatch(fname, pat) for pat in exclude):
            return False
    return True


def inspect_dataset(name: str, spec: dict, token: str):
    print(f"\n=== Inspecting: {name} ({spec['id']}) ===")
    print(f"  {spec['note']}")
    try:
        files = list_repo_files(spec["id"], repo_type=spec["repo_type"], token=token)
    except Exception as exc:
        print(f"  Could not list files: {exc}", file=sys.stderr)
        return

    total_bytes = 0
    for fname in files:
        print(f"    {fname}")
    print(f"  Files listed: {len(files)}")


def inspect_dataset_detailed(name: str, spec: dict, token: str):
    """List files (attempt to include sizes when the local Hub library supports it)."""
    print(f"\n=== Inspecting: {name} ({spec['id']}) ===")
    print(f"  {spec['note']}")
    try:
        files = list_repo_files(spec["id"], repo_type=spec["repo_type"], token=token)
    except Exception as exc:
        print(f"  Could not list files: {exc}", file=sys.stderr)
        return

    # Try to attach size information using the model_info / dataset_info helper.
    size_map = {}
    try:
        from huggingface_hub import dataset_info
        info = dataset_info(spec["id"], token=token)
        for sibling in getattr(info, "siblings", []):
            if getattr(sibling, "rfilename", None) and getattr(sibling, "size", None):
                size_map[sibling.rfilename] = sibling.size
    except Exception:
        pass

    total = 0
    for entry in files:
        fname = str(entry)
        size = size_map.get(fname)
        if size is not None:
            print(f"    {fname:80s} {size:>15,} bytes")
            total += size
        else:
            print(f"    {fname}")
    print(f"  Files listed: {len(files)} (size total where reported: {total / 1e9:.2f} GB)")


def download_dataset(name: str, spec: dict, out_dir: Path, token: str,
                     include: list[str] = None, exclude: list[str] = None,
                     resume: bool = True):
    target_dir = out_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)

    include = include or spec["default_include"]
    exclude = exclude or spec["default_exclude"]

    print(f"\n=== Downloading: {name} ({spec['id']}) ===")
    print(f"  Output directory: {target_dir}")
    print(f"  Include patterns: {include}")
    print(f"  Exclude patterns: {exclude}")

    # If broad include filter, use snapshot_download and let allow_patterns do the work.
    try:
        downloaded = snapshot_download(
            repo_id=spec["id"],
            repo_type=spec["repo_type"],
            token=token,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=include,
            ignore_patterns=exclude,
            resume_download=resume,
        )
        print(f"  Saved to: {downloaded}")
        return downloaded
    except Exception as exc:
        print(f"  Error during snapshot download: {exc}", file=sys.stderr)
        return None


def download_single_files(name: str, spec: dict, out_dir: Path, token: str,
                          include: list[str] = None, exclude: list[str] = None,
                          resume: bool = True):
    """Alternative to snapshot_download for fine-grained file-by-file progress."""
    target_dir = out_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)

    include = include or spec["default_include"]
    exclude = exclude or spec["default_exclude"]

    print(f"\n=== Downloading file-by-file: {name} ({spec['id']}) ===")
    try:
        files = list_repo_files(spec["id"], repo_type=spec["repo_type"], token=token)
    except Exception as exc:
        print(f"  Could not list files: {exc}", file=sys.stderr)
        return []

    selected = [f for f in files if _matches_filters(f, include, exclude)]
    downloaded = []
    for fname in selected:
        local_path = target_dir / fname
        if local_path.exists() and resume:
            print(f"  -> {fname} already present, skipping.")
            downloaded.append(local_path)
            continue
        print(f"  -> {fname}")
        try:
            path = hf_hub_download(
                repo_id=spec["id"],
                repo_type=spec["repo_type"],
                filename=fname,
                token=token,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=resume,
            )
            downloaded.append(Path(path))
        except Exception as exc:
            print(f"     Error: {exc}", file=sys.stderr)
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download MIMIC subsets from Hugging Face for the CMA benchmark"
    )
    parser.add_argument(
        "--dataset",
        choices=["mimiciii","mimiciv-note","mimiciv-ppg-ecg","mimic-cxr-rrg","all"],
        default=None,
        help="Which Hugging Face dataset to fetch.",
    )
    parser.add_argument(
        "--out", default="data/raw/huggingface", help="Output directory."
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Only list files (with sizes) in each dataset, do not download.",
    )
    parser.add_argument(
        "--file-by-file", action="store_true",
        help="Download files one-by-one instead of using snapshot_download.",
    )
    parser.add_argument(
        "--include", nargs="+", default=None,
        help="Glob patterns of files to download (e.g., NOTEEVENTS.csv *.txt)."
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Glob patterns of files to skip."
    )
    parser.add_argument(
        "--instructions", action="store_true",
        help="Print usage instructions and exit.",
    )
    args = parser.parse_args()

    if args.instructions or args.dataset is None:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CMA Clinical Search Benchmark — Hugging Face Data Download                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Required:
  export HF_TOKEN=hf_...

Inspect one dataset (list files + approximate sizes):
  python src/data/download_huggingface.py --dataset mimiciii --inspect

Download all tables for one dataset:
  python src/data/download_huggingface.py --dataset mimiciii --out data/raw/huggingface

Download only specific files:
  python src/data/download_huggingface.py --dataset mimiciii --include "*EVENTS.csv" "ADMISSIONS.csv"

Download all three supported datasets:
  python src/data/download_huggingface.py --dataset all --out data/raw/huggingface

# Datasets used (as requested):
#   • MIMIC-III: ntphuc149/MIMIC-III-Clinical-Database
#   • MIMIC-IV:  lucky9-cyou/mimic-iv-aligned-ppg-ecg
#   • MIMIC-CXR: Yamini-1628/MIMIC-CXR-RRG
Datasets:

  • MIMIC-III Clinical Database
      ntphuc149/MIMIC-III-Clinical-Database

  • MIMIC-IV Notes
      mimic-capstone/mimic-iv-note

  • MIMIC-IV ECG/PPG
      lucky9-cyou/mimic-iv-aligned-ppg-ecg

  • MIMIC-CXR Report Generation
      Yamini-1628/MIMIC-CXR-RRG

Notes:
  • Downloaded PHI/notes must stay under data/raw/ (git-ignored).
  • The MIMIC-IV Hugging Face repository is an aligned PPG/ECG subset, not the
    full MIMIC-IV clinical-note release. If you need clinical notes, use the
    PhysioNet downloader (src/data/download.py) or a HF dataset that contains
    the note tables.
""")
        return

    token = get_token()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(DATASETS.items()) if args.dataset == "all" else [
        (args.dataset, DATASETS[args.dataset])
    ]

    for name, spec in selected:
        if args.inspect:
            inspect_dataset_detailed(name, spec, token)
        elif args.file_by_file:
            download_single_files(
                name, spec, out_dir, token,
                include=args.include, exclude=args.exclude
            )
        else:
            download_dataset(
                name, spec, out_dir, token,
                include=args.include, exclude=args.exclude
            )

    print(f"\nOutputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
