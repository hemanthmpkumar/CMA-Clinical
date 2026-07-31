#!/usr/bin/env python3
"""
src/data/download_hf.py

Download MIMIC-related datasets directly from Hugging Face.

Supported datasets:
  • MIMIC-III: ntphuc149/MIMIC-III-Clinical-Database
  • MIMIC-IV:  lucky9-cyou/MIMIC-IV
  • MIMIC-CXR: EvidenceAIResearch/MIMIC-CXR-VReason
  • eICU:      wshi83/EHRAgent-eicu

Requires an HF token in the environment:
  export HF_TOKEN=hf_...
"""

import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

DATASETS = {
    "mimiciii": {
        "id": "ntphuc149/MIMIC-III-Clinical-Database",
        "repo_type": "dataset",
        "note": "MIMIC-III clinical database snapshot",
        "default_include": ["*"],
        "default_exclude": [],
    },
    "mimiciv": {
        "id": "lucky9-cyou/MIMIC-IV",
        "repo_type": "dataset",
        "note": "MIMIC-IV dataset snapshot",
        "default_include": ["*"],
        "default_exclude": [],
    },
    "mimiciv-cxr": {
        "id": "EvidenceAIResearch/MIMIC-CXR-VReason",
        "repo_type": "dataset",
        "note": "MIMIC-CXR VReason dataset snapshot",
        "default_include": ["*"],
        "default_exclude": [],
    },
    "eicu": {
        "id": "wshi83/EHRAgent-eicu",
        "repo_type": "dataset",
        "note": "eICU dataset snapshot",
        "default_include": ["*"],
        "default_exclude": [],
    },
}


def get_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Set HF_TOKEN before running this script.", file=sys.stderr)
        sys.exit(1)
    return token


def _matches_filters(fname: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(fnmatch.fnmatch(fname, pat) for pat in include):
        return False
    if exclude and any(fnmatch.fnmatch(fname, pat) for pat in exclude):
        return False
    return True


def print_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CMA Clinical Search Benchmark — Hugging Face Downloader                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Required:
  export HF_TOKEN=hf_...

Examples:
  python src/data/download_hf.py --dataset mimiciii --out data/raw/huggingface
  python src/data/download_hf.py --dataset mimiciv --out data/raw/huggingface
  python src/data/download_hf.py --dataset mimiciv-cxr --out data/raw/huggingface
  python src/data/download_hf.py --dataset eicu --out data/raw/huggingface
  python src/data/download_hf.py --dataset all --out data/raw/huggingface

Inspect files without downloading:
  python src/data/download_hf.py --dataset mimiciii --inspect
""")


def inspect_dataset(name: str, spec: dict, token: str) -> None:
    print(f"\n=== Inspecting: {name} ({spec['id']}) ===")
    print(f"  {spec['note']}")
    try:
        files = list_repo_files(spec["id"], repo_type=spec["repo_type"], token=token)
    except Exception as exc:
        print(f"  Could not list files: {exc}", file=sys.stderr)
        return

    for fname in files:
        print(f"    {fname}")
    print(f"  Files listed: {len(files)}")


def download_dataset(name: str, spec: dict, out_dir: Path, token: str,
                     include: Optional[list[str]] = None,
                     exclude: Optional[list[str]] = None,
                     resume: bool = True) -> Optional[list[str]]:
    target_dir = out_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    include = include or spec["default_include"]
    exclude = exclude or spec["default_exclude"]

    print(f"\n=== Downloading: {name} ({spec['id']}) ===")
    print(f"  Output directory: {target_dir}")
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
                           include: Optional[list[str]] = None,
                           exclude: Optional[list[str]] = None,
                           resume: bool = True) -> list[Path]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MIMIC datasets from Hugging Face")
    parser.add_argument(
        "--dataset",
        choices=["mimiciii", "mimiciv", "mimiciv-cxr", "eicu", "all"],
        default=None,
        help="Which dataset to download.",
    )
    parser.add_argument("--out", default="data/raw/huggingface", help="Output directory.")
    parser.add_argument("--inspect", action="store_true", help="List repository files without downloading.")
    parser.add_argument("--file-by-file", action="store_true", help="Download files one-by-one.")
    parser.add_argument("--include", nargs="+", default=None, help="Glob patterns of files to include.")
    parser.add_argument("--exclude", nargs="+", default=None, help="Glob patterns of files to skip.")
    parser.add_argument("--instructions", action="store_true", help="Print usage instructions and exit.")
    args = parser.parse_args()

    if args.instructions or args.dataset is None:
        print_instructions()
        return

    token = get_token()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(DATASETS.items()) if args.dataset == "all" else [(args.dataset, DATASETS[args.dataset])]

    for name, spec in selected:
        if args.inspect:
            inspect_dataset(name, spec, token)
        elif args.file_by_file:
            download_single_files(name, spec, out_dir, token, include=args.include, exclude=args.exclude)
        else:
            download_dataset(name, spec, out_dir, token, include=args.include, exclude=args.exclude)

    print(f"\nOutputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
