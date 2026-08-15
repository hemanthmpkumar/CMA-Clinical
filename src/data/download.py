#!/usr/bin/env python3
"""
src/data/download.py

Download publicly available PhysioNet data for the CMA clinical search benchmark.

Supported datasets (all require credentialed PhysioNet access + DUA):
  • MIMIC-III / MIMIC-IV / MIMIC-IV + MIMIC-CXR
  • eICU Collaborative Research Database
  • HiRID (High Time Resolution ICU Dataset from Zurich)
  • Generic PhysioNet projects (by slug + file list)

Set PHYSIONET_USER and PHYSIONET_PASS as environment variables.

If you do not have credentialed access, use the built-in synthetic generator:
  python src/data/prepare.py --synthetic-only --n-patients 10000
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# MIMIC-III default structured + note tables.
MIMICIII_FILES = {
    "NOTEEVENTS.csv.gz": 3_100_000_000,
    "ADMISSIONS.csv.gz": 120_000_000,
    "PATIENTS.csv.gz": 5_000_000,
    "DIAGNOSES_ICD.csv.gz": 70_000_000,
    "PROCEDURES_ICD.csv.gz": 50_000_000,
    "PRESCRIPTIONS.csv.gz": 300_000_000,
    "LABEVENTS.csv.gz": 1_900_000_000,
}
MIMICIII_FULL_FILES = {
    "CHARTEVENTS.csv.gz": 32_000_000_000,
}

# MIMIC-IV default structured + note tables.
MIMICIV_FILES = {
    "hosp/admissions.csv.gz": 60_000_000,
    "hosp/patients.csv.gz": 3_000_000,
    "hosp/diagnoses_icd.csv.gz": 70_000_000,
    "hosp/procedures_icd.csv.gz": 50_000_000,
    "hosp/prescriptions.csv.gz": 250_000_000,
    "hosp/labevents.csv.gz": 1_500_000_000,
    "note/discharge.csv.gz": 900_000_000,
    "note/radiology.csv.gz": 300_000_000,
}

# MIMIC-CXR metadata + reports. The actual JPG/TXT reports live under files/.
MIMICCXR_FILES = {
    "mimic-cxr-2.0.0-metadata.csv.gz": 30_000_000,
    "mimic-cxr-2.0.0-chexpert.csv.gz": 30_000_000,
    "mimic-cxr-2.0.0-split.csv.gz": 2_000_000,
}

# eICU-CRD commonly used tables. Note: eICU has no large free-text notes table;
# text-like records are built from coded diagnoses, treatments, labs and vitals.
EICU_FILES = {
    "patient.csv.gz": 60_000_000,
    "admissionDx.csv.gz": 20_000_000,
    "diagnosis.csv.gz": 90_000_000,
    "treatment.csv.gz": 100_000_000,
    "lab.csv.gz": 1_200_000_000,
    "vitalPeriodic.csv.gz": 1_500_000_000,
    "vitalAperiodic.csv.gz": 300_000_000,
    "intakeOutput.csv.gz": 700_000_000,
    "infusionDrug.csv.gz": 300_000_000,
    "medication.csv.gz": 300_000_000,
    "note.csv": 0,  # optional; not all releases include it
}

# HiRID (parquet format, values are approximate compressed sizes).
HIRID_FILES = {
    "general_table.parquet": 20_000_000,
    "observation_tables/pharma_records.parquet": 500_000_000,
    "observation_tables/observations.parquet": 2_500_000_000,
}


def print_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CMA Clinical Search Benchmark — PhysioNet Data Download                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

All datasets below are hosted on PhysioNet and require credentialed access +
a signed Data Use Agreement.

1. Register / log in at https://physionet.org and complete the required CITI
   training ("Data or Specimens Only Research").

2. Request access for the projects you want to use:
      • mimic-iii (MIMIC-III)
      • mimiciv/2.2 (MIMIC-IV)
      • mimic-cxr-jpg/2.0.0 (MIMIC-CXR)
      • eicu-crd/2.0 (eICU Collaborative Research Database)
      • hirid/1.1.1 (HiRID)

3. Set credentials:
      export PHYSIONET_USER=your_user
      export PHYSIONET_PASS=your_pass

4. Download the data you need:

   MIMIC-III default tables (~5.6 GB compressed)
      python src/data/download.py --dataset mimiciii --out data/raw

   MIMIC-III + CHARTEVENTS (~38 GB compressed)
      python src/data/download.py --dataset mimiciii --full --out data/raw

   MIMIC-IV default tables (~3.1 GB compressed)
      python src/data/download.py --dataset mimiciv --out data/raw

   MIMIC-IV + MIMIC-CXR metadata and reports (~7 GB compressed + files/)
      python src/data/download.py --dataset mimiciv-cxr --out data/raw

   eICU-CRD (~5 GB compressed)
      python src/data/download.py --dataset eicu --out data/raw

   HiRID (~3 GB compressed, parquet)
      python src/data/download.py --dataset hirid --out data/raw

   Both MIMIC-III and MIMIC-IV (excludes CXR/eICU/HiRID)
      python src/data/download.py --dataset all --out data/raw

5. Do NOT commit downloaded files. data/raw/ is git-ignored by default.

6. Prepare the corpus after downloading:
      python src/data/prepare.py --huggingface-dir data/raw --hybrid --n-patients 10000
   or
      python src/data/prepare.py --eicu-dir data/raw --hybrid --n-patients 10000

7. No credentials? Use the built-in synthetic-only pipeline:
      python src/data/prepare.py --synthetic-only
""")


def _check_size(path: Path, expected: int):
    if path.exists() and expected > 0 and path.stat().st_size < expected * 0.5:
        print(f"Warning: {path.name} is much smaller than expected "
              f"({path.stat().st_size / 1e6:.1f} MB vs ~{expected / 1e9:.1f} GB).",
              file=sys.stderr)


def _download(url: str, target: Path, user: str, password: str) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    auth = f"{user}:{password}"
    cmd = [
        "curl", "-L", "-C", "-", "-u", auth, "-f",
        "--connect-timeout", "30", "--max-time", "3600",
        "-o", str(target), url
    ]
    # Show a command summary without exposing the password.
    print(f"  -> {target.name} ({url})")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Error downloading {target.name}: {exc}", file=sys.stderr)
        return False


def download_set(file_map: dict, base_url: str, out_dir: Path, user: str, password: str,
                 prefix_replacement: dict = None) -> list[Path]:
    """Download a set of files, resuming partial downloads."""
    downloaded = []
    for fname, expected_size in file_map.items():
        url = f"{base_url}{fname}?download"
        target_name = fname.replace("/", "_")
        target = out_dir / target_name
        if target.exists() and (expected_size == 0 or target.stat().st_size >= expected_size * 0.5):
            print(f"  -> {target.name} already present, skipping.")
            downloaded.append(target)
            continue
        if _download(url, target, user, password):
            _check_size(target, expected_size)
            downloaded.append(target)
    return downloaded


def _download_mimic_cxr_reports(out_dir: Path, user: str, password: str):
    """
    Recursively download MIMIC-CXR free-text reports under files/.
    This can take a long time and several GB; make sure you want the full set.
    """
    if not shutil.which("wget"):
        print("  -> wget is not installed; cannot recursively fetch MIMIC-CXR reports. "
              "Install wget or download the files/ tree manually.", file=sys.stderr)
        return
    base = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
    target_dir = out_dir / "mimic-cxr-reports"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("  -> Recursively downloading MIMIC-CXR text reports (this may take a while)...")
    cmd = [
        "wget", "--user", user, "--password", password,
        "--continue", "--recursive", "--no-parent", "--level=inf",
        "-nH", "--cut-dirs=4",
        "--accept=.txt,.csv,.csv.gz",
        "-P", str(target_dir), base
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error during recursive MIMIC-CXR download: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet data for CMA benchmark")
    parser.add_argument("--dataset",
                        choices=["mimiciii", "mimiciv", "mimiciv-cxr", "eicu", "hirid", "all"],
                        default=None,
                        help="Which dataset to download.")
    parser.add_argument("--full", action="store_true",
                        help="Also download very large tables such as CHARTEVENTS.")
    parser.add_argument("--include-cxr-reports", action="store_true",
                        help="Recursively download MIMIC-CXR text reports as well as metadata.")
    parser.add_argument("--out", default="data/raw", help="Output directory.")
    parser.add_argument("--instructions", action="store_true",
                        help="Print download instructions and exit.")
    args = parser.parse_args()

    if args.instructions or args.dataset is None:
        print_instructions()
        return

    user = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASS")

    if not user or not password:
        print("Error: PHYSIONET_USER and PHYSIONET_PASS must be set for automated download.",
              file=sys.stderr)
        print_instructions()
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("mimiciii", "all"):
        print("Downloading MIMIC-III default tables...")
        downloaded = download_set(
            MIMICIII_FILES,
            "https://physionet.org/files/mimiciii/1.4/",
            out_dir, user, password
        )
        print(f"  MIMIC-III default: {len(downloaded)}/{len(MIMICIII_FILES)} files")
        if args.full:
            print("Downloading MIMIC-III full tables (CHARTEVENTS)...")
            downloaded_full = download_set(
                MIMICIII_FULL_FILES,
                "https://physionet.org/files/mimiciii/1.4/",
                out_dir, user, password
            )
            print(f"  MIMIC-III full: {len(downloaded_full)}/{len(MIMICIII_FULL_FILES)} files")

    if args.dataset in ("mimiciv", "mimiciv-cxr", "all"):
        print("Downloading MIMIC-IV default tables...")
        downloaded = download_set(
            MIMICIV_FILES,
            "https://physionet.org/files/mimiciv/2.2/",
            out_dir, user, password
        )
        print(f"  MIMIC-IV: {len(downloaded)}/{len(MIMICIV_FILES)} files")

    if args.dataset == "mimiciv-cxr":
        print("Downloading MIMIC-CXR metadata...")
        downloaded = download_set(
            MIMICCXR_FILES,
            "https://physionet.org/files/mimic-cxr-jpg/2.0.0/",
            out_dir, user, password
        )
        print(f"  MIMIC-CXR metadata: {len(downloaded)}/{len(MIMICCXR_FILES)} files")
        if args.include_cxr_reports:
            _download_mimic_cxr_reports(out_dir, user, password)

    if args.dataset == "eicu":
        print("Downloading eICU-CRD tables...")
        downloaded = download_set(
            EICU_FILES,
            "https://physionet.org/files/eicu-crd/2.0/",
            out_dir, user, password
        )
        print(f"  eICU-CRD: {len(downloaded)}/{len(EICU_FILES)} files")

    if args.dataset == "hirid":
        print("Downloading HiRID tables...")
        downloaded = download_set(
            HIRID_FILES,
            "https://physionet.org/files/hirid/1.1.1/",
            out_dir, user, password
        )
        print(f"  HiRID: {len(downloaded)}/{len(HIRID_FILES)} files")

    print(f"\nDownload outputs written to {out_dir.resolve()}")
    print("Next step: python src/data/prepare.py --<dataset>-dir " + str(out_dir))


if __name__ == "__main__":
    main()
