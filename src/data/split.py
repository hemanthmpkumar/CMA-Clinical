#!/usr/bin/env python3
"""
src/data/split.py

Split vignettes into train / validation / test sets using patient-level
stratification. Patient-level splitting prevents note-level leakage across splits.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split


def group_patients(vignettes: list[dict]) -> dict[str, list[dict]]:
    by_patient = defaultdict(list)
    for v in vignettes:
        by_patient[v["patient_id"]].append(v)
    return by_patient


def patient_label(patient_id: str, groups: dict) -> str:
    """Create a stratification label from the first vignette of a patient."""
    v = groups[patient_id][0]
    return v["specialty"]


def split_vignettes(vignettes: list[dict], train_frac=0.70, val_frac=0.15,
                    seed: int = 20260618) -> tuple[list[dict], list[dict], list[dict]]:
    """Patient-level stratified split."""
    groups = group_patients(vignettes)
    patient_ids = list(groups.keys())
    labels = [patient_label(pid, groups) for pid in patient_ids]

    train_ids, temp_ids, _, temp_labels = train_test_split(
        patient_ids, labels, train_size=train_frac, random_state=seed,
        stratify=labels
    )
    # Split temp into val and test with relative fractions.
    relative_val = val_frac / (1 - train_frac)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=relative_val, random_state=seed,
        stratify=temp_labels
    )

    train_v = [v for pid in train_ids for v in groups[pid]]
    val_v = [v for pid in val_ids for v in groups[pid]]
    test_v = [v for pid in test_ids for v in groups[pid]]

    return train_v, val_v, test_v


def load_jsonl(path: Path) -> list[dict]:
    """Load corpus.jsonl lines into a list of record dicts."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_and_split_scale(corpus: list[dict], n: int, seed: int,
                             out_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate N vignettes from the corpus and write them plus splits.

    Mirrors the scaling-study pipeline (scripts/run_scaling_study.py); the
    session scales 1, 10, 100, 1{,}000, 10{,}000, 100{,}000 map to 3, 30, 300,
    3{,}000, 30{,}000, 300{,}000 vignettes (3 vignettes per session).
    """
    from src.data.prepare import generate_real_vignettes

    print(f"[{n}] Generating {n} vignettes from {len(corpus)} records (seed={seed})...")
    vignettes = generate_real_vignettes(corpus, n_vignettes=n, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vignettes.json").write_text(json.dumps(vignettes, indent=2), encoding="utf-8")

    try:
        train_v, val_v, test_v = split_vignettes(vignettes, seed=seed)
    except ValueError:
        # Fall back to a plain random split when stratification classes are
        # too sparse (e.g. tiny scales or capped smoke-test corpora).
        rng = random.Random(seed)
        shuffled = vignettes[:]
        rng.shuffle(shuffled)
        n_train = int(0.70 * len(shuffled))
        n_val = int(0.15 * len(shuffled))
        train_v, val_v, test_v = (
            shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]
        )
        print(f"[{n}] Stratified split unavailable; used plain random split.")
    (out_dir / "train_vignettes.json").write_text(json.dumps(train_v, indent=2), encoding="utf-8")
    (out_dir / "val_vignettes.json").write_text(json.dumps(val_v, indent=2), encoding="utf-8")
    (out_dir / "test_vignettes.json").write_text(json.dumps(test_v, indent=2), encoding="utf-8")
    print(f"[{n}] Wrote vignettes/splits to {out_dir} "
          f"(train={len(train_v)}, val={len(val_v)}, test={len(test_v)})")
    return train_v, val_v, test_v


def main():
    parser = argparse.ArgumentParser(description="Split vignettes into train/val/test")
    parser.add_argument("--vignettes", default="data/processed/vignettes.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--scales", type=int, nargs="*", default=None,
                        help="Generate N vignettes per scale and split them, e.g. "
                             "--scales 3 30 300 3000 30000 300000 (session scales "
                             "1, 10, 100, 1000, 10000, 100000 at 3 vignettes per "
                             "session).")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl",
                        help="Corpus used by --scales mode.")
    parser.add_argument("--data-root", default="data/scaling",
                        help="Root directory for per-scale outputs in --scales mode.")
    args = parser.parse_args()

    if args.scales:
        corpus = load_jsonl(Path(args.corpus))
        print(f"Loaded {len(corpus)} corpus records from {args.corpus}")
        data_root = Path(args.data_root)
        for n in sorted(args.scales):
            generate_and_split_scale(
                corpus, n, seed=args.seed + n, out_dir=data_root / str(n)
            )
        print(f"Saved per-scale vignettes/splits under {data_root} "
              f"(session scales = vignettes // 3):")
        for n in sorted(args.scales):
            print(f"  {n:6d} vignettes -> {n // 3:6d} sessions")
        return

    vignettes = json.loads(Path(args.vignettes).read_text(encoding="utf-8"))
    train_v, val_v, test_v = split_vignettes(
        vignettes, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_vignettes.json").write_text(json.dumps(train_v, indent=2), encoding="utf-8")
    (out_dir / "val_vignettes.json").write_text(json.dumps(val_v, indent=2), encoding="utf-8")
    (out_dir / "test_vignettes.json").write_text(json.dumps(test_v, indent=2), encoding="utf-8")

    print(f"Split {len(vignettes)} vignettes:")
    print(f"  train: {len(train_v)} vignettes from {len({v['patient_id'] for v in train_v})} patients")
    print(f"  val:   {len(val_v)} vignettes from {len({v['patient_id'] for v in val_v})} patients")
    print(f"  test:  {len(test_v)} vignettes from {len({v['patient_id'] for v in test_v})} patients")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
