#!/usr/bin/env python3
"""
src/data/split.py

Split vignettes into train / validation / test sets using patient-level
stratification. Patient-level splitting prevents note-level leakage across splits.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Split vignettes into train/val/test")
    parser.add_argument("--vignettes", default="data/processed/vignettes.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()

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
