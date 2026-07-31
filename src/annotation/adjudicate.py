import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

from .schemas import AdjudicationRecord, adjudication_csv_columns


DISPUTED_FIELDS = ["accuracy", "time_to_info", "cognitive_load"]


def _dc_to_dict(dc) -> dict:
    return {f.name: getattr(dc, f.name) for f in fields(dc)}


def _load_previous(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=adjudication_csv_columns())
        writer.writeheader()
        writer.writerows(rows)


def _get_contested_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (vignette_id, field) for rows where control != cma."""
    contested = []
    for vid in df["vignette_id"].unique():
        sub = df[df["vignette_id"] == vid]
        ctrl = sub[sub["condition"] == "control"]
        cma = sub[sub["condition"] == "cma"]
        if ctrl.empty or cma.empty:
            continue
        for field in DISPUTED_FIELDS:
            cv = ctrl[field].values[0]
            mv = cma[field].values[0]
            if cv != mv:
                contested.append((vid, field, cv, mv))
    return contested


def main():
    parser = argparse.ArgumentParser(
        description="Blinded clinical adjudication for contested CMA retrieval outputs"
    )
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--adjudicator", default="adjudicator_001")
    parser.add_argument("--out", default="outputs/annotations/adjudications.csv")
    parser.add_argument("--vignette", default=None,
                        help="Adjudicate a single vignette (default: all contested)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        sys.exit(1)

    df = pd.read_csv(results_path)
    previous = _load_previous(Path(args.out))
    adjudicated_keys = {(r["vignette_id"], r["disputed_field"]) for r in previous}

    contested = _get_contested_pairs(df)
    if not contested:
        print("No contested outputs found — control and CMA results agree on all fields.")
        return

    print(f"Found {len(contested)} contested output(s) across {DISPUTED_FIELDS}.")
    rows = list(previous)

    for vid, field, cv, mv in sorted(contested):
        if (vid, field) in adjudicated_keys:
            continue

        print(f"\n{'='*60}")
        print(f"Vignette: {vid}  |  Disputed field: {field}")
        print(f"{'='*60}")
        print(f"  Control value: {cv}")
        print(f"  CMA value:      {mv}")

        print("\nEnter adjudicated value (or press Enter to skip for now):")
        adjudicated = input(f"  [{cv} / {mv} / custom]: ").strip()
        if not adjudicated:
            print("  Skipped.")
            continue

        rationale = input("Rationale for this decision (optional): ").strip()

        rec = AdjudicationRecord(
            vignette_id=vid,
            condition="adjudicated",
            disputed_field=field,
            control_value=str(cv),
            cma_value=str(mv),
            adjudicated_value=adjudicated,
            adjudicator_id=args.adjudicator,
            rationale=rationale,
        )
        rows.append(_dc_to_dict(rec))
        print(f"  ✓ Adjudication recorded for {vid} / {field}")

    _save(rows, Path(args.out))
    print(f"\nAdjudication complete for '{args.adjudicator}'")
    print(f"  Saved: {args.out}")
    print(f"  Total adjudications: {len(rows)}")


if __name__ == "__main__":
    main()
