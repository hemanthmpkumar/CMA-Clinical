import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

from .schemas import (
    QueryAnnotation,
    SessionAnnotation,
    USEFULNESS_LABELS,
    TRUST_LABELS,
    SAFETY_LEVELS,
    query_annotation_csv_columns,
    session_annotation_csv_columns,
)


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _pick_option(prompt: str, options: list) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            choice = input(f"Enter number (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, IndexError):
            pass
        print(f"Invalid choice. Enter 1-{len(options)}.")


def _pick_likert(prompt: str, labels: dict) -> int:
    print(f"\n{prompt}")
    for val in sorted(labels):
        print(f"  {val}. {labels[val]}")
    while True:
        try:
            choice = input(f"Enter rating (1-{max(labels)}): ").strip()
            val = int(choice)
            if val in labels:
                return val
        except (ValueError, IndexError):
            pass
        print(f"Enter a number 1-{max(labels)}.")


def _dc_to_dict(dc) -> dict:
    return {f.name: getattr(dc, f.name) for f in fields(dc)}


def _load_vignette_metadata(vignettes_path: Path) -> dict:
    vignettes = json.loads(vignettes_path.read_text(encoding="utf-8"))
    return {v["vignette_id"]: v for v in vignettes}


def _prepare_retrieval_log(df: pd.DataFrame, vignette_id: str, condition: str,
                           corpus_path: Path) -> list[dict]:
    corpus = []
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                corpus.append(json.loads(line))
    note_map = {r["note_id"]: r for r in corpus}

    vignettes_path = corpus_path.parent / "vignettes.json"
    vmap = _load_vignette_metadata(vignettes_path)
    vmeta = vmap.get(vignette_id, {})
    queries = vmeta.get("queries", [])
    if not queries:
        return []

    q_log = []
    for qi, qdef in enumerate(queries):
        target_id = qdef.get("target_note_id", "unknown")
        target_text = note_map.get(target_id, {}).get("text", "")[:200] if target_id != "unknown" else ""
        q_log.append({
            "query_index": qi,
            "query_text": qdef["text"],
            "target_note_id": target_id,
            "target_text_preview": target_text,
        })
    return q_log


def _review_vignette(df: pd.DataFrame, vignette_id: str,
                     condition: str, annotator_id: str,
                     query_out: list, session_out: list,
                     corpus_path: Path):
    q_log = _prepare_retrieval_log(df, vignette_id, condition, corpus_path)
    if not q_log:
        print(f"  No query data for {vignette_id} / {condition}, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Vignette: {vignette_id}  |  Condition: {condition.upper()}")
    print(f"{'='*60}")

    row = df[(df["vignette_id"] == vignette_id) & (df["condition"] == condition)].iloc[0]
    print(f"  Time: {row['time_to_info']}s  |  Accuracy: {row['accuracy']}  |  "
          f"NASA-TLX: {row['cognitive_load']}")

    for entry in q_log:
        print(f"\n{'─'*40}")
        print(f"Query {entry['query_index']+1}: \"{entry['query_text']}\"")
        print(f"Target note: {entry['target_note_id']}")
        if entry["target_text_preview"]:
            print(f"  Preview: {entry['target_text_preview'][:120]}...")

        usefulness = _pick_likert("How useful was this query's retrieval result?",
                                  USEFULNESS_LABELS)
        safety = _pick_option("How would you rate the safety of the retrieved results?",
                              SAFETY_LEVELS)
        notes = input("Safety notes (optional, press Enter to skip): ").strip()

        ann = QueryAnnotation(
            vignette_id=vignette_id,
            condition=condition,
            query_index=entry["query_index"],
            query_text=entry["query_text"],
            retrieved_note_ids=[entry["target_note_id"]],
            usefulness=usefulness,
            safety=safety,
            safety_notes=notes,
            annotator_id=annotator_id,
        )
        query_out.append(_dc_to_dict(ann))

    trust = _pick_likert("Overall, how much do you trust this system's retrieval?",
                         TRUST_LABELS)
    feedback = input("Any additional feedback on workload or usability? ").strip()

    sess_ann = SessionAnnotation(
        vignette_id=vignette_id,
        condition=condition,
        trust=trust,
        workload_feedback=feedback,
        annotator_id=annotator_id,
    )
    session_out.append(_dc_to_dict(sess_ann))
    print(f"  ✓ Annotations saved for {vignette_id} / {condition}")


def _save_csv(rows: list[dict], path: Path, columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    parser = argparse.ArgumentParser(
        description="Clinical expert annotation review instrument for CMA retrieval evaluation"
    )
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--annotator", default="expert_001")
    parser.add_argument("--annotations-dir", default="outputs/annotations")
    parser.add_argument("--vignette", default=None,
                        help="Specific vignette ID (default: all unannotated)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run 'python src/experiments/run.py' first.")
        sys.exit(1)

    df = load_results(results_path)
    corpus_path = Path(args.processed_dir) / "corpus.jsonl"

    query_path = Path(args.annotations_dir) / "query_annotations.csv"
    session_path = Path(args.annotations_dir) / "session_annotations.csv"

    existing_qa = _load_csv(query_path)
    existing_sa = _load_csv(session_path)
    done = {(r["vignette_id"], r["condition"]) for r in existing_sa}

    vignettes_to_do = df["vignette_id"].unique()
    if args.vignette:
        vignettes_to_do = [v for v in vignettes_to_do if v == args.vignette]

    qa_rows = list(existing_qa)
    sa_rows = list(existing_sa)

    for vid in sorted(vignettes_to_do):
        for cond in ["control", "cma"]:
            if (vid, cond) not in done:
                _review_vignette(df, vid, cond, args.annotator,
                                 qa_rows, sa_rows, corpus_path)

    _save_csv(qa_rows, query_path, query_annotation_csv_columns())
    _save_csv(sa_rows, session_path, session_annotation_csv_columns())

    print(f"\n{'='*60}")
    print(f"Annotation session complete for annotator '{args.annotator}'")
    print(f"  Query annotations:  {query_path}")
    print(f"  Session annotations: {session_path}")
    print(f"  Total query annotations: {len(qa_rows)}")
    print(f"  Total session annotations: {len(sa_rows)}")


if __name__ == "__main__":
    main()
