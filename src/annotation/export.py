import json
from pathlib import Path

import pandas as pd


def load_annotations(annotations_dir: Path) -> dict:
    """Load all annotation CSV files and return as dict of DataFrames."""
    result = {}
    for name in ["query_annotations", "session_annotations", "adjudications"]:
        path = annotations_dir / f"{name}.csv"
        if path.exists():
            result[name] = pd.read_csv(path)
        else:
            result[name] = pd.DataFrame()
    return result


def merge_with_results(results_df: pd.DataFrame,
                       annotations: dict) -> pd.DataFrame:
    """Merge session-level annotations into the results DataFrame."""
    df = results_df.copy()
    sa = annotations.get("session_annotations", pd.DataFrame())
    if not sa.empty:
        merge_cols = ["vignette_id", "condition"]
        for col in ["trust", "workload_feedback", "annotator_id", "annotation_id"]:
            if col in sa.columns:
                suffix = "" if col == "vignette_id" or col == "condition" else f"_{col}"
                df = df.merge(
                    sa[merge_cols + [col] if col not in merge_cols else merge_cols],
                    on=merge_cols, how="left", suffixes=("", suffix),
                )
    return df


def apply_adjudications(results_df: pd.DataFrame,
                        adjudications: pd.DataFrame) -> pd.DataFrame:
    """Overwrite disputed fields with adjudicated values."""
    df = results_df.copy()
    if adjudications.empty:
        return df

    for _, row in adjudications.iterrows():
        vid = row["vignette_id"]
        field = row["disputed_field"]
        adj_val = row["adjudicated_value"]
        mask = df["vignette_id"] == vid
        try:
            df.loc[mask, field] = float(adj_val)
        except (ValueError, TypeError):
            df.loc[mask, field] = adj_val
    return df


def summary(annotations: dict) -> dict:
    """Compute summary statistics from annotation data."""
    out = {}
    sa = annotations.get("session_annotations", pd.DataFrame())
    if not sa.empty and "trust" in sa.columns:
        grp = sa.groupby("condition")["trust"]
        out["trust"] = {
            "control_mean": float(grp.mean().get("control", 0)),
            "cma_mean": float(grp.mean().get("cma", 0)),
            "n_control": int((sa["condition"] == "control").sum()),
            "n_cma": int((sa["condition"] == "cma").sum()),
        }

    qa = annotations.get("query_annotations", pd.DataFrame())
    if not qa.empty:
        for metric in ["usefulness", "safety"]:
            if metric in qa.columns:
                grp = qa.groupby("condition")[metric]
                try:
                    vals = grp.value_counts(normalize=True)
                    out[f"{metric}_distribution"] = vals.to_dict()
                except Exception:
                    pass

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export and merge human annotations with experiment results"
    )
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--annotations-dir", default="outputs/annotations")
    parser.add_argument("--out", default="outputs/results_with_annotations.csv")
    parser.add_argument("--summary-out", default="outputs/annotation_summary.json")
    args = parser.parse_args()

    results = pd.read_csv(Path(args.results))
    annotations = load_annotations(Path(args.annotations_dir))
    merged = merge_with_results(results, annotations)
    merged = apply_adjudications(merged, annotations.get("adjudications", pd.DataFrame()))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Merged results with annotations: {out_path} ({len(merged)} rows)")

    summ = summary(annotations)
    summary_path = Path(args.summary_out)
    summary_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"Annotation summary: {summary_path}")
    print(f"  Trust ratings: {summ.get('trust', 'N/A')}")


if __name__ == "__main__":
    main()
