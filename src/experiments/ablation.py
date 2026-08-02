"""
src/experiments/ablation.py

Multi-variant ablation study runner.

Compares Full CMA, GSI-only, and JEPA-only configurations against the
session-based baseline across the full crossover simulation.

Usage:
    python src/experiments/ablation.py [--processed-dir data/processed]
                                       [--top-k 10] [--out-dir outputs/ablation]
"""

import argparse
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

np.seterr(divide="ignore", invalid="ignore", over="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd

from src.data.prepare import load_corpus_and_vignettes
from src.experiments.simulate_users import run_experiment
from src.models.baseline import BaselineRetriever
from src.models.bm25 import BM25Retriever
from src.models.cma import CMARetriever


ABLATION_CONFIGS = OrderedDict({
    "full_cma": {
        "label": "Full CMA",
        "curvature_threshold": 0.65,
        "gate_discount": 0.05,
        "prefetch_weight": 0.4,
    },
    "gsi_only": {
        "label": "GSI only",
        "curvature_threshold": 0.65,
        "gate_discount": 0.05,
        "prefetch_weight": 0.0,
    },
    "jepa_only": {
        "label": "JEPA only",
        "curvature_threshold": float("inf"),
        "gate_discount": 0.05,
        "prefetch_weight": 0.4,
    },
})

BASELINE_CONFIG = {"label": "Baseline (control)", "is_baseline": True}


def build_cma_ablation(cma_full: CMARetriever, config: dict) -> CMARetriever:
    return cma_full.copy_with_hyperparams(
        curvature_threshold=config["curvature_threshold"],
        gate_discount=config["gate_discount"],
        prefetch_weight=config["prefetch_weight"],
    )


def run_ablation(corpus: list[dict], vignettes: list[dict],
                 seed: int = 20260617, top_k: int = 10) -> pd.DataFrame:
    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)

    print("\nTraining baseline retriever...")
    baseline = BaselineRetriever(corpus)

    print("\nTraining BM25 retriever...")
    bm25 = BM25Retriever(corpus)

    print("\nTraining full CMA retriever (SPD encoder + JEPA predictor)...")
    cma_full = CMARetriever(corpus)
    cma_full.fit_predictor(vignettes, epochs=120, batch_size=64)

    all_rows = []
    n_vignettes = len(vignettes)

    # Run each ablation variant.
    for key, config in ABLATION_CONFIGS.items():
        label = config["label"]
        print(f"\n{'─' * 50}")
        print(f"Ablation variant: {label}")
        print(f"{'─' * 50}")
        print(f"  curvature_threshold={config['curvature_threshold']}, "
              f"gate_discount={config['gate_discount']}, "
              f"prefetch_weight={config['prefetch_weight']}")

        if key == "full_cma":
            cma_variant = cma_full
        else:
            cma_variant = build_cma_ablation(cma_full, config)

        variant_seed = seed + hash(key) % 10000
        df = run_experiment(baseline, bm25, cma_variant, vignettes,
                            seed=variant_seed, top_k=top_k)
        df["ablation_variant"] = label
        all_rows.append(df)

    results = pd.concat(all_rows, ignore_index=True)
    return results


def ablation_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by ablation variant and condition."""
    rows = []
    for variant in results["ablation_variant"].unique():
        sub = results[results["ablation_variant"] == variant]
        for cond in ["control", "cma"]:
            cond_sub = sub[sub["condition"] == cond]
            rows.append({
                "variant": variant,
                "condition": cond,
                "n_sessions": len(cond_sub),
                "mean_time_to_info": round(cond_sub["time_to_info"].mean(), 2),
                "median_time_to_info": round(cond_sub["time_to_info"].median(), 2),
                "accuracy": round(cond_sub["accuracy"].mean(), 3),
                "mean_cognitive_load": round(cond_sub["cognitive_load"].mean(), 2),
                "mean_latency_ms": round(cond_sub["latency_ms"].mean(), 2),
                "mean_queries": round(cond_sub["n_queries_issued"].mean(), 2),
            })

    summary_df = pd.DataFrame(rows)

    # Add pct change vs baseline per variant (control vs cma within variant).
    pct_rows = []
    for variant in results["ablation_variant"].unique():
        sub = results[results["ablation_variant"] == variant]
        ctrl = sub[sub["condition"] == "control"]
        cma_cond = sub[sub["condition"] == "cma"]
        if ctrl.empty or cma_cond.empty:
            continue
        for metric in ["time_to_info", "accuracy", "cognitive_load", "latency_ms"]:
            cv = ctrl[metric].mean()
            mv = cma_cond[metric].mean()
            pct = ((mv - cv) / cv * 100) if cv != 0 else 0.0
            pct_rows.append({
                "variant": variant,
                "metric": metric,
                "control_mean": round(cv, 3),
                "cma_mean": round(mv, 3),
                "pct_change": round(pct, 2),
            })

    change_df = pd.DataFrame(pct_rows)
    return summary_df, change_df


def main():
    parser = argparse.ArgumentParser(description="Run CMA ablation study")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--out-dir", default="outputs/ablation")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    corpus, vignettes = load_corpus_and_vignettes(processed_dir)
    print(f"  Corpus records: {len(corpus)}")
    print(f"  Vignettes: {len(vignettes)}")

    results = run_ablation(corpus, vignettes, seed=args.seed, top_k=args.top_k)

    raw_path = out_dir / "ablation_results.csv"
    results.to_csv(raw_path, index=False)
    print(f"\nSaved raw ablation results: {raw_path}")
    print(f"Total sessions: {len(results)}")

    summary_df, change_df = ablation_summary(results)
    summary_path = out_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    change_path = out_dir / "ablation_pct_change.csv"
    change_df.to_csv(change_path, index=False)
    print(f"Saved pct change: {change_path}")

    print("\nAblation summary:")
    print(summary_df.to_string(index=False))

    print("\nPercentage change (CMA vs control within each variant):")
    print(change_df.to_string(index=False))

    # Write a human-readable report.
    report_path = out_dir / "ablation_report.json"
    report = {
        "configs": {k: dict(v) for k, v in ABLATION_CONFIGS.items()},
        "n_vignettes": len(vignettes),
        "n_sessions": len(results),
        "summary": summary_df.to_dict(orient="records"),
        "pct_change": change_df.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
