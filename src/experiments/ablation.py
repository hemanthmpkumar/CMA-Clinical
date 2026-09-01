"""
src/experiments/ablation.py

Multi-variant ablation study runner.

The ablation target is **GDT** (the primary intervention), which shares CMA's
architecture (SPD log-Euclidean encoder + GSI gate + JEPA predictor) but with
its own tuned hyper-parameters. Each GDT variant is evaluated under the full
four-arm crossover (TF-IDF control, BM25, CMA, GDT) so the results stay
directly comparable to the main experiment and comparison figures.

Variants (gate / prefetch switches on GDT):
  - full_gdt : GSI gate ON, JEPA prefetch ON  (tuned hyper-parameters)
  - gsi_only : GSI gate ON, JEPA prefetch OFF (prefetch_weight=0)
  - jepa_only: GSI gate OFF (curvature_threshold=inf), JEPA prefetch ON

Two modes:
  1. From-scratch (default): train baseline, BM25, CMA, GDT retrievers from
     the corpus + vignettes, then run each ablation variant.
  2. Trained-model mode (--models-dir): load pickled baseline/bm25/cma/gdt
     retrievers trained elsewhere (e.g. per-scale models from the scaling
     study) and run the ablation variants with those exact components. Keeps
     the ablation consistent with the running experiments.

Usage:
    python src/experiments/ablation.py [--processed-dir data/processed]
                                       [--top-k 10] [--out-dir outputs/ablation]
    python src/experiments/ablation.py --models-dir models --out-dir outputs/ablation
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

import joblib
import pandas as pd

from src.data.prepare import load_corpus_and_vignettes
from src.experiments.simulate_users import run_experiment
from src.models.baseline import BaselineRetriever
from src.models.bm25 import BM25Retriever
from src.models.cma import CMARetriever
from src.models.gdt import GDTRetriever

# The four benchmark arms, in figure order.
ALL_CONDITIONS = ["control", "bm25", "cma", "gdt"]
ARM_LABELS = {
    "control": "TF-IDF",
    "bm25": "BM25",
    "cma": "CMA",
    "gdt": "GDT",
}
ARM_COLORS = {
    "control": "#E74C3C",
    "bm25": "#F39C12",
    "cma": "#27AE60",
    "gdt": "#8E44AD",
}

# Ablation variants of the GDT target. `overrides` are passed to
# GDTRetriever.copy_with_hyperparams; the tuned defaults (curvature_threshold,
# gate_temperature, gate_lexical_include, prefetch_weight, etc.) come from the
# loaded/fitted model.
GDT_ABLATION_CONFIGS = OrderedDict({
    "full_gdt": {
        "label": "Full GDT",
        "description": "GSI gate + JEPA prefetch (tuned)",
        "overrides": {},
    },
    "gsi_only": {
        "label": "GSI only",
        "description": "gate on, prefetch off",
        "overrides": {"prefetch_weight": 0.0},
    },
    "jepa_only": {
        "label": "JEPA only",
        "description": "gate off (threshold=inf), prefetch on",
        "overrides": {"curvature_threshold": float("inf")},
    },
})

BASELINE_CONFIG = {"label": "Baseline (control)", "is_baseline": True}


def build_gdt_ablation(gdt_full: GDTRetriever, config: dict) -> GDTRetriever:
    return gdt_full.copy_with_hyperparams(**config["overrides"])


def run_ablation(corpus: list[dict], vignettes: list[dict],
                 seed: int = 20260617, top_k: int = 10,
                 baseline=None, bm25=None, cma_full=None,
                 gdt=None) -> pd.DataFrame:
    print("=" * 60)
    print("Ablation Study (four-arm: TF-IDF vs BM25 vs CMA vs GDT)")
    print("Ablation target: GDT (primary intervention)")
    print("=" * 60)

    if baseline is None:
        print("\nTraining baseline retriever...")
        baseline = BaselineRetriever(corpus)

    if bm25 is None:
        print("\nTraining BM25 retriever...")
        bm25 = BM25Retriever(corpus)

    if cma_full is None:
        print("\nTraining full CMA retriever (benchmark)...")
        cma_full = CMARetriever(corpus)
        cma_full.fit_predictor(vignettes, epochs=120, batch_size=64)

    if gdt is None:
        print("\nTraining GDT retriever (SPD encoder + JEPA predictor)...")
        gdt = GDTRetriever(corpus)
        gdt.fit_predictor(vignettes, epochs=120, batch_size=64)

    all_rows = []
    n_vignettes = len(vignettes)

    # Run each GDT ablation variant. Every variant evaluates the full four-arm
    # crossover so the ablated GDT remains comparable to all benchmarks.
    for key, config in GDT_ABLATION_CONFIGS.items():
        label = config["label"]
        print(f"\n{'─' * 50}")
        print(f"Ablation variant: {label}")
        print(f"{'─' * 50}")
        print(f"  {config['description']}")
        overrides = config.get("overrides", {})
        for param, val in overrides.items():
            print(f"  {param}={val}")

        if key == "full_gdt":
            gdt_variant = gdt
        else:
            gdt_variant = build_gdt_ablation(gdt, config)

        variant_seed = seed + hash(key) % 10000
        df = run_experiment(baseline, bm25, cma_full, vignettes,
                            seed=variant_seed, top_k=top_k,
                            gdt_retriever=gdt_variant)
        df["ablation_variant"] = label
        all_rows.append(df)

    results = pd.concat(all_rows, ignore_index=True)
    return results


def ablation_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by ablation variant and condition (all four arms)."""
    rows = []
    for variant in results["ablation_variant"].unique():
        sub = results[results["ablation_variant"] == variant]
        for cond in ALL_CONDITIONS:
            cond_sub = sub[sub["condition"] == cond]
            rows.append({
                "variant": variant,
                "condition": cond,
                "arm": ARM_LABELS.get(cond, cond),
                "n_sessions": len(cond_sub),
                "mean_time_to_info": round(cond_sub["time_to_info"].mean(), 2),
                "median_time_to_info": round(cond_sub["time_to_info"].median(), 2),
                "accuracy": round(cond_sub["accuracy"].mean(), 3),
                "mean_cognitive_load": round(cond_sub["cognitive_load"].mean(), 2),
                "mean_latency_ms": round(cond_sub["latency_ms"].mean(), 2),
                "mean_queries": round(cond_sub["n_queries_issued"].mean(), 2),
            })

    summary_df = pd.DataFrame(rows)

    # Add pct change vs TF-IDF control per arm within each variant.
    pct_rows = []
    for variant in results["ablation_variant"].unique():
        sub = results[results["ablation_variant"] == variant]
        ctrl = sub[sub["condition"] == "control"]
        if ctrl.empty:
            continue
        for cond in ALL_CONDITIONS:
            if cond == "control":
                continue
            arm_sub = sub[sub["condition"] == cond]
            if arm_sub.empty:
                continue
            for metric in ["time_to_info", "accuracy", "cognitive_load", "latency_ms"]:
                cv = ctrl[metric].mean()
                mv = arm_sub[metric].mean()
                pct = ((mv - cv) / cv * 100) if cv != 0 else 0.0
                pct_rows.append({
                    "variant": variant,
                    "condition": cond,
                    "arm": ARM_LABELS.get(cond, cond),
                    "metric": metric,
                    "control_mean": round(cv, 3),
                    f"{cond}_mean": round(mv, 3),
                    "pct_change": round(pct, 2),
                })

    change_df = pd.DataFrame(pct_rows)
    return summary_df, change_df


def main():
    parser = argparse.ArgumentParser(description="Run GDT ablation study")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default=None,
                        help="Load trained baseline/bm25/cma/gdt.pkl from "
                             "this directory instead of training from scratch, "
                             "so the ablation matches the main experiment's "
                             "components.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--out-dir", default="outputs/ablation")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models_dir:
        models_dir = Path(args.models_dir)
        print(f"Loading trained retrievers from {models_dir}...")
        baseline = joblib.load(models_dir / "baseline.pkl")
        bm25 = joblib.load(models_dir / "bm25.pkl")
        cma_full = joblib.load(models_dir / "cma.pkl")
        gdt = joblib.load(models_dir / "gdt.pkl")
        vignettes = json.loads(
            (processed_dir / "vignettes.json").read_text(encoding="utf-8")
        )
        print(f"  Loaded baseline/bm25/cma/gdt + {len(vignettes)} vignettes")
    else:
        print("Loading processed data...")
        corpus, vignettes = load_corpus_and_vignettes(processed_dir)
        print(f"  Corpus records: {len(corpus)}")
        print(f"  Vignettes: {len(vignettes)}")
        baseline = bm25 = cma_full = gdt = None

    results = run_ablation(
        corpus if args.models_dir is None else [],
        vignettes, seed=args.seed, top_k=args.top_k,
        baseline=baseline, bm25=bm25, cma_full=cma_full,
        gdt=gdt,
    )

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

    print("\nPercentage change (each arm vs control within each variant):")
    print(change_df.to_string(index=False))

    # Write a human-readable report.
    report_path = out_dir / "ablation_report.json"
    report = {
        "configs": {k: dict(v) for k, v in GDT_ABLATION_CONFIGS.items()},
        "arms": [{"condition": c, "label": ARM_LABELS[c]} for c in ALL_CONDITIONS],
        "n_vignettes": len(vignettes),
        "n_sessions": len(results),
        "summary": summary_df.to_dict(orient="records"),
        "pct_change": change_df.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
