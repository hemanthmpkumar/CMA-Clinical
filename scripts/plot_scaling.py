#!/usr/bin/env python3
"""
scripts/plot_scaling.py

Generate the manuscript figures for each vignette scale produced by
`scripts/run_scaling_study.py`.

For every scale N (default 1 10 100 1000 10000) it loads:
    outputs/scaling/<N>/results.csv       - per-session benchmark rows
    outputs/scaling/<N>/statistics.json   - statistical analysis (analyze.py)

and renders the plots from src/viz/plots.py into:
    outputs/scaling/<N>/figures/

Usage:
  python scripts/plot_scaling.py
  python scripts/plot_scaling.py --scales 100 1000 10000
  python scripts/plot_scaling.py --out-root outputs/scaling
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz.plots import (  # noqa: E402
    ensure_dirs,
    plot_accuracy_comparison,
    plot_cma_components,
    plot_consort_flow,
    plot_intent_trajectory,
    plot_latency,
    plot_subgroup_forest,
    plot_time_distribution,
    plot_tlx,
)

DEFAULT_SCALES = [1, 10, 100, 1000, 10000]


def plot_scale(scale: int, out_root: Path):
    scale_dir = out_root / str(scale)
    results_csv = scale_dir / "results.csv"
    stats_json = scale_dir / "statistics.json"
    if not results_csv.exists() or not stats_json.exists():
        raise FileNotFoundError(
            f"Missing results for scale {scale}: need {results_csv} and {stats_json}. "
            f"Run: python scripts/run_scaling_study.py --scales {scale}"
        )

    df = pd.read_csv(results_csv)
    stats = json.loads(stats_json.read_text(encoding="utf-8"))

    out_dir = scale_dir / "figures"
    ensure_dirs(out_dir)

    n_sessions = int(stats.get("n_sessions", len(df)))
    n_vignettes = int(stats.get("n_vignettes", df["vignette_id"].nunique()))
    print(f"[{scale}] {n_vignettes} vignettes / {n_sessions} sessions "
          f"({n_sessions // 3} per condition) -> {out_dir}")

    plot_consort_flow(df, out_dir)
    plot_time_distribution(df, out_dir)
    plot_accuracy_comparison(df, out_dir)
    plot_intent_trajectory(out_dir)
    plot_tlx(df, out_dir)
    plot_latency(df, out_dir)
    plot_subgroup_forest(stats, out_dir)
    plot_cma_components(out_dir)

    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Generate figures for each scaling-study scale")
    ap.add_argument("--scales", type=int, nargs="*", default=None,
                    help=f"Vignette scales to plot (default: {' '.join(map(str, DEFAULT_SCALES))})")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs/scaling",
                    help="Directory containing outputs/scaling/<N>/ scale results.")
    args = ap.parse_args()

    scales = sorted(args.scales) if args.scales else DEFAULT_SCALES

    for n in scales:
        out_dir = plot_scale(n, args.out_root)
        print(f"  figures:")
        for p in sorted(out_dir.glob("*.png")):
            print(f"    {p.name}")

    print("\nScaling plots complete.")


if __name__ == "__main__":
    main()
