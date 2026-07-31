#!/usr/bin/env python3
"""
src/experiments/run.py

End-to-end experiment entrypoint.

Two modes:
  1. From-scratch (default): build baseline and CMA retrievers from the raw
     corpus and evaluate on all vignettes.
  2. Trained-model mode (--use-trained): load pickled retrievers from models/
     and evaluate on the held-out test vignettes.

Usage:
  python src/experiments/run.py [--processed-dir data/processed] [--top-k 10]
  python src/experiments/run.py --use-trained [--models-dir models] [--top-k 10]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Suppress noisy BLAS/sklearn runtime warnings that are often spurious on macOS.
np.seterr(divide="ignore", invalid="ignore", over="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import joblib
import pandas as pd

from src.data.prepare import load_corpus_and_vignettes
from src.experiments.simulate_users import run_experiment
from src.models.baseline import BaselineRetriever
from src.models.cma import CMARetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMA clinical search experiment")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory containing corpus.jsonl and vignettes.json")
    parser.add_argument("--use-trained", action="store_true",
                        help="Load trained retrievers from --models-dir and evaluate on test set")
    parser.add_argument("--models-dir", default="models",
                        help="Directory containing baseline.pkl and cma.pkl")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of documents retrieved per query")
    parser.add_argument("--seed", type=int, default=20260617,
                        help="Random seed for crossover randomization")
    parser.add_argument("--out", default="outputs/results.csv",
                        help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)

    if args.use_trained:
        models_dir = Path(args.models_dir)
        print(f"Loading trained retrievers from {models_dir}...")
        baseline = joblib.load(models_dir / "baseline.pkl")
        cma = joblib.load(models_dir / "cma.pkl")
        test_path = processed_dir / "test_vignettes.json"
        if not test_path.exists():
            raise FileNotFoundError(
                f"{test_path} not found. Run src/data/split.py and src/models/train.py first."
            )
        vignettes = json.loads(test_path.read_text(encoding="utf-8"))
        print(f"  Loaded test vignettes: {len(vignettes)}")
    else:
        print("Loading processed data...")
        corpus, vignettes = load_corpus_and_vignettes(processed_dir)
        print(f"  Corpus records: {len(corpus)}")
        print(f"  Vignettes: {len(vignettes)}")

        print("Building retrievers...")
        baseline = BaselineRetriever(corpus)
        cma = CMARetriever(corpus)
        cma.fit_predictor(vignettes, epochs=120, batch_size=64)

    print("Running simulated crossover experiment...")
    df = run_experiment(baseline, cma, vignettes, seed=args.seed, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved raw results: {out_path}")
    print(f"Total simulated sessions: {len(df)}")
    print("\nNext step: python src/analysis/analyze.py")


if __name__ == "__main__":
    main()
