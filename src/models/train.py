#!/usr/bin/env python3
"""
src/models/train.py

Train (fit + tune) the baseline, BM25 and CMA retrievers.

Training here means:
  1. Splitting vignettes into train / validation / test sets.
  2. Building the sparse retrieval indexes (TF-IDF baseline, BM25) on the full corpus.
  3. Grid-searching retriever hyper-parameters on the validation set.
  4. Pickling the best retriever objects to `models/`.

Usage:
  python src/models/train.py --corpus data/processed/corpus.jsonl \
      --train data/processed/train_vignettes.json \
      --val data/processed/val_vignettes.json \
      --out-dir models
"""

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Suppress noisy BLAS/sklearn runtime warnings that are often spurious on macOS.
np.seterr(divide="ignore", invalid="ignore", over="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.experiments.simulate_users import simulate_session
from src.models.baseline import BaselineRetriever
from src.models.bm25 import BM25Retriever
from src.models.cma import CMARetriever


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n")]


def load_vignettes(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(retriever, vignettes: list[dict], condition: str, seed: int = 20260620):
    """Run simulate_session on a set of vignettes and return aggregate metrics."""
    times = []
    accs = []
    queries = []
    for idx, v in enumerate(vignettes):
        res = simulate_session(retriever, v, condition, seed + idx, top_k=10)
        times.append(res["time_to_info"])
        accs.append(res["accuracy"])
        queries.append(res["n_queries_issued"])
    return {
        "mean_time": float(np.mean(times)),
        "mean_acc": float(np.mean(accs)),
        "mean_queries": float(np.mean(queries)),
    }


def evaluate_retrieval(retriever, vignettes: list[dict], top_k: int = 50) -> dict:
    """Per-query partial-credit retrieval metrics for hyper-parameter tuning.

    Unlike the binary all-or-nothing vignette accuracy used in the final
    evaluation, these metrics give the tuner a smooth gradient even when no
    full vignette completes: a window that lifts the target from rank 11 to
    rank 9 (inside top-10) is rewarded, and one that moves it from rank 50 to
    rank 2 earns more MRR credit. This is what lets tuning actually separate
    window sizes that the strict metric collapses to 0.000.
    """
    hits_10 = 0
    hits_top = 0
    rr_sum = 0.0
    n_q = 0
    for v in vignettes:
        retriever.reset_session()
        session_history: list[str] = []
        for q in v["queries"]:
            target_id = q["target_note_id"]
            results = retriever.search(q["text"], session_history=session_history,
                                       top_k=top_k, prefetch=True)
            session_history.append(q["text"])
            ids = [note_id for note_id, _ in results]
            n_q += 1
            if target_id in ids:
                rank = ids.index(target_id) + 1
                hits_top += 1
                rr_sum += 1.0 / rank
                if rank <= 10:
                    hits_10 += 1
    if n_q == 0:
        return {"recall_10": 0.0, "recall_50": 0.0, "mrr": 0.0, "n_queries": 0}
    return {
        "recall_10": hits_10 / n_q,
        "recall_50": hits_top / n_q,
        "mrr": rr_sum / n_q,
        "n_queries": n_q,
    }


def tune_baseline(corpus: list[dict], train_v: list, val_v: list):
    """Tune the TF-IDF baseline retriever's context window."""
    print("\nTuning baseline retriever...")
    best = None
    best_score = -1e9
    for window in [3, 5, 10, 20, 50]:
        retriever = BaselineRetriever(corpus, window_size=window)
        metrics = evaluate_retrieval(retriever, val_v)
        # Partial-credit retrieval score: prefer more targets inside top-10,
        # with MRR as a tie-breaker for ranking quality within the window.
        score = metrics["recall_10"] * 100 + metrics["mrr"]
        print(f"  window={window:2d} -> recall@10={metrics['recall_10']:.3f}, "
              f"recall@50={metrics['recall_50']:.3f}, mrr={metrics['mrr']:.3f}, "
              f"score={score:.2f}")
        if score > best_score:
            best_score = score
            best = (window, metrics)
    window, _ = best
    print(f"Best baseline: window_size={window}")
    return BaselineRetriever(corpus, window_size=window), {"window_size": window, "val_score": float(best_score)}


def tune_bm25(corpus: list[dict], train_v: list, val_v: list, baseline_retriever): ### Changes (Aniruddha)
    """Tune the BM25 retriever's context window (k1=1.5, b=0.75 fixed)."""
    print("\nTuning BM25 retriever...")
    best = None
    best_score = -1e9
    tune_train = train_v[: min(len(train_v), 30)]
    for window in [3, 5, 10, 20, 50]:
        retriever = BM25Retriever(corpus, window_size=window)
        #baseline_for_compare = evaluate(retriever, tune_train, "control")
        baseline_for_compare = evaluate(baseline_retriever, tune_train, "control") ### Changes (Aniruddha)
        bm25 = evaluate(retriever, tune_train, "cma")
        if baseline_for_compare["mean_time"] > 0:
            reduction = (baseline_for_compare["mean_time"] - bm25["mean_time"]) / baseline_for_compare["mean_time"]
        else:
            reduction = 0.0
        #score = reduction + bm25["mean_acc"] - 0.05 * bm25["mean_queries"]
        score = reduction + (bm25["mean_acc"] - baseline_for_compare["mean_acc"]) - 0.05 * bm25["mean_queries"]  ### Changes (Aniruddha)
        print(f"  window={window:2d} -> "
              f"reduction={reduction*100:5.1f}%, accuracy={bm25['mean_acc']:.3f}, score={score:.3f}")
        if score > best_score:
            best_score = score
            best = (window, bm25)
    window, _ = best
    print(f"Best BM25: window_size={window}")
    metrics = evaluate(BM25Retriever(corpus, window_size=window), val_v, "cma")
    print(f"Validation BM25 metrics: accuracy={metrics['mean_acc']:.3f}, mean_time={metrics['mean_time']:.1f}s")
    return BM25Retriever(corpus, window_size=window), {
        "window_size": window,
        "train_score": float(best_score),
        "val_accuracy": float(metrics["mean_acc"]),
        "val_mean_time": float(metrics["mean_time"]),
    }


#def tune_cma(corpus: list[dict], train_v: list, val_v: list):
def tune_cma(corpus: list[dict], train_v: list, val_v: list, baseline_retriever): ### Changes (Aniruddha)
    """Tune CMA retriever hyper-parameters on the validation set."""
    print("\nTuning CMA retriever...")
    # Small but informative grid for CPU-friendly tuning on the synthetic benchmark.
    configs = list(itertools.product(
        [0.55, 0.75, 0.95],           # curvature_threshold
        [0.0, 0.1, 0.3],              # gate_discount
        [0.3, 0.6],                   # prefetch_weight
        [3, 5],                       # context_window
    ))
    best = None
    best_score = -1e9
    # Use a smaller training subsample to keep grid search fast.
    tune_train = train_v[: min(len(train_v), 30)]

    # Build the TF-IDF/SVD projection and train the JEPA predictor once.
    # Hyper-parameter search only changes the gate/search behavior, not the
    # learned latent dynamics, so the expensive components can be shared.
    print("  Fitting CMA latent projection and JEPA predictor on training vignettes...")
    base_retriever = CMARetriever(corpus)
    base_retriever.fit_predictor(train_v, epochs=120, batch_size=64)

    for thr, disc, pre_w, ctx in configs:
        retriever = base_retriever.copy_with_hyperparams(
            curvature_threshold=thr,
            gate_discount=disc,
            prefetch_weight=pre_w,
            context_window=ctx,
        )
        #baseline_for_compare = evaluate(retriever, tune_train, "control")
        baseline_for_compare = evaluate(baseline_retriever, tune_train, "control") ### Changes (Aniruddha)
        cma = evaluate(retriever, tune_train, "cma")
        # Relative time reduction, plus accuracy reward.
        if baseline_for_compare["mean_time"] > 0:
            reduction = (baseline_for_compare["mean_time"] - cma["mean_time"]) / baseline_for_compare["mean_time"]
        else:
            reduction = 0.0
        #score = reduction + cma["mean_acc"] - 0.05 * cma["mean_queries"]
        score = reduction + (cma["mean_acc"] - baseline_for_compare["mean_acc"]) - 0.05 * cma["mean_queries"]  ### Changes (Aniruddha)  
        print(f"  thr={thr:.2f} disc={disc:.1f} pre={pre_w:.1f} ctx={ctx:2d} -> "
              f"reduction={reduction*100:5.1f}%, accuracy={cma['mean_acc']:.3f}, score={score:.3f}")
        if score > best_score:
            best_score = score
            best = (thr, disc, pre_w, ctx)

    thr, disc, pre_w, ctx = best
    print(f"\nBest CMA config: threshold={thr}, discount={disc}, prefetch={pre_w}, context_window={ctx}")

    # Return a retriever with the best hyper-parameters sharing the fitted components.
    retriever = base_retriever.copy_with_hyperparams(
        curvature_threshold=thr,
        gate_discount=disc,
        prefetch_weight=pre_w,
        context_window=ctx,
    )
    metrics = evaluate(retriever, val_v, "cma")
    print(f"Validation CMA metrics: accuracy={metrics['mean_acc']:.3f}, mean_time={metrics['mean_time']:.1f}s")
    return retriever, {
        "curvature_threshold": thr,
        "gate_discount": disc,
        "prefetch_weight": pre_w,
        "context_window": ctx,
        "train_score": float(best_score),
        "val_accuracy": float(metrics["mean_acc"]),
        "val_mean_time": float(metrics["mean_time"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline and CMA retrievers")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--train", default="data/processed/train_vignettes.json")
    parser.add_argument("--val", default="data/processed/val_vignettes.json")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus and vignettes...")
    corpus = load_jsonl(Path(args.corpus))
    train_v = load_vignettes(Path(args.train))
    val_v = load_vignettes(Path(args.val))
    print(f"  corpus: {len(corpus)} records, train vignettes: {len(train_v)}, val vignettes: {len(val_v)}")

    baseline, baseline_cfg = tune_baseline(corpus, train_v, val_v)
    # cma, cma_cfg = tune_cma(corpus, train_v, val_v)
    bm25, bm25_cfg = tune_bm25(corpus, train_v, val_v, baseline)  ### Changes (Aniruddha)
    cma, cma_cfg = tune_cma(corpus, train_v, val_v, baseline)  ### Changes (Aniruddha)

    joblib.dump(baseline, out_dir / "baseline.pkl")
    joblib.dump(bm25, out_dir / "bm25.pkl")
    joblib.dump(cma, out_dir / "cma.pkl")

    config = {
        "baseline": baseline_cfg,
        "bm25": bm25_cfg,
        "cma": cma_cfg,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nModels saved to {out_dir}:")
    print(f"  {out_dir / 'baseline.pkl'}")
    print(f"  {out_dir / 'bm25.pkl'}")
    print(f"  {out_dir / 'cma.pkl'}")
    print(f"  {out_dir / 'config.json'}")
    print("\nNext step: python src/experiments/run.py --use-trained")


if __name__ == "__main__":
    main()
