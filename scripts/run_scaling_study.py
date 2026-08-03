#!/usr/bin/env python3
"""
scripts/run_scaling_study.py

Scaling study for the three-arm crossover benchmark (CMA vs BM25 vs TF-IDF).

For each vignette scale N in {100, 1000, 10000, 100000}:
  1. Generate N chart-review vignettes from the corpus (target notes are
     query-boosted exactly like the original benchmark build).
  2. Write vignettes + patient-stratified splits to data/scaling/<N>/.
  3. Rebuild each retriever's corpus-derived index from the boosted corpus,
     REUSING the trained components saved in models/ (CMA neural encoder,
     JEPA predictor, TF-IDF vocabulary, tuned hyper-parameters). This keeps
     each scale's targets retrievable while avoiding re-training.
  4. Run the randomized three-period crossover experiment on all N vignettes
     (parallelised per condition) -> outputs/scaling/<N>/results.csv.
  5. Run the statistical analysis -> outputs/scaling/<N>/statistics.json and
     outputs/scaling/<N>/primary_results.csv.

Each scale is generated independently (seed = --seed + N).

Optional first step: run src/data/prepare.py to build the corpus from raw
Hugging Face data (--prepare). Off by default because re-ingesting the raw
snapshot is slow; the corpus is normally prepared once beforehand.

Usage:
  python scripts/run_scaling_study.py                                   # all four scales
  python scripts/run_scaling_study.py --prepare                         # ingest raw data first
  python scripts/run_scaling_study.py --scales 100 10000                # subset
  python scripts/run_scaling_study.py --only-scale 100 --limit 50       # quick smoke test
"""

import argparse
import gc
import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

np.seterr(divide="ignore", invalid="ignore", over="ignore")

CONDITIONS = ["control", "bm25", "cma"]
MODEL_FILES = {"control": "baseline.pkl", "bm25": "bm25.pkl", "cma": "cma.pkl"}
DEFAULT_SCALES = [100, 1000, 10000, 100000]

MODEL = None  # module-level handle shared with forked workers (copy-on-write)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _capped_corpus_path(corpus_path: Path, limit: int) -> Path:
    """Write a small prefix of the corpus to a temp file for smoke testing."""
    target = 40 * limit  # enough patients with >=8 notes for `limit` vignettes
    cap_path = corpus_path.parent / f"_smoke_{limit}.jsonl"
    with corpus_path.open("r", encoding="utf-8") as src, \
         cap_path.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= target:
                break
            dst.write(line)
    print(f"Smoke-test corpus capped to {target} records -> {cap_path}")
    return cap_path


def generate_scale(corpus_path: Path, n: int, seed: int, out_dir: Path) -> list[dict]:
    """Load the corpus, generate N boosted vignettes, write them + splits."""
    from src.data.prepare import generate_vignettes
    from src.data.split import split_vignettes

    print(f"\n[{n}] Generating {n} vignettes (seed={seed})...")
    corpus = _load_jsonl(corpus_path)
    vignettes = generate_vignettes(corpus, n_vignettes=n, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vignettes.json").write_text(json.dumps(vignettes, indent=2), encoding="utf-8")

    try:
        train_v, val_v, test_v = split_vignettes(vignettes, seed=seed)
    except ValueError:
        # Fall back to a plain random split when stratification classes are
        # too sparse (e.g. tiny capped smoke-test corpora).
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
    return corpus  # mutated with query boosts -> used to build each scale's index


def build_retriever(condition: str, corpus: list[dict], models_dir: Path):
    """Build a single retriever from the boosted corpus to save memory."""
    from src.models.baseline import BaselineRetriever
    from src.models.bm25 import BM25Retriever
    from src.models.cma import CMARetriever

    config = json.loads((models_dir / "config.json").read_text(encoding="utf-8"))
    
    t0 = time.time()
    if condition == "control":
        baseline_cfg = config.get("baseline", {})
        print(f"  building TF-IDF baseline (window={baseline_cfg.get('window_size', 3)})...")
        model = BaselineRetriever(corpus, window_size=baseline_cfg.get("window_size", 3))
        print(f"    baseline index built in {time.time() - t0:.0f}s")
        return model

    elif condition == "bm25":
        print("  building BM25 (window=3)...")
        model = BM25Retriever(corpus, window_size=3)
        print(f"    bm25 index built in {time.time() - t0:.0f}s")
        return model

    elif condition == "cma":
        cma_cfg = config.get("cma", {})
        print("  building CMA (reusing trained encoder/predictor + tuned hyper-params)...")
        cma_p = joblib_load(models_dir / MODEL_FILES["cma"])
        model = CMARetriever(
            corpus,
            curvature_threshold=cma_cfg.get("curvature_threshold", 0.65),
            gate_discount=cma_cfg.get("gate_discount", 0.05),
            context_window=cma_cfg.get("context_window", 5),
            prefetch_weight=cma_cfg.get("prefetch_weight", 0.4),
            vectorizer=cma_p.vectorizer,
            encoder=cma_p.encoder,
            predictor=cma_p.predictor,
            encoder_pretrain_epochs=0,
            encoder_finetune_epochs=0,
        )
        del cma_p
        gc.collect()
        print(f"    cma index built in {time.time() - t0:.0f}s")
        return model


def joblib_load(path: Path):
    import joblib
    return joblib.load(path)


def _eval_worker(task):
    """Run one condition-period of one vignette. MODEL is inherited via fork."""
    from src.experiments.simulate_users import simulate_session

    vignette, condition, period, run_seed, top_k = task
    row = simulate_session(MODEL, vignette, condition, run_seed, top_k=top_k)
    row["period"] = period
    row["specialty"] = vignette["specialty"]
    row["experience_group"] = vignette["experience_group"]
    row["complexity"] = vignette["complexity"]
    row["n_pivots"] = len(vignette["pivots"])
    return row


def evaluate_condition(condition: str, model, vignettes: list[dict],
                       orders: list[dict], seed: int, top_k: int,
                       workers: int) -> list[dict]:
    """Run one arm of the crossover for all vignettes, parallelised by fork."""
    global MODEL
    MODEL = model
    tasks = []
    for idx, v in enumerate(vignettes):
        period = orders[idx][condition]
        run_seed = seed + idx * 1000 + period
        tasks.append((v, condition, period, run_seed, top_k))

    rows = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            for i, row in enumerate(pool.imap(_eval_worker, tasks, chunksize=8)):
                if (i + 1) % 1000 == 0:
                    print(f"    {condition}: {i + 1}/{len(tasks)} sessions", flush=True)
                rows.append(row)
    else:
        for i, task in enumerate(tasks):
            if (i + 1) % 1000 == 0:
                print(f"    {condition}: {i + 1}/{len(tasks)} sessions", flush=True)
            rows.append(_eval_worker(task))
    return rows


def run_experiment_parallel(corpus: list[dict], models_dir: Path, vignettes: list[dict], 
                            seed: int, top_k: int, workers: int) -> pd.DataFrame:
    """Reproduce run_experiment's randomized three-period crossover layout."""
    rng = random.Random(seed)
    orders = []
    for _ in vignettes:
        order = rng.sample(CONDITIONS, k=3)
        orders.append({cond: period for period, cond in enumerate(order, start=1)})

    all_rows = []
    for condition in CONDITIONS:
        print(f"  evaluating condition '{condition}' over {len(vignettes)} vignettes...")
        
        # 1. Build ONLY the model for the current condition to save RAM
        model = build_retriever(condition, corpus, models_dir)
        
        # 2. Evaluate
        t0 = time.time()
        rows = evaluate_condition(condition, model, vignettes,
                                  orders, seed, top_k, workers)
        print(f"    {condition} done in {time.time() - t0:.0f}s ({len(rows)} rows)")
        all_rows.extend(rows)
        
        # 3. Clean up before the next model builds its index
        global MODEL
        MODEL = None
        del model
        gc.collect()

    df = pd.DataFrame(all_rows)
    df["condition_code"] = (df["condition"] == "cma").astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description="Run CMA/BM25/TF-IDF scaling study")
    ap.add_argument("--scales", type=int, nargs="*", default=None,
                    help="Vignette scales to run (default: 100 1000 10000 100000)")
    ap.add_argument("--only-scale", type=int, default=None,
                    help="Run a single scale (overrides --scales).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap vignettes used at each scale (smoke-testing only).")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/scaling_prepared/corpus.jsonl")
   # ap.add_argument("--corpus", type=Path, default=ROOT / "data/processed/corpus.jsonl")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--prepare", action="store_true",
                    help="Run src/data/prepare.py to build the corpus from raw data first.")
    ap.add_argument("--raw-hf-dir", type=Path, default=ROOT / "data/raw/huggingface",
                    help="Hugging Face raw snapshot dir used by --prepare.")
    ap.add_argument("--prepare-out", type=Path, default=ROOT / "data/scaling_prepared",
                    help="Where --prepare writes the fresh corpus.jsonl.")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/scaling")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs/scaling")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--workers", type=int, default=min(mp.cpu_count(), 14))
    ap.add_argument("--skip-analysis", action="store_true")
    args = ap.parse_args()

    if args.only_scale:
        scales = [args.only_scale]
    elif args.scales:
        scales = sorted(args.scales)
    else:
        scales = DEFAULT_SCALES

    mp.set_start_method("fork", force=True)

    corpus_path = args.corpus
    if args.prepare:
        prepare_out = args.prepare_out
        prepare_out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "src/data/prepare.py"),
               "--huggingface-dir", str(args.raw_hf_dir),
               "--n-vignettes", "0",
               "--out-dir", str(prepare_out)]
        print(f"Running data preparation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)
        corpus_path = prepare_out / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"{corpus_path} not created by prepare.py")
        print(f"Prepared corpus: {corpus_path}")

    print(f"Scales: {scales}")
    print(f"Workers: {args.workers} | top_k: {args.top_k} | seed: {args.seed}")
    print(f"Corpus: {corpus_path}")

    for n in scales:
        scale_dir = args.data_root / str(n)
        out_dir = args.out_root / str(n)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.limit:
            # Smoke-test mode: build everything from a small capped corpus so
            # the index build and generation both stay fast and self-consistent
            # (targets are boosted inside the capped corpus too).
            capped = _capped_corpus_path(corpus_path, args.limit)
            corpus = generate_scale(capped, n, args.seed + n, scale_dir)
        else:
            corpus = generate_scale(corpus_path, n, args.seed + n, scale_dir)

        vignettes = _load_json(scale_dir / "vignettes.json")
        if args.limit:
            vignettes = vignettes[: args.limit]

        df = run_experiment_parallel(corpus, args.models_dir, vignettes, seed=args.seed,
                                     top_k=args.top_k, workers=args.workers)
        results_path = out_dir / "results.csv"
        df.to_csv(results_path, index=False)
        print(f"[{n}] Saved results: {results_path} ({len(df)} rows)")

        del df, corpus
        gc.collect()

        if not args.skip_analysis:
            print(f"[{n}] Running statistical analysis...")
            subprocess.run(
                [sys.executable, str(ROOT / "src/analysis/analyze.py"),
                 "--results", str(results_path), "--out-dir", str(out_dir)],
                check=True, cwd=ROOT,
            )

    print("\nScaling study complete.")
    print(f"Per-scale artifacts under {args.data_root} and {args.out_root}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()