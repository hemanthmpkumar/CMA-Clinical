#!/usr/bin/env python3
"""
scripts/run_scaling_study.py

Scaling study for the four-arm crossover benchmark
(GDT vs CMA vs BM25 vs TF-IDF).

Scales are specified as SESSIONS. Each session scale S is paired with
VIGNETTES_PER_SESSION v vignettes (default 3), so:

    sessions      : 10, 100, 1000
    vignettes     : 30, 300, 3000

For each session scale S:
  1. Stream a corpus subset sized to the scale (~40 records per vignette) so
     index build time and memory actually scale with S instead of always
     rebuilding over the full corpus.
  2. Generate S * VIGNETTES_PER_SESSION chart-review vignettes from that
     subset (target notes are query-boosted exactly like the original
     benchmark build).
  3. Write the boosted corpus + vignettes + patient-stratified splits to
     data/scaling/<S>/.
  4. Run src/models/train.py on that scale's corpus and train/val splits,
     training fresh CMA/GDT encoders and JEPA predictors per scale. Models are
     saved to <models-dir>/<S>/. Use --skip-train to reuse existing per-scale
     models.
  5. Build each condition's index from the trained per-scale components and
     run the randomized four-period crossover experiment on all vignettes. The
     parent never forks (macOS fork-safety): it dumps each built model to a
     pickle and spawns a fresh eval subprocess which loads it and forks a
     multithreaded worker pool -> outputs/scaling/<S>/results.csv.
  6. Run the statistical analysis -> outputs/scaling/<S>/statistics.json and
     outputs/scaling/<S>/primary_results.csv.
  7. Run the ablation study with the SAME per-scale trained models ->
     <out-root>/<S>/ablation/.

Each scale is generated independently (seed = --seed + S).

Optional first step: run src/data/prepare.py to build the corpus from raw
Hugging Face data (--prepare). Off by default because re-ingesting the raw
snapshot is slow; the corpus is normally prepared once beforehand.

Usage:
  python scripts/run_scaling_study.py                                   # all three session scales
  python scripts/run_scaling_study.py --prepare                         # ingest raw data first
  python scripts/run_scaling_study.py --scales 100 1000                 # subset of session scales
  python scripts/run_scaling_study.py --scales 10 100                   # fast smoke test
  python scripts/run_scaling_study.py --only-scale 10 --limit 50        # tiny smoke test
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import os

# Set environment variables BEFORE importing torch, numpy, or pyarrow
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
import torch.multiprocessing as mp

# Fork-safety on macOS: forking a process whose BLAS/OpenMP have already
# spawned worker threads leaves the children with broken dispatch/OpenMP state
# -> SIGSEGV in Accelerate's dispatch_apply / OpenBLAS. The parent NEVER forks:
# it builds each scale's index (multithreaded) and hands the model to a fresh
# eval subprocess via a pickle file. That subprocess loads the model (pure
# deserialization, so BLAS has not spun up threads yet) before mp.Pool forks
# its eval workers. The fork therefore always happens from a thread-free
# process while both index builds and eval workers stay multithreaded.

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

np.seterr(divide="ignore", invalid="ignore", over="ignore")

CONDITIONS = ["control", "bm25", "cma", "gdt"]
MODEL_FILES = {"control": "baseline.pkl", "bm25": "bm25.pkl",
               "cma": "cma.pkl", "gdt": "gdt.pkl"}
DEFAULT_SESSION_SCALES = [10, 100, 1000]
# Each session scale S is paired with VIGNETTES_PER_SESSION v vignettes:
#     1 -> 3, 10 -> 30, ..., 100000 -> 300000 vignettes.
VIGNETTES_PER_SESSION = 3
# Corpus is sized at ~RECORDS_PER_VIGNETTE records per vignette so index build
# cost scales with the vignette count. This mirrors the smoke-cap heuristic (40 * limit).
RECORDS_PER_VIGNETTE = 40
# Aim for >= vignettes / PATIENTS_PER_VIGNETTE eligible patients (>=8 notes each) in the
# subset so vignettes spread across patients instead of recycling one or two.
PATIENTS_PER_VIGNETTE = 10

MODEL = None  # module-level handle shared with forked workers (copy-on-write)


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_corpus_subset(corpus_path: Path, max_records: int,
                        min_eligible: int = 1, min_notes: int = 1) -> tuple[list[dict], int, int]:
    """Stream records until max_records are read AND min_eligible patients
    (>=min_notes notes each) are present, or the file ends.

    Returns (records, n_read, n_eligible). Streaming keeps small scales cheap;
    the eligible-patient guard keeps the vignette generator from failing on
    tiny subsets (eligible patients are sparse in the corpus).
    """
    records = []
    notes_by_patient: dict[str, int] = {}
    eligible = set()
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(rec)
            pid = rec["patient_id"]
            n_notes = notes_by_patient.get(pid, 0) + 1
            notes_by_patient[pid] = n_notes
            if n_notes >= min_notes:
                eligible.add(pid)
            if len(records) >= max_records and len(eligible) >= min_eligible:
                break
    return records, len(records), len(eligible)


def generate_scale(corpus: list[dict], n: int, seed: int, out_dir: Path) -> list[dict]:
    """Generate N vignettes from the scaled corpus, write them + splits."""
    from src.data.prepare import generate_real_vignettes
    from src.data.split import split_vignettes

    print(f"\n[{n}] Generating {n} vignettes from {len(corpus)} records (seed={seed})...")
    vignettes = generate_real_vignettes(corpus, n_vignettes=n, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vignettes.json").write_text(json.dumps(vignettes, indent=2), encoding="utf-8")
    # Write the (target-boosted) corpus subset so train.py and per-scale
    # retrievers all train/index on the same data.
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for rec in corpus:
            fh.write(json.dumps(rec) + "\n")

    try:
        train_v, val_v, test_v = split_vignettes(vignettes, seed=seed)
    except ValueError:
        # Fall back to a plain random split when stratification classes are
        # too sparse (e.g. tiny capped smoke-test corpora). Guarantee a
        # non-empty train/val so per-scale train.py can always tune.
        rng = random.Random(seed)
        shuffled = vignettes[:]
        rng.shuffle(shuffled)
        n_train = max(int(0.70 * len(shuffled)), min(1, len(shuffled)))
        n_rem = len(shuffled) - n_train
        n_val = max(int(0.15 * len(shuffled)), min(1, n_rem))
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
    from src.models.gdt import GDTRetriever

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
        # Prefer the lightweight components artifact; fall back to the full
        # retriever pickle (which embeds the whole training corpus) if absent.
        components_path = models_dir / "cma_components.pkl"
        if components_path.exists():
            comp = joblib_load(components_path)
            vectorizer, encoder, predictor = (
                comp["vectorizer"], comp["encoder"], comp["predictor"]
            )
            del comp
        else:
            cma_p = joblib_load(models_dir / MODEL_FILES["cma"])
            vectorizer, encoder, predictor = (
                cma_p.vectorizer, cma_p.encoder, cma_p.predictor
            )
            del cma_p
            gc.collect()
        model = CMARetriever(
            corpus,
            curvature_threshold=cma_cfg.get("curvature_threshold", 0.65),
            gate_discount=cma_cfg.get("gate_discount", 0.05),
            context_window=cma_cfg.get("context_window", 5),
            prefetch_weight=cma_cfg.get("prefetch_weight", 0.4),
            vectorizer=vectorizer,
            encoder=encoder,
            predictor=predictor,
            encoder_pretrain_epochs=0,
            encoder_finetune_epochs=0,
        )
        gc.collect()
        print(f"    cma index built in {time.time() - t0:.0f}s")
        return model

    elif condition == "gdt":
        gdt_cfg = config.get("gdt", {})
        print("  building GDT (reusing trained encoder/predictor + tuned hyper-params)...")
        components_path = models_dir / "gdt_components.pkl"
        if components_path.exists():
            comp = joblib_load(components_path)
            vectorizer, encoder, predictor = (
                comp["vectorizer"], comp["encoder"], comp["predictor"]
            )
            del comp
        else:
            gdt_p = joblib_load(models_dir / MODEL_FILES["gdt"])
            vectorizer, encoder, predictor = (
                gdt_p.vectorizer, gdt_p.encoder, gdt_p.predictor
            )
            del gdt_p
            gc.collect()
        model = GDTRetriever(
            corpus,
            curvature_threshold=gdt_cfg.get("curvature_threshold", 1.0),
            gate_temperature=gdt_cfg.get("gate_temperature", 0.5),
            gate_lexical_include=gdt_cfg.get("gate_lexical_include", 0.75),
            prefetch_weight=gdt_cfg.get("prefetch_weight", 0.3),
            semantic_weight=gdt_cfg.get("semantic_weight", 0.15),
            vectorizer=vectorizer,
            encoder=encoder,
            predictor=predictor,
        )
        gc.collect()
        print(f"    gdt index built in {time.time() - t0:.0f}s")
        return model


def joblib_load(path: Path):
    import joblib
    return joblib.load(path)


def joblib_dump(obj, path: Path):
    import joblib
    joblib.dump(obj, path)


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


def run_eval_only(args):
    """Entry point for the per-condition eval subprocess launched by main().

    Runs in a fresh process so it can safely fork an eval pool from a
    thread-free state: the model is loaded from disk (deserialization runs no
    matrix ops), so BLAS has not spun up threads before the fork. Workers then
    run multithreaded searches. Writes one condition's results CSV.
    """
    mp.set_start_method("fork", force=True)
    model = joblib_load(args.model)
    global MODEL
    MODEL = model

    vignettes = _load_json(args.vignettes)
    if args.limit:
        vignettes = vignettes[: args.limit]

    rng = random.Random(args.seed)
    orders = [
        {cond: period for period, cond in enumerate(rng.sample(CONDITIONS, k=len(CONDITIONS)), start=1)}
        for _ in vignettes
    ]

    t0 = time.time()
    rows = evaluate_condition(args.condition, model, vignettes, orders,
                              args.seed, args.top_k, args.workers)
    print(f"    eval subprocess {args.condition}: {len(rows)} rows in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df["condition_code"] = (df["condition"].isin(["cma", "gdt"])).astype(int)
    df.to_csv(args.results, index=False)
    print(f"    wrote {args.results}")


def main():
    ap = argparse.ArgumentParser(description="Run CMA/BM25/TF-IDF scaling study")
    ap.add_argument("--scales", type=int, nargs="*", default=None,
                    help="Session scales to run (default: 10 100 1000). "
                         "Each session scale S generates S * --vignettes-per-session vignettes.")
    ap.add_argument("--only-scale", type=int, default=None,
                    help="Run a single session scale (overrides --scales).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap vignettes used at each scale (smoke-testing only).")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/scaling_prepared/corpus.jsonl")
   # ap.add_argument("--corpus", type=Path, default=ROOT / "data/processed/corpus.jsonl")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models",
                    help="Root models dir. Each session scale S trains into "
                         "<models-dir>/<S>/ and builds retrievers from there.")
    ap.add_argument("--vignettes-per-session", type=int, default=VIGNETTES_PER_SESSION,
                    help="Vignettes generated per session scale (default 3).")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip per-scale train.py and reuse existing models in "
                         "<models-dir>/<S>/ (no-op if models are missing).")
    ap.add_argument("--pretrain-max-docs", type=int, default=100000,
                    help="Passed to train.py: cap SPD autoencoder pretrain docs.")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="Skip the per-scale ablation study step.")
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
    ap.add_argument("--records-per-vignette", type=int, default=RECORDS_PER_VIGNETTE,
                    help="Target corpus records per vignette for each scale "
                         "(corpus is capped at the full file size).")
    ap.add_argument("--skip-analysis", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="Eval-subprocess entry point (used internally by main()).")
    ap.add_argument("--model", type=Path, default=None,
                    help="Pickled retriever model for --eval-only.")
    ap.add_argument("--vignettes", type=Path, default=None,
                    help="Vignettes JSON for --eval-only.")
    ap.add_argument("--condition", type=str, default=None, choices=CONDITIONS,
                    help="Condition arm for --eval-only.")
    ap.add_argument("--results", type=Path, default=None,
                    help="Where --eval-only writes its results CSV.")
    args = ap.parse_args()

    if args.eval_only:
        if not (args.model and args.vignettes and args.condition and args.results):
            ap.error("--eval-only requires --model, --vignettes, --condition, --results")
        run_eval_only(args)
        return

    if args.only_scale:
        sessions = [args.only_scale]
    elif args.scales:
        sessions = sorted(args.scales)
    else:
        sessions = DEFAULT_SESSION_SCALES

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

    print(f"Sessions: {sessions}")
    print(f"Workers: {args.workers} | top_k: {args.top_k} | seed: {args.seed}")
    print(f"Corpus: {corpus_path}")

    for s in sessions:
        n = s * args.vignettes_per_session
        scale_dir = args.data_root / str(s)
        out_dir = args.out_root / str(s)
        models_dir = args.models_dir / str(s)
        out_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Session scale {s} -> {n} vignettes ===")

        # Scale the corpus with the vignette count so index build time/memory
        # grow with the scale instead of always rebuilding over the full corpus.
        if args.limit:
            max_records = 40 * args.limit
            min_eligible = 1
        else:
            max_records = args.records_per_vignette * n
            min_eligible = max(1, n // PATIENTS_PER_VIGNETTE)
        corpus, n_read, n_eligible = _load_corpus_subset(
            corpus_path, max_records, min_eligible, min_notes=1
        )
        print(f"[{s}] Corpus subset: {n_read} records read "
              f"({n_eligible} eligible patients; target {max_records} records)")

        corpus = generate_scale(corpus, n, args.seed + s, scale_dir)

        vignettes = _load_json(scale_dir / "vignettes.json")
        print(f"[{s}] {len(vignettes)} vignettes loaded")

        # Train fresh per-scale models (or reuse existing ones with --skip-train).
        if not args.skip_train:
            print(f"[{s}] Running train.py for session scale {s} (out: {models_dir})...")
            t0 = time.time()
            train_cmd = [
                sys.executable, str(ROOT / "src/models/train.py"),
                "--corpus", str(scale_dir / "corpus.jsonl"),
                "--train", str(scale_dir / "train_vignettes.json"),
                "--val", str(scale_dir / "val_vignettes.json"),
                "--out-dir", str(models_dir),
                "--pretrain-max-docs", str(args.pretrain_max_docs),
            ]
            subprocess.run(train_cmd, check=True, cwd=ROOT)
            print(f"[{s}] train.py done in {time.time() - t0:.0f}s")
        else:
            print(f"[{s}] --skip-train set; reusing per-scale models from {models_dir}")
            if not (models_dir / "config.json").exists():
                raise FileNotFoundError(
                    f"{models_dir / 'config.json'} missing; cannot use --skip-train "
                    f"before running the scale at least once."
                )

        parts = []
        for condition in CONDITIONS:
            print(f"  evaluating condition '{condition}' over {len(vignettes)} vignettes...")
            t0 = time.time()

            # Build only the current condition's index (multithreaded) and hand
            # it to a fresh eval subprocess via a pickle so the eval pool can be
            # forked from a thread-free process without re-pickling per worker.
            model_path = scale_dir / f"model_{condition}.pkl"
            model = build_retriever(condition, corpus, models_dir)
            joblib_dump(model, model_path)
            del model
            gc.collect()

            # CMA search runs dense numpy matmuls (doc_latent @ intent) which
            # Accelerate implements with dispatch_apply. Forked workers must not
            # call that (SIGSEGV "multi-threaded process forked"), so the CMA
            # eval subprocess forces single-threaded BLAS: the model load then
            # spawns no threads (safe fork) and workers never hit dispatch_apply.
            # Control/BM25 search over sparse/dict indexes and are safe with
            # multithreaded workers, so only CMA gets the single-thread env.
            eval_env = None
            if condition in ("cma", "gdt"):
                eval_env = os.environ.copy()
                for _k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    eval_env[_k] = "1"

            part_path = out_dir / f"results_{condition}.csv"
            subprocess.run(
                [
                    sys.executable, str(Path(__file__)), "--eval-only",
                    "--model", str(model_path),
                    "--vignettes", str(scale_dir / "vignettes.json"),
                    "--condition", condition,
                    "--results", str(part_path),
                    "--seed", str(args.seed),
                    "--top-k", str(args.top_k),
                    "--workers", str(args.workers),
                    "--limit", str(args.limit),
                ],
                check=True, cwd=ROOT, env=eval_env,
            )
            model_path.unlink(missing_ok=True)
            parts.append(pd.read_csv(part_path))
            part_path.unlink(missing_ok=True)
            print(f"    {condition} done in {time.time() - t0:.0f}s")

        df = pd.concat(parts, ignore_index=True)
        results_path = out_dir / "results.csv"
        df.to_csv(results_path, index=False)
        print(f"[{s}] Saved results: {results_path} ({len(df)} rows)")

        del df, parts, corpus
        gc.collect()

        if not args.skip_analysis:
            print(f"[{s}] Running statistical analysis...")
            subprocess.run(
                [sys.executable, str(ROOT / "src/analysis/analyze.py"),
                 "--results", str(results_path), "--out-dir", str(out_dir)],
                check=True, cwd=ROOT,
            )

        # Ablation study using the SAME per-scale trained models and vignettes.
        if not args.skip_ablation:
            print(f"[{s}] Running ablation study (per-scale models + vignettes)...")
            ablation_out = out_dir / "ablation"
            subprocess.run(
                [sys.executable, str(ROOT / "src/experiments/ablation.py"),
                 "--processed-dir", str(scale_dir),
                 "--models-dir", str(models_dir),
                 "--out-dir", str(ablation_out)],
                check=True, cwd=ROOT,
            )

    print("\nScaling study complete.")
    print(f"Per-scale artifacts under {args.data_root} and {args.out_root}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()