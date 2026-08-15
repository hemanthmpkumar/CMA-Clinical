# CMA Clinical Search Benchmark

A reproducible implementation of the simulation-based evaluation described in the manuscript **“Continuum Memory Architecture (CMA) for Clinical Search”**. The codebase downloads (or synthesizes) de-identified EHR-style notes, builds four retrieval systems (a conventional TF-IDF session-based baseline, a BM25 session-based baseline, a geometry-aware “distance to target” (GDT) retriever, and the CMA system with curvature-aware gating + JEPA prefetching), runs a randomized four-arm crossover experiment, and produces statistical summaries and manuscript figures.

## What's included

* `src/data/` — download, preparation, and patient-level split scripts for MIMIC-III/IV/CXR, eICU, HiRID, and synthetic data.
* `src/models/` — TF-IDF baseline, BM25 baseline, GDT, CMA retrieval model, and model-training script.
* `src/experiments/` — simulated four-arm crossover evaluation, result recording, and component-wise ablation.
* `src/analysis/` — paired statistical tests, effect sizes, mixed-effects models, and GEE.
* `src/viz/` — figure generation for the manuscript plots.
* `latex/` — the journal-ready manuscript (`main.tex`), BibTeX library, and compiled PDF.

## Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `rank-bm25`, `joblib`, `torch`, `geoopt` (plus `datasets`/`pyarrow` for the Hugging Face data sources).

## Step-by-step usage

### 0. Clone / inspect the repository structure

```bash
ls src/
# data/  models/  experiments/  analysis/  viz/
```

All commands below assume the repository root as working directory.

---

### 1. Data download

#### Option A — Use real MIMIC-III/IV data (credentialed access, Q1-level)

MIMIC-III and MIMIC-IV are hosted on PhysioNet and require a completed CITI training and a signed Data Use Agreement. This project can download the clinical-note tables plus the structured tables that are commonly used in peer-reviewed clinical-AI benchmarks (admissions, patients, diagnoses, procedures, prescriptions, labs).

> **If you do not have PhysioNet credentials**, use the synthetic-only pipeline in Option B. The experiments below were validated on the default 10,000-patient synthetic corpus.

```bash
# View detailed download instructions
python src/data/download.py --instructions

# Set PhysioNet credentials
export PHYSIONET_USER=your_user
export PHYSIONET_PASS=your_pass

# MIMIC-III default tables (~5.6 GB compressed)
python src/data/download.py --dataset mimiciii --out data/raw

# MIMIC-III including CHARTEVENTS (~38 GB compressed)
python src/data/download.py --dataset mimiciii --full --out data/raw

# MIMIC-IV default tables (~3.1 GB compressed)
python src/data/download.py --dataset mimiciv --out data/raw

# MIMIC-IV + MIMIC-CXR metadata and radiology reports (~7 GB compressed + files/)
python src/data/download.py --dataset mimiciv-cxr --out data/raw
python src/data/download.py --dataset mimiciv-cxr --include-cxr-reports --out data/raw

# eICU Collaborative Research Database (~5 GB compressed)
python src/data/download.py --dataset eicu --out data/raw

# HiRID high-resolution ICU dataset (~3 GB compressed, parquet)
python src/data/download.py --dataset hirid --out data/raw

# Both MIMIC-III and MIMIC-IV
python src/data/download.py --dataset all --out data/raw
```

Downloaded PHI/notes must stay under `data/raw/` (git-ignored). Do not commit them.

#### Option B — Use the built-in synthetic corpus (default / no credentials)

No manual download is needed. The default target size is **10,000 patients** (producing ~80,000 notes) to match larger-scale benchmark standards.

#### Option C — Use the Hugging Face mirrors

Requires a Hugging Face access token with read permission for the gated repositories.

```bash
# Set token once per shell session
export HF_TOKEN=hf_...

# Inspect the requested Hugging Face datasets
python src/data/download_hf.py --dataset all --inspect

# Download MIMIC-III from Hugging Face
python src/data/download_hf.py --dataset mimiciii --out data/raw/huggingface

# Download MIMIC-IV from Hugging Face
python src/data/download_hf.py --dataset mimiciv --out data/raw/huggingface

# Download MIMIC-CXR from Hugging Face
python src/data/download_hf.py --dataset mimiciv-cxr --out data/raw/huggingface

# Download eICU from Hugging Face
python src/data/download_hf.py --dataset eicu --out data/raw/huggingface

# Download MIMIC-IV notes, PPG/ECG signals, and MIMIC-CXR-RRG reports
python src/data/download_huggingface.py --dataset mimiciv-note --out data/raw/huggingface
python src/data/download_huggingface.py --dataset mimiciv-ppg-ecg --out data/raw/huggingface
python src/data/download_huggingface.py --dataset mimic-cxr-rrg --out data/raw/huggingface
```

The supported datasets are:

* MIMIC-III: `ntphuc149/MIMIC-III-Clinical-Database`
* MIMIC-IV: `lucky9-cyou/MIMIC-IV`
* MIMIC-CXR: `EvidenceAIResearch/MIMIC-CXR-VReason`
* eICU: `wshi83/EHRAgent-eicu`
* MIMIC-IV notes: `mimic-capstone/mimic-iv-note`
* MIMIC-IV PPG-ECG: `lucky9-cyou/mimic-iv-aligned-ppg-ecg`
* MIMIC-CXR-RRG: `Yamini-1628/MIMIC-CXR-RRG`

For dataset preparation, point `--huggingface-dir` at the snapshot root. The
ingestor auto-detects every present dataset subdirectory (`mimiciii/`, `mimiciv/`,
`mimiciv-note/`, `mimiciv-cxr/`, `mimic-cxr-rrg/`, `eicu/`, `mimiciv-ppg-ecg/`,
`hirid/`):

```bash
python src/data/prepare.py --huggingface-dir data/raw/huggingface --n-vignettes 100000
```

---

### 2. Prepare data

#### Synthetic-only mode (default, no PHI)

```bash
# Default: 10,000 synthetic patients and 60 vignettes
python src/data/prepare.py --synthetic-only

# Fast smoke test with fewer patients
python src/data/prepare.py --synthetic-only --n-patients 500 --n-vignettes 60

# Generate 10,000 complex chart-review vignettes
python src/data/prepare.py --synthetic-only --n-patients 100000 --n-vignettes 100000 --complexity-filter high
```

Outputs written to `data/processed/`:

* `corpus.jsonl` — searchable clinical-note corpus.
* `vignettes.json` — simulated chart-review tasks with ground-truth targets. Queries are derived from the **actual text** of the target notes (top TF-IDF content terms), not from a synthetic keyword vocabulary, so retrieval is exercised against real clinical language.
* `case_metadata.csv` — vignette-level metadata (specialty, experience group, complexity, pivots).
* `seed_info.json` — reproducibility summary.

#### Using real MIMIC-III notes (directory mode, supports structured tables)

```bash
# Ingest NOTEEVENTS plus PATIENTS, ADMISSIONS, DIAGNOSES_ICD, PROCEDURES_ICD, etc.
python src/data/prepare.py --mimiciii-dir data/raw

# Use all available real patients and generate as many vignettes as the corpus allows
python src/data/prepare.py --mimiciii-dir data/raw --n-vignettes 100000
```

#### Using MIMIC-IV

```bash
python src/data/prepare.py --mimiciv-dir data/raw

# Use all available real patients and generate as many vignettes as the corpus allows
python src/data/prepare.py --mimiciv-dir data/raw --n-vignettes 100000
```

#### Using MIMIC-IV + MIMIC-CXR

There is no `--mimiciv-cxr-dir` flag. MIMIC-CXR and the other Hugging Face
mirror datasets (MIMIC-CXR-RRG, MIMIC-IV notes, MIMIC-IV PPG-ECG, HiRID) are
ingested through `--huggingface-dir`, which auto-detects each dataset
subdirectory:

```bash
python src/data/prepare.py --huggingface-dir data/raw/huggingface --n-vignettes 100000
```

#### Using eICU-CRD

```bash
python src/data/prepare.py --eicu-dir data/raw/huggingface/eicu --n-vignettes 100000
```

#### Using HiRID

HiRID has no `--hirid-dir` flag; ingest it through `--huggingface-dir` as above.

#### Legacy single-file paths still work

```bash
python src/data/prepare.py --mimiciii data/raw/NOTEEVENTS.csv.gz --n-vignettes 60
```

---

### 3. Train the retrieval models

The project follows a standard ML workflow:

1. Vignettes are split into **train / validation / test** sets by patient to avoid leakage.
2. The TF-IDF baseline, BM25 baseline, CMA, and GDT retrievers are fit on the corpus and their hyper-parameters are chosen on the validation set.
3. Best models are pickled to `models/` for reproducible evaluation.

The two commands are sequential: `split.py` writes the train/val/test vignette
splits that `train.py` reads to fit and validate the retrievers.

```bash
# Split vignettes (must run before train.py)
python src/data/split.py

# Train baseline + BM25 + CMA + GDT and save models/
python src/models/train.py
```

`split.py` also supports a `--scales` mode that generates and splits the exact
vignette counts used by the scaling study. Give VIGNETTE counts: the session
scales 1, 10, 100, 1,000, 10,000, 100,000 correspond to 3, 30, 300, 3,000,
30,000, 300,000 vignettes (3 per session). Like the scaling study, it draws
vignettes from the **real corpus** records (each query built from its target
note's actual text):

```bash
# Real-data vignettes for the six session scales, written to data/scaling/<S>/
python src/data/split.py --scales 3 30 300 3000 30000 300000 --corpus data/processed/corpus.jsonl

# Custom corpus / output root
python src/data/split.py --scales 30 300 --corpus data/scaling_prepared/corpus.jsonl \
    --data-root data/scaling
```

> **Apple Silicon note:** the CMA SPD encoder needs `torch.linalg.eigh`, which is
> not implemented on the MPS backend. Training automatically probes the device
> and falls back to CPU (`src/models/spd_encoder.py::pick_device`), so no manual
> `PYTORCH_ENABLE_MPS_FALLBACK` or device flag is required.

Outputs:

* `data/processed/train_vignettes.json`, `val_vignettes.json`, `test_vignettes.json`
* `models/baseline.pkl` — trained TF-IDF baseline retriever
* `models/bm25.pkl` — trained BM25 baseline retriever
* `models/cma.pkl` — trained CMA retriever
* `models/gdt.pkl` — trained GDT retriever
* `models/cma_components.pkl` — lightweight CMA encoder/predictor (reused by the scaling study)
* `models/gdt_components.pkl` — lightweight GDT encoder/predictor (reused by the scaling study)
* `models/config.json` — selected hyper-parameters and validation scores

---

### 4. Conduct experiments

The experiment is a randomized within-subject four-arm crossover: every vignette is completed once under the TF-IDF control, once under the BM25 baseline, once under GDT, and once under CMA, with order counterbalanced.

For a one-shot reproduction of the paper pipeline, run:

```bash
python run_experiments.py
```

This will split the vignettes, train the four retrievers (baseline, BM25, GDT, CMA), run the crossover experiment, analyze the results, and generate the figures in `outputs/figures/`.

#### 4a. Run the full experiment on the held-out test set

```bash
python src/experiments/run.py --use-trained
```

To run from scratch without the trained-model split:

```bash
python src/experiments/run.py \
    --processed-dir data/processed \
    --top-k 10 \
    --seed 20260617 \
    --out outputs/results.csv
```

This writes `outputs/results.csv` with one row per simulated session and metrics including time-to-correct-information, accuracy, number of queries, latency, and simulated NASA-TLX subscales.

#### 4b. Inspect raw results quickly

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/results.csv")
print(df.groupby("condition")[["time_to_info", "accuracy", "cognitive_load", "latency_ms"]].mean())
PY
```

#### 4c. Run the component-wise ablation

```bash
python src/experiments/ablation.py --out-dir outputs/ablation
```

Each ablation variant targets **GDT** (the primary intervention, which shares
CMA's SPD-encoder + GSI-gate + JEPA-prefetch architecture). The three variants
are Full GDT, GSI-only (gate on, prefetch off), and JEPA-only (gate disabled,
prefetch on). Each variant is evaluated under the **same four-arm crossover**
used by the main experiment — TF-IDF control, BM25, CMA, and GDT — so the
ablation results stay directly comparable to the `results.csv` and comparison
figures. The study writes `ablation_results.csv`
(raw per-session rows), `ablation_summary.csv` (variant × arm means),
`ablation_pct_change.csv` (each arm vs control per variant), and
`ablation_report.json`

To reuse already-trained models from `models/` instead of re-training the
components (fast path used by the scaling study):

```bash
python src/experiments/ablation.py --models-dir models --out-dir outputs/ablation
```

#### 4d. Run individual components separately (for debugging)

```bash
# Build only the corpus and vignettes
python src/data/prepare.py --synthetic-only --n-patients 200 --n-vignettes 30

# Run only the simulated experiment
python src/experiments/run.py --processed-dir data/processed --top-k 10
```

#### 4e. Scaling study (index build time vs. corpus scale)

`scripts/run_scaling_study.py` reproduces the manuscript's scaling analysis. Scales are
specified as **sessions**; each session scale `S` is paired with `S × 3` vignettes
(`--vignettes-per-session`, default 3), so:

| sessions | 1 | 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|---|
| vignettes | 3 | 30 | 300 | 3,000 | 30,000 | 300,000 |

For each session scale it streams a corpus subset sized to the scale (~40 records per
vignette), generates the real-corpus vignettes (each query derived from its target note's
actual text), **runs `src/models/train.py` fresh on that scale's corpus and splits**
(so CMA/GDT encoders and JEPA predictors are re-trained per scale into
`models/<S>/`), builds each retriever's index from those per-scale components, runs the
four-arm crossover on all vignettes, writes per-scale stats, then runs the ablation study
with the **same** per-scale trained models into `outputs/scaling/<S>/ablation/`.

The identical per-scale vignette generation is available standalone through
`split.py --scales ...` (see Section 3) if you want the vignettes before running the
full study.

Prerequisites: the raw (Hugging Face) corpus at `data/scaling_prepared/corpus.jsonl`.

```bash
# All six session scales (1, 10, 100, 1000, 10000, 100000) + stats + ablation
python scripts/run_scaling_study.py

# Subset of session scales
python scripts/run_scaling_study.py --scales 10 1000

# Fast smoke test (degenerate scales still produce stats)
python scripts/run_scaling_study.py --scales 1 10

# Reuse already-trained per-scale models instead of re-running train.py
python scripts/run_scaling_study.py --skip-train

# Skip per-scale ablation study (stats + experiments only)
python scripts/run_scaling_study.py --skip-ablation

# Tiny smoke test capped to 50 vignettes
python scripts/run_scaling_study.py --only-scale 100 --limit 50

# Scale up parallel eval workers (default: min(cpu_count, 14))
python scripts/run_scaling_study.py --scales 10000 --workers 14

# Skip the per-scale statistics (results CSV only)
python scripts/run_scaling_study.py --skip-analysis

# Ingest the raw Hugging Face snapshot into data/scaling_prepared/ first
python scripts/run_scaling_study.py --prepare
```

Other knobs: `--records-per-vignette` (corpus records targeted per vignette, default 40),
`--pretrain-max-docs` (SPD autoencoder pretrain cap passed to train.py, default 100000),
`--corpus` (default `data/scaling_prepared/corpus.jsonl`), `--seed`, `--top-k`,
`--data-root`/`--out-root` (default `data/scaling`/`outputs/scaling`).

Outputs:

* `data/scaling/<S>/` — scaled corpus artifacts: `corpus.jsonl`, `vignettes.json`,
  `train/val/test_vignettes.json`.
* `models/<S>/` — per-scale trained retrievers (`baseline/bm25/cma/gdt.pkl`, components, `config.json`).
* `outputs/scaling/<S>/results.csv` — `4 × vignettes` crossover session rows (one per condition arm).
* `outputs/scaling/<S>/statistics.json` + `primary_results.csv` — per-scale analysis.
* `outputs/scaling/<S>/ablation/` — per-scale ablation study using the same trained models.

To plot build-time/memory scaling and per-scale comparison figures from the resulting
per-scale stats:

```bash
python scripts/plot_scaling.py
python scripts/plot_scaling.py --scales 10 1000   # subset of session scales
```

To **combine every scale that has been run into cumulative cross-scale result
figures** (all four arms — TF-IDF, BM25, CMA, GDT), including only the scales
actually present under `outputs/scaling/`:

```bash
python scripts/plot_scaling_combined.py
python scripts/plot_scaling_combined.py --scales 1 10 100 1000 10000
```

The combined script recomputes all metrics directly from each scale's
`results.csv` (so it works even when a scale predates the four-arm setup —
missing arms are skipped on that scale) and writes to
`outputs/scaling/combined/`:

* `scaling_trends.png` — mean time-to-info, cognitive load, latency, and query count vs session scale `S` (log x-axis), one line per arm.
* `scaling_pct_change.png` — % change vs the TF-IDF control for BM25, CMA, and GDT across scales.
* `scaling_accuracy.png` — task accuracy vs scale per arm.
* `combined_results.csv` / `combined_pct_change.csv` — the underlying per-scale aggregations for manuscript tables.

Design notes:

* Vignettes use **real corpus data**: queries are the top content terms of their
  target note (stopwords removed), so no synthetic keyword vocabulary is involved.
  The per-patient note threshold is relaxed automatically so single-summary corpora
  (e.g. eICU, one structured summary per stay) still work.
* The parent process never forks (macOS fork-safety). Each condition's index is built
  multithreaded in the parent, pickled, then handed to a fresh eval subprocess that loads
  it and forks the worker pool from a thread-free state; parallelism comes from the worker
  processes.
* The CMA eval subprocess forces single-threaded BLAS (`VECLIB_MAXIMUM_THREADS=1`, etc.)
  because CMA search runs dense matmuls that SIGSEGV inside forked workers on macOS
  (`dispatch_apply` after fork). Control/BM25 use sparse/dict indexes and keep
  multithreaded workers.
* At the largest session scale (100,000 sessions / 300,000 vignettes) the target corpus
  (~12M records) exceeds the full MIMIC-IV corpus (546K records), so the index covers the
  entire corpus and the run is gated mostly by per-scale model training
  (`--pretrain-max-docs` caps the SPD autoencoder pretrain to keep it tractable; the full
  corpus traines realizistically in a few hours per retriever).

---

### 5. Result data analysis

```bash
python src/analysis/analyze.py \
    --results outputs/results.csv \
    --out-dir outputs
```

Outputs:

* `outputs/statistics.json` — full numerical results (Wilcoxon tests, mixed-effects model, GEE, subgroup analyses, and a dedicated `gdt_vs_benchmarks` block of paired GDT-vs-{TF-IDF, BM25, CMA} contrasts for every outcome).
* `outputs/primary_results.csv` — human-readable summary table with means/medians per condition and GDT-vs-{BM25, CMA} p-values/Cohen's d plus the primary GDT-vs-Control contrast.

GDT is the primary intervention; CMA is treated as a **benchmark** (alongside
TF-IDF and BM25), so every outcome reports the three benchmark-vs-GDT contrasts.

Key analyses performed:

* **Primary (time):** paired Wilcoxon signed-rank test on time-to-correct-information; mixed-effects linear model on log-transformed time with random intercept per vignette.
* **Accuracy:** McNemar test on discordant pairs; generalized estimating equations (GEE) with binomial family clustered by vignette.
* **Secondary outcomes:** paired tests and Cohen's d for NASA-TLX composite, query latency, and number of queries issued.
* **Subgroups:** median percentage time improvement by specialty, clinician experience, and case complexity.
* **Benchmark contrasts:** paired GDT vs each of TF-IDF, BM25, and CMA for time, cognitive load, latency, query count, and accuracy (consumed by `gdt_vs_benchmarks*.png`).

---

### 6. Generate graphs and discuss results

```bash
python src/viz/plots.py \
    --results outputs/results.csv \
    --stats outputs/statistics.json \
    --out-dir outputs/figures
```

Generated figures (matching the manuscript):

1. `consort_flow.png` — case flow and condition randomization.
2. `time_distribution.png` — time-to-correct-information density by condition.
3. `accuracy_comparison.png` — accuracy by condition.
4. `intent_trajectory.png` — schematic trajectory on the CMA topic manifold with a curvature gate.
5. `tlx_subscales.png` — NASA-TLX subscale comparison.
6. `latency.png` — query latency box plot.
7. `subgroup_forest.png` — forest plot of time-to-info improvement across subgroups.
8. `gdt_vs_benchmarks.png` — GDT (primary) vs each of the three benchmarks (TF-IDF, BM25, CMA) for time-to-info, cognitive load, latency, and query count.
9. `gdt_vs_benchmarks_accuracy.png` — task-accuracy delta of GDT vs each benchmark.
10. `ablation_four_arm.png` — ablation study with all four arms (TF-IDF, BM25, CMA, GDT) per variant (rendered when `--ablation-report` is passed or via `plot_scaling.py`).
11. `cma_components.png` — CMA retrieval architecture diagram.

#### Interpreting the outputs

The printed analysis summary reports the median percentage reduction in time-to-correct-information for GDT versus each benchmark (TF-IDF, BM25, CMA), the comparison of cognitive-load scores, and the latency improvement. Example interpretation points:

* In `gdt_vs_benchmarks.png`, a positive bar for a benchmark means GDT retrieved the correct information faster (time-to-info/latency) or imposed lower cognitive load than that benchmark; the accuracy figure shows the task-accuracy delta in percentage points (GDT minus benchmark).
* Latency reductions reflect the JEPA prefetch caching warm-start documents.
* Accuracy should remain non-inferior: GDT/CMA are intended to speed retrieval without harming correctness.
* NASA-TLX reductions suggest lower cognitive burden when stale context is suppressed.

The exact numbers depend on the corpus (MIMIC vs. synthetic) and on retriever hyper-parameters in `src/models/baseline.py`, `src/models/bm25.py`, `src/models/gdt.py`, and `src/models/cma.py`.

---

## One-command full pipeline (10,000 synthetic patients)

```bash
source .venv/bin/activate
python src/data/prepare.py --synthetic-only
python src/data/split.py
python src/models/train.py
python src/experiments/run.py --use-trained
python src/analysis/analyze.py
python src/viz/plots.py
```

On a laptop the full pipeline runs in a few minutes with 10,000 synthetic patients/80,000 notes.

---

## LaTeX manuscript

To rebuild the manuscript PDF:

```bash
cd latex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Output: `latex/main.pdf`.

---

## Project structure

```
.
├── data/
│   ├── raw/                  # downloaded MIMIC data (git-ignored)
│   ├── processed/            # corpus.jsonl, vignettes.json, metadata
│   ├── scaling_prepared/     # corpus.jsonl for the scaling study (4.5 GB, 2.79M records)
│   └── scaling/              # per-scale vignettes + splits (data/scaling/<S>/)
├── models/                 # trained retriever checkpoints
│   ├── baseline.pkl          # TF-IDF baseline
│   ├── bm25.pkl              # BM25 baseline
│   ├── gdt.pkl               # GDT retriever
│   ├── cma.pkl               # CMA retriever (full, embeds training corpus)
│   ├── gdt_components.pkl    # lightweight GDT vectorizer/encoder/predictor
│   ├── cma_components.pkl    # lightweight CMA vectorizer/encoder/predictor
│   └── config.json           # selected hyper-parameters + validation scores
├── outputs/
│   ├── results.csv           # raw experiment results
│   ├── statistics.json       # statistical analysis output
│   ├── primary_results.csv   # summary table
│   ├── figures/              # generated PNG figures
│   └── scaling/              # per-scale results.csv + stats (outputs/scaling/<S>/)
├── scripts/
│   ├── build_bm25.py         # one-off BM25 index builder
│   ├── run_scaling_study.py  # scaling study (train → build → eval → stats)
│   ├── plot_scaling.py       # per-scale scaling-study figures
│   └── plot_scaling_combined.py  # cumulative cross-scale result figures
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── download_hf.py
│   │   ├── download_huggingface.py
│   │   ├── prepare.py
│   │   ├── record_clusters.py
│   │   └── split.py
│   ├── models/
│   │   ├── base.py           # TF-IDF vectorizer factory (tiny-corpus fallback)
│   │   ├── baseline.py       # TF-IDF retriever
│   │   ├── bm25.py           # BM25 retriever
│   │   ├── cma.py            # CMA retriever (GSI gate + JEPA prefetch)
│   │   ├── gdt.py            # geometry-aware "distance to target" retriever
│   │   ├── dataloader.py
│   │   ├── gsi_gate.py
│   │   ├── jepa.py
│   │   ├── spd_encoder.py    # neural TF-IDF→SPD encoder (log-Euclidean)
│   │   └── train.py
│   ├── experiments/
│   │   ├── ablation.py
│   │   ├── run.py
│   │   └── simulate_users.py
│   ├── analysis/
│   │   └── analyze.py
│   ├── annotation/
│   │   ├── adjudicate.py
│   │   ├── export.py
│   │   ├── instrument.py
│   │   └── schemas.py
│   └── viz/
│       └── plots.py
├── latex/
│   ├── main.tex
│   ├── references.bib
│   ├── main.pdf
│   └── jbi/                  # JBI-format manuscript (main_jbi.tex)
├── run_experiments.py        # one-shot full pipeline
├── requirements.txt
└── README.md
```

## Notes on reproducibility and ethics

* All synthetic data are procedurally generated and contain no protected health information.
* Real MIMIC data require credentialed access and must be handled under the applicable Data Use Agreement.
* The simulator uses deterministic seeds by default; change `--seed` to explore uncertainty or generate bootstrap replicates.
* The pipeline is a research benchmark, not a clinical product. Any translation to live EHR systems requires institutional review, safety testing, bias audits, and clinician-in-the-loop oversight as discussed in the manuscript.

## Citation

If you use this code or the corresponding manuscript, please cite the relevant CMA manuscript and the clinical data sources (e.g., MIMIC-III/IV) when applicable.
