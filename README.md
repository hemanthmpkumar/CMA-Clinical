# CMA Clinical Search Benchmark

A reproducible implementation of the simulation-based evaluation described in the manuscript **“Continuum Memory Architecture (CMA) for Clinical Search”**. The codebase downloads (or synthesizes) de-identified EHR-style notes, builds two retrieval systems (a conventional session-based baseline and the CMA system with curvature-aware gating + JEPA prefetching), runs a randomized crossover experiment, and produces statistical summaries and manuscript figures.

## What's included

* `src/data/` — download, preparation, and patient-level split scripts for MIMIC-III/IV/CXR, eICU, HiRID, and synthetic data.
* `src/models/` — baseline keyword-retrieval model, CMA retrieval model, and model-training script.
* `src/experiments/` — simulated crossover evaluation and result recording.
* `src/analysis/` — paired statistical tests, effect sizes, mixed-effects models, and GEE.
* `src/viz/` — figure generation for the seven manuscript plots.
* `latex/` — the journal-ready manuscript (`main.tex`), BibTeX library, and compiled PDF.

## Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `rank-bm25`.

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
```

The supported datasets are:

* MIMIC-III: `ntphuc149/MIMIC-III-Clinical-Database`
* MIMIC-IV: `lucky9-cyou/MIMIC-IV`
* MIMIC-CXR: `EvidenceAIResearch/MIMIC-CXR-VReason`
* eICU: `wshi83/EHRAgent-eicu`

For dataset preparation, use the existing prepare scripts with the downloaded directory:

```bash
python src/data/prepare.py --mimiciii-dir data/raw/huggingface/mimiciii --n-vignettes 100000
python src/data/prepare.py --mimiciv-dir data/raw/huggingface/mimiciv --n-vignettes 100000
python src/data/prepare.py --mimiciv-cxr-dir data/raw/huggingface/mimiciv-cxr --n-vignettes 100000
python src/data/prepare.py --eicu-dir data/raw/huggingface/eicu --n-vignettes 100000
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
python src/data/prepare.py --synthetic-only --n-patients 10000 --n-vignettes 10000 --complexity-filter high
```

Outputs written to `data/processed/`:

* `corpus.jsonl` — searchable clinical-note corpus.
* `vignettes.json` — simulated chart-review tasks with forced pivots and ground-truth targets.
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

```bash
python src/data/prepare.py --mimiciv-cxr-dir data/raw --n-vignettes 100000
```

#### Using eICU-CRD

```bash
python src/data/prepare.py --eicu-dir data/raw --n-vignettes 100000
```

#### Using HiRID

```bash
python src/data/prepare.py --hirid-dir data/raw --n-vignettes 100000
```

#### Legacy single-file paths still work

```bash
python src/data/prepare.py --mimiciii data/raw/NOTEEVENTS.csv.gz --n-vignettes 60
```

---

### 3. Train the retrieval models

The project follows a standard ML workflow:

1. Vignettes are split into **train / validation / test** sets by patient to avoid leakage.
2. The baseline TF-IDF retriever and the CMA retriever are fit on the corpus and their hyper-parameters are chosen on the validation set.
3. Best models are pickled to `models/` for reproducible evaluation.

```bash
# Split vignettes
python src/data/split.py

# Train baseline + CMA and save models/
python src/models/train.py
```

Outputs:

* `data/processed/train_vignettes.json`, `val_vignettes.json`, `test_vignettes.json`
* `models/baseline.pkl` — trained baseline retriever
* `models/cma.pkl` — trained CMA retriever
* `models/config.json` — selected hyper-parameters and validation scores

---

### 4. Conduct experiments

The experiment is a randomized within-subject crossover: every vignette is completed once under the baseline (control) condition and once under CMA, with order counterbalanced.

For a one-shot reproduction of the paper pipeline, run:

```bash
python run_experiments.py
```

This will split the vignettes, train the baseline and CMA retrievers, run the crossover experiment, analyze the results, and generate the figures in `outputs/figures/`.

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

#### 4c. Run individual components separately (for debugging)

```bash
# Build only the corpus and vignettes
python src/data/prepare.py --synthetic-only --n-patients 200 --n-vignettes 30

# Run only the simulated experiment
python src/experiments/run.py --processed-dir data/processed --top-k 10
```

---

### 5. Result data analysis

```bash
python src/analysis/analyze.py \
    --results outputs/results.csv \
    --out-dir outputs
```

Outputs:

* `outputs/statistics.json` — full numerical results (Wilcoxon tests, mixed-effects model, GEE, subgroup analyses).
* `outputs/primary_results.csv` — human-readable summary table of primary and secondary outcomes.

Key analyses performed:

* **Primary (time):** paired Wilcoxon signed-rank test on time-to-correct-information; mixed-effects linear model on log-transformed time with random intercept per vignette.
* **Accuracy:** McNemar test on discordant pairs; generalized estimating equations (GEE) with binomial family clustered by vignette.
* **Secondary outcomes:** paired tests and Cohen's d for NASA-TLX composite, query latency, and number of queries issued.
* **Subgroups:** median percentage time improvement by specialty, clinician experience, and case complexity.

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
3. `intent_trajectory.png` — schematic trajectory on the CMA topic manifold with a curvature gate.
4. `tlx_subscales.png` — NASA-TLX subscale comparison.
5. `latency.png` — query latency box plot.
6. `subgroup_forest.png` — forest plot of time-to-info improvement across subgroups.
7. `cma_components.png` — CMA retrieval architecture diagram.

#### Interpreting the outputs

The printed analysis summary reports the median percentage reduction in time-to-correct-information, the comparison of cognitive-load scores, and the latency improvement. Example interpretation points:

* A positive median reduction means CMA retrieved the correct information faster on average.
* Latency reductions reflect the JEPA prefetch caching warm-start documents.
* Accuracy should remain non-inferior: CMA is intended to speed retrieval without harming correctness.
* NASA-TLX reductions suggest lower cognitive burden when stale context is suppressed.

The exact numbers depend on the corpus (MIMIC vs. synthetic) and on retriever hyper-parameters in `src/models/baseline.py` and `src/models/cma.py`.

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
│   └── processed/            # corpus.jsonl, vignettes.json, metadata
├── models/                 # trained retriever checkpoints
│   ├── baseline.pkl
│   ├── cma.pkl
│   └── config.json
├── outputs/
│   ├── results.csv           # raw experiment results
│   ├── statistics.json       # statistical analysis output
│   ├── primary_results.csv   # summary table
│   └── figures/              # generated PNG figures
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── prepare.py
│   │   └── split.py
│   ├── models/
│   │   ├── base.py
│   │   ├── baseline.py
│   │   ├── cma.py
│   │   └── train.py
│   ├── experiments/
│   │   ├── simulate_users.py
│   │   └── run.py
│   ├── analysis/
│   │   └── analyze.py
│   └── viz/
│       └── plots.py
├── latex/
│   ├── main.tex
│   ├── references.bib
│   └── main.pdf
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
