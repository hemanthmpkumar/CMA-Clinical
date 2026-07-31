#!/usr/bin/env python3
"""One-shot entrypoint for reproducing the CMA experiment pipeline."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], description: str) -> None:
    print(f"\n== {description} ==")
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    run([sys.executable, "src/data/split.py"], "Splitting vignettes")
    run([sys.executable, "src/models/train.py"], "Training retrieval models")
    run([sys.executable, "src/experiments/run.py", "--use-trained"], "Running crossover experiment")
    run([sys.executable, "src/analysis/analyze.py"], "Analyzing experiment results")
    run([sys.executable, "src/viz/plots.py"], "Generating manuscript figures")
    print("\nReproduction complete. Outputs are in outputs/ and outputs/figures/.")
