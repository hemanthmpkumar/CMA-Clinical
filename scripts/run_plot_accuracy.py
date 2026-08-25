#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("plots", "src/viz/plots.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)

# Synthetic minimal dataset covering all arms
rows = []
conds = ['control', 'bm25', 'cma', 'gdt']
for i, cond in enumerate(conds):
    for j in range(4):
        rows.append({
            'vignette_id': j,
            'condition': cond,
            'accuracy': 0.70 + 0.05 * i + 0.01 * j,
            'time_to_info': 10 + j + i,
            'latency_ms': 100 + 5 * i,
            'tlx_mental': 50,
            'tlx_physical': 40,
            'tlx_temporal': 30,
            'tlx_performance': 60,
            'tlx_effort': 45,
            'tlx_frustration': 20,
            'n_queries_issued': 3 + i,
        })

df = pd.DataFrame(rows)
out_dir = Path('outputs/figures')
out_dir.mkdir(parents=True, exist_ok=True)

plots.plot_accuracy_comparison(df, out_dir)
print('Wrote', out_dir / 'accuracy_comparison.png')
