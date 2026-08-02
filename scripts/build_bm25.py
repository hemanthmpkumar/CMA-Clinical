#!/usr/bin/env python3
"""Build the BM25 retriever index on the full corpus and save to models/bm25.pkl.

Uses window_size=3 to match the tuned TF-IDF baseline config.json, and default
BM25 hyper-parameters (k1=1.5, b=0.75).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.bm25 import BM25Retriever

corpus_path = Path("data/processed/corpus.jsonl")
out_path = Path("models/bm25.pkl")
out_path.parent.mkdir(parents=True, exist_ok=True)

print("Loading corpus...", flush=True)
t0 = time.time()
corpus = []
with corpus_path.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            corpus.append(json.loads(line))
print(f"Loaded {len(corpus)} records in {time.time()-t0:.1f}s", flush=True)

print("Building BM25 index...", flush=True)
t0 = time.time()
retriever = BM25Retriever(corpus, window_size=3)
print(f"BM25 index built in {time.time()-t0:.1f}s", flush=True)

import joblib
t0 = time.time()
joblib.dump(retriever, out_path, compress=0)
print(f"Saved {out_path} in {time.time()-t0:.1f}s", flush=True)
print(f"BM25Retriever window_size={retriever.window_size}, k1={retriever.k1}, b={retriever.b}", flush=True)
