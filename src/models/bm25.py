#!/usr/bin/env python3
"""
src/models/bm25.py

Session-based BM25 (Okapi) retrieval comparison.

The retriever encodes the current query together with recent session queries
(uniform weighting, no pivot detection) and scores documents with BM25
(Robertson & Sparck Jones, 1976). It is included as a stronger classical
sparse baseline alongside the TF-IDF control so the value of CMA's session
management can be assessed against both weighting schemes.

The term-frequency matrix is stored as a sparse matrix so that the BM25 index
stays memory-bounded on large clinical corpora (rank_bm25's per-document
dictionaries would be prohibitive here).
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, corpus: list[dict], window_size: int = 20,
                 k1: float = 1.5, b: float = 0.75):
        super().__init__(corpus)
        self.window_size = window_size
        self.k1 = k1
        self.b = b

        doc_texts = [rec["text"] for rec in corpus]
        # Same tokenizer/vocabulary as the TF-IDF baseline so the comparison
        # isolates the weighting scheme (BM25 vs tf-idf), not the feature space.
        self.vectorizer = CountVectorizer(
            max_df=0.85, min_df=2, stop_words="english", max_features=4000
        )
        self.doc_counts = self.vectorizer.fit_transform(doc_texts)
        self.doc_counts = self.doc_counts.astype(np.float32)

        n_docs = len(self.corpus)
        self.doc_len = np.asarray(self.doc_counts.sum(axis=1)).ravel().astype(np.float64)
        self.avgdl = float(self.doc_len.mean()) if n_docs else 1.0
        df = np.asarray((self.doc_counts > 0).sum(axis=0)).ravel().astype(np.float64)
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, session_history: list[str], top_k: int = 10,
               **kwargs) -> list[tuple[str, float]]:
        # Uniform session expansion with the most recent prior queries.
        if self.window_size > 1:
            prior = session_history[-(self.window_size - 1):]
        else:
            prior = []
        expanded = " ".join([query] + prior)

        vocab = self.vectorizer.vocabulary_
        tokens = self.vectorizer.build_analyzer()(expanded)
        q_terms = [t for t in tokens if t in vocab]
        if not q_terms:
            return []
        uniq, qf = np.unique([vocab[t] for t in q_terms], return_counts=True)
        qf = qf.astype(np.float64)

        # Column slice of the doc-term frequency matrix for the query terms.
        f = self.doc_counts[:, uniq].astype(np.float64)
        rows = np.repeat(np.arange(f.shape[0]), np.diff(f.indptr))
        cols = f.indices

        k1, b = self.k1, self.b
        c = k1 * (1.0 - b + b * self.doc_len / self.avgdl)
        numer = f.data * (k1 + 1.0) * self.idf[uniq][cols] * qf[cols]
        denom = f.data + c[rows]
        term_scores = numer / denom

        scores = np.bincount(rows, weights=term_scores, minlength=len(self.corpus))
        ranked = np.argsort(scores)[::-1]
        return [(self.note_ids[i], float(scores[i])) for i in ranked[:top_k]]

    def reset_session(self):
        pass
