#!/usr/bin/env python3
"""
src/models/baseline.py

Standard session-based retrieval.

The baseline encodes the current query together with recent session queries
(uniform weighting, no pivot detection) in a single TF-IDF vector and scores
documents by cosine similarity. This models a conventional search UI that is
vulnerable to stale-context interference after abrupt topic shifts.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseRetriever


class BaselineRetriever(BaseRetriever):
    def __init__(self, corpus: list[dict], window_size: int = 20):
        super().__init__(corpus)
        self.window_size = window_size

        doc_texts = [rec["text"] for rec in corpus]
        self.vectorizer = TfidfVectorizer(
            max_df=0.85, min_df=2, stop_words="english", max_features=4000,
            sublinear_tf=True
        )
        self.doc_tfidf = self.vectorizer.fit_transform(doc_texts)

    def search(self, query: str, session_history: list[str], top_k: int = 10,
               filter_ids: set = None, **kwargs) -> list[tuple[str, float]]:
        if self.window_size > 1:
            prior = session_history[-(self.window_size - 1):]
        else:
            prior = []
        expanded = " ".join([query] + prior)
        q_vec = self.vectorizer.transform([expanded])
        scores = (self.doc_tfidf @ q_vec.T).toarray().ravel()
        
        # Apply patient-level filtering
        if filter_ids is not None:
            mask = np.array([nid not in filter_ids for nid in self.note_ids])
            scores[mask] = -np.inf

        ranked = np.argsort(scores)[::-1]
        return [(self.note_ids[i], float(scores[i])) for i in ranked[:top_k]]

    def reset_session(self):
        pass
