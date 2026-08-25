#!/usr/bin/env python3
"""Base retriever interface for the clinical search benchmark."""

from abc import ABC, abstractmethod
from typing import Optional


def build_tfidf(doc_texts: list[str], **overrides):
    """Construct and fit a bounded TF-IDF vectorizer over doc_texts.

    The default parameters (min_df=2, max_df=0.85) assume a reasonably large
    vocabulary. Tiny scale-study corpora (e.g. 3-vignette scales backed by
    only ~120 records) can prune every term, so we fall back to min_df=1.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    params = dict(
        max_df=0.85, min_df=2, stop_words="english",
        max_features=4000, sublinear_tf=True,
    )
    params.update(overrides)
    try:
        vec = TfidfVectorizer(**params)
        return vec.fit(doc_texts)
    except ValueError as exc:
        if "no terms remain" not in str(exc):
            raise
        # Tiny scale-study corpora can prune every term: relax the document
        # frequency bounds that caused the empty vocabulary.
        if params["min_df"] > 1:
            params["min_df"] = 1
        if params["max_df"] < 1.0:
            params["max_df"] = 1.0
        vec = TfidfVectorizer(**params)
        return vec.fit(doc_texts)


class BaseRetriever(ABC):
    def __init__(self, corpus: list[dict]):
        """
        Args:
            corpus: list of dicts with keys 'note_id', 'text', and optional metadata.
        """
        self.corpus = corpus
        self.note_ids = [rec["note_id"] for rec in corpus]

    @abstractmethod
    def search(self, query: str, session_history: list[str], top_k: int = 10,
               filter_ids: Optional[set[str]] = None, **kwargs) -> list[tuple[str, float]]:
        raise NotImplementedError

    def reset_session(self):
        """Optional hook to clear any per-session state."""
        pass
