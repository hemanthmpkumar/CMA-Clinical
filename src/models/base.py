#!/usr/bin/env python3
"""Base retriever interface for the clinical search benchmark."""

from abc import ABC, abstractmethod
from typing import Optional


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
