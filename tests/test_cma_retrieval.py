import unittest

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.cma import CMARetriever


class DummyEncoder:
    def __init__(self):
        self.spd_dim = 2
        self.n_latent = 3
        self.device = torch.device("cpu")

    def eval(self):
        return None

    def train(self):
        return None

    def fit(self, *args, **kwargs):
        return self

    def encode_to_log_vec(self, x):
        if hasattr(x, "shape") and len(x.shape) > 1:
            batch = x.shape[0]
            return torch.tensor(np.tile([[0.0, 1.0, 0.0]], (batch, 1)), dtype=torch.float32)
        return torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)


class NoisyPredictor:
    def predict(self, x):
        return np.zeros_like(np.asarray(x), dtype=np.float32)


class CMARetrievalTests(unittest.TestCase):
    def test_cma_search_keeps_lexical_matches_when_semantic_signal_is_noisy(self):
        corpus = [
            {"note_id": "doc_pneumonia", "text": "pneumonia treatment guidelines for clinicians"},
            {"note_id": "doc_surgery", "text": "surgical consultation and operative planning"},
            {"note_id": "doc_shock", "text": "shock and vasopressor dosing guidance"},
        ]
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words="english")
        vectorizer.fit([c["text"] for c in corpus])
        doc_latent = np.array([
            [0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.2, 0.0],
        ], dtype=np.float32)

        retriever = CMARetriever(
            corpus,
            vectorizer=vectorizer,
            encoder=DummyEncoder(),
            doc_latent=doc_latent,
            encoder_pretrain_epochs=0,
            encoder_finetune_epochs=0,
            curvature_threshold=float("inf"),
            prefetch_weight=0.0,
        )

        results = retriever.search("pneumonia treatment", session_history=[], top_k=3, prefetch=False)

        self.assertEqual(results[0][0], "doc_pneumonia")

    def test_cma_search_ignores_noisy_prefetch_signal(self):
        corpus = [
            {"note_id": "doc_pneumonia", "text": "pneumonia treatment guidelines for clinicians"},
            {"note_id": "doc_surgery", "text": "surgical consultation and operative planning"},
            {"note_id": "doc_shock", "text": "shock and vasopressor dosing guidance"},
        ]
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words="english")
        vectorizer.fit([c["text"] for c in corpus])
        doc_latent = np.array([
            [0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.2, 0.0],
        ], dtype=np.float32)

        retriever = CMARetriever(
            corpus,
            vectorizer=vectorizer,
            encoder=DummyEncoder(),
            doc_latent=doc_latent,
            encoder_pretrain_epochs=0,
            encoder_finetune_epochs=0,
            curvature_threshold=float("inf"),
            prefetch_weight=0.4,
            predictor=NoisyPredictor(),
        )

        results = retriever.search("pneumonia treatment", session_history=["fever"], top_k=3, prefetch=True)

        self.assertEqual(results[0][0], "doc_pneumonia")

    def test_cma_search_keeps_a_relevant_lexical_match_outside_a_small_candidate_pool(self):
        corpus = [
            {"note_id": "doc_high1", "text": "pneumonia treatment guidelines"},
            {"note_id": "doc_high2", "text": "pneumonia treatment for adults"},
            {"note_id": "doc_high3", "text": "pneumonia treatment update"},
            {"note_id": "doc_target", "text": "pneumonia management support"},
            {"note_id": "doc_low", "text": "common cold symptoms"},
        ]
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, stop_words="english")
        vectorizer.fit([c["text"] for c in corpus])
        doc_latent = np.array([
            [0.2, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.1, 0.2, 0.0],
            [0.3, 0.1, 0.0],
            [0.0, 0.0, 0.6],
        ], dtype=np.float32)

        retriever = CMARetriever(
            corpus,
            vectorizer=vectorizer,
            encoder=DummyEncoder(),
            doc_latent=doc_latent,
            encoder_pretrain_epochs=0,
            encoder_finetune_epochs=0,
            curvature_threshold=float("inf"),
            semantic_candidate_k=3,
            prefetch_weight=0.0,
        )

        results = retriever.search("pneumonia treatment", session_history=[], top_k=5, prefetch=False)
        result_ids = [note_id for note_id, _ in results]

        self.assertIn("doc_target", result_ids)


if __name__ == "__main__":
    unittest.main()
