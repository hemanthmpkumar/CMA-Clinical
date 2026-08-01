#!/usr/bin/env python3
"""
src/models/cma.py

Continuum Memory Architecture (CMA) retriever.

The model:
  1. Encodes queries/documents with TF-IDF, then maps the TF-IDF vector to a
     Symmetric Positive Definite (SPD) matrix through a trainable neural encoder.
     The SPD matrix is lifted to its log-Euclidean tangent vector, giving the
     latent representation used for retrieval, curvature gating, and JEPA
     prediction (replacing the previous flat TruncatedSVD Euclidean projection).
  2. Aggregates recent session latent vectors into a session intent vector.
  3. Detects abrupt topic shifts with a geodesic-shift-interference (GSI) gate.
  4. Forecasts the next query with a globally trained JEPA-style latent predictor
     (Stiefel MLP + slow EMA target network + stop-gradient).
"""

import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseRetriever
from .gsi_gate import GSIGate
from .jepa import JEPAPredictor
from .spd_encoder import SPDEncoder


SPD_DIM = 16  # yields n_latent = 136 log-Euclidean coordinates


class CMARetriever(BaseRetriever):
    def __init__(self, corpus: list[dict],
                 curvature_threshold: float = 0.65,
                 gate_discount: float = 0.05,
                 context_window: int = 5,
                 prefetch_weight: float = 0.4,
                 spd_dim: int = SPD_DIM,
                 encoder_hidden_dim: int = 512,
                 encoder_pretrain_epochs: int = 50,
                 encoder_finetune_epochs: int = 100,
                 encoder_lr: float = 1e-3,
                 predictor: Optional[JEPAPredictor] = None,
                 vectorizer: Optional[TfidfVectorizer] = None,
                 encoder: Optional[SPDEncoder] = None,
                 doc_latent: Optional[np.ndarray] = None,
                 seed: int = 42):
        super().__init__(corpus)
        self.curvature_threshold = curvature_threshold
        self.gate_discount = gate_discount
        self.context_window = context_window
        self.prefetch_weight = prefetch_weight
        self.encoder_pretrain_epochs = encoder_pretrain_epochs
        self.encoder_finetune_epochs = encoder_finetune_epochs

        # Suppress noisy BLAS/numpy warnings on some Apple-Silicon builds.
        np.seterr(divide="ignore", over="ignore", invalid="ignore")
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        doc_texts = [rec["text"] for rec in corpus]

        # Encode corpus with TF-IDF. Allow reusing a pre-fitted vectorizer
        # so hyper-parameter searches do not refit the vocabulary.
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_df=0.85, min_df=2, stop_words="english", max_features=4000,
                sublinear_tf=True
            )
            doc_tfidf = self.vectorizer.fit_transform(doc_texts)
        else:
            self.vectorizer = vectorizer
            doc_tfidf = self.vectorizer.transform(doc_texts)

        if encoder is None:
            n_features = doc_tfidf.shape[1]
            print(f"  Training neural SPD encoder (TF-IDF dim={n_features}, "
                  f"SPD dim={spd_dim})...")
            self.encoder = SPDEncoder(
                input_dim=n_features,
                hidden_dim=encoder_hidden_dim,
                spd_dim=spd_dim,
                device="cpu",
            )
            if encoder_pretrain_epochs > 0:
                self.encoder.fit(doc_tfidf, epochs=encoder_pretrain_epochs,
                                  batch_size=256, lr=encoder_lr, seed=seed)
        else:
            self.encoder = encoder

        self.spd_dim = self.encoder.spd_dim
        self.latent_dim = self.encoder.n_latent

        if doc_latent is None:
            with torch.no_grad():
                doc_tfidf_torch = torch.tensor(doc_tfidf.toarray(), dtype=torch.float32)
                self.doc_latent = self.encoder.encode_to_log_vec(doc_tfidf_torch).cpu().numpy()
        else:
            self.doc_latent = doc_latent

        if self.doc_latent.shape[0] < 2:
            raise ValueError(f"Corpus too small: only {self.doc_latent.shape[0]} documents.")

        # Globally trained JEPA predictor (fitted once on historical sessions).
        self.predictor = predictor

        # GSI gate for curvature-aware context suppression.
        self.gsi_gate = GSIGate(
            spd_dim=self.spd_dim,
            curvature_threshold=curvature_threshold,
            gate_discount=gate_discount,
        )

        # Per-session state.
        self.session_latents: list[np.ndarray] = []
        self.session_weights: list[float] = []

    # ─────────────────────────── Latent encoding ─────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        q_tfidf = self.vectorizer.transform([query])
        return self.encoder.encode_to_log_vec(q_tfidf)[0].cpu().numpy()

    # ─────────────────────────── Intent aggregation ─────────────────────

    def _current_intent(self) -> np.ndarray:
        if not self.session_latents:
            return np.zeros(self.latent_dim)
        latents = np.stack(self.session_latents[-self.context_window:], axis=0)
        weights = np.array(self.session_weights[-self.context_window:], dtype=float)
        wsum = weights.sum()
        if wsum <= 0:
            return np.zeros(self.latent_dim)
        return (latents * (weights / wsum)[:, None]).sum(axis=0)

    # ─────────────────────── GSI gate ───────────────────────

    def _apply_gate(self) -> bool:
        weights, triggered = self.gsi_gate.apply(self.session_latents, self.session_weights)
        self.session_weights = weights
        return triggered

    # ─────────────────────────── JEPA Predictor training ────────────────────

    def fit_predictor(self, vignettes: list[dict], epochs: int = 120,
                       batch_size: int = 64) -> "CMARetriever":
        """Fine-tune the SPD encoder on query-target pairs and train the JEPA
        predictor on query transitions from vignettes."""

        # -------------------------------------------------------------------
        # 1. Fine-tune the SPD encoder with query-target contrastive loss.
        # This pulls target notes closer to their queries in log-SPD space
        # than random corpus documents.
        # -------------------------------------------------------------------
        note_text = {rec["note_id"]: rec["text"] for rec in self.corpus}
        query_texts = []
        target_texts = []
        for v in vignettes:
            for q in v.get("queries", []):
                t = q.get("target_note_id")
                if t and t in note_text:
                    query_texts.append(q["text"])
                    target_texts.append(note_text[t])

        if query_texts:
            queries_tfidf = self.vectorizer.transform(query_texts)
            positives_tfidf = self.vectorizer.transform(target_texts)
            doc_texts = [rec["text"] for rec in self.corpus]
            corpus_tfidf = self.vectorizer.transform(doc_texts)
            self.encoder.fit_retrieval(
                queries_tfidf, positives_tfidf, corpus_tfidf,
                epochs=self.encoder_finetune_epochs, n_negatives=10,
                batch_size=64, lr=1e-3, seed=42
            )
            # Recompute document latents with the fine-tuned encoder.
            with torch.no_grad():
                doc_tfidf_torch = torch.tensor(corpus_tfidf.toarray(), dtype=torch.float32)
                self.doc_latent = self.encoder.encode_to_log_vec(doc_tfidf_torch).cpu().numpy()

        # -------------------------------------------------------------------
        # 2. Train the JEPA predictor on consecutive query latents.
        # -------------------------------------------------------------------
        X, Y = [], []
        for v in vignettes:
            queries = v.get("queries", [])
            if len(queries) < 2:
                continue
            latents = [self._encode_query(q["text"]) for q in queries]
            for z_t, z_tp1 in zip(latents[:-1], latents[1:]):
                X.append(z_t)
                Y.append(z_tp1)

        n_transitions = len(X)
        if n_transitions == 0:
            print("  Warning: no query transitions found; JEPA predictor disabled.")
            self.predictor = None
            return self

        print(f"  Training JEPA predictor on {n_transitions} query transitions "
              f"({self.latent_dim}-D log-SPD latents)...")
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)

        self.predictor = JEPAPredictor(
            latent_dim=self.latent_dim,
            hidden_dim=max(64, self.latent_dim * 2),
            spd_dim=self.spd_dim,
            seed=42,
        ).fit(X, Y, epochs=epochs, batch_size=batch_size)
        return self

    def _predict_next_latent(self) -> Optional[np.ndarray]:
        if self.predictor is None or len(self.session_latents) == 0:
            return None
        z = self.session_latents[-1].reshape(1, -1)
        return self.predictor.predict(z).ravel()

    # ─────────────────────────── Retrieval interface ─────────────────────────

    def search(self, query: str, session_history: list[str], top_k: int = 10,
               prefetch: bool = True, **kwargs) -> list[tuple[str, float]]:
        q_vec = self._encode_query(query)
        self.session_latents.append(q_vec)
        self.session_weights.append(1.0)

        self._apply_gate()

        intent = q_vec + self._current_intent()

        scores = self.doc_latent @ intent
        scores = np.asarray(scores).ravel()

        if prefetch:
            z_next = self._predict_next_latent()
            if z_next is not None:
                scores += self.prefetch_weight * (self.doc_latent @ z_next)

        ranked = np.argsort(scores)[::-1]
        return [(self.note_ids[i], float(scores[i])) for i in ranked[:top_k]]

    def reset_session(self):
        self.session_latents = []
        self.session_weights = []

    def copy_with_hyperparams(self, **overrides) -> "CMARetriever":
        """Return a new retriever sharing the fitted TF-IDF/SPD encoder/JEPA
        components but with overridden gate/search hyper-parameters."""
        params = {
            "corpus": self.corpus,
            "curvature_threshold": self.curvature_threshold,
            "gate_discount": self.gate_discount,
            "context_window": self.context_window,
            "prefetch_weight": self.prefetch_weight,
            "spd_dim": self.spd_dim,
            "encoder_hidden_dim": 512,
            "encoder_pretrain_epochs": 0,  # already trained; skip retraining
            "encoder_finetune_epochs": 0,
            "encoder_lr": 1e-3,
            "predictor": self.predictor,
            "vectorizer": self.vectorizer,
            "encoder": self.encoder,
            "doc_latent": self.doc_latent,
            "seed": 42,
        }
        params.update(overrides)
        return CMARetriever(**params)
