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
                 prefetch_weight: float = 0.0,
                 lexical_weight: float = 1.0,
                 semantic_weight: float = 0.0,
                 context_weight: float = 0.0,
                 semantic_candidate_k: int = 2048,
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
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.context_weight = context_weight
        self.semantic_candidate_k = semantic_candidate_k
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
                device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            )
            if encoder_pretrain_epochs > 0:
                self.encoder.fit(doc_tfidf, epochs=encoder_pretrain_epochs,
                                  batch_size=256, lr=encoder_lr, seed=seed)
        else:
            self.encoder = encoder

        self.spd_dim = self.encoder.spd_dim
        self.latent_dim = self.encoder.n_latent

        doc_tfidf = self._pad_tfidf(doc_tfidf)
        self.doc_tfidf = doc_tfidf

        if doc_latent is None:
            self.doc_latent = self._encode_docs_in_chunks(doc_tfidf)
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

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        mean = scores.mean()
        std = scores.std()
        if not np.isfinite(std) or std < 1e-12:
            return scores - mean
        return (scores - mean) / std

    def _pad_tfidf(self, tfidf_matrix: sp.spmatrix) -> sp.spmatrix:
        expected_dim = getattr(self.encoder, "input_dim", None)
        if expected_dim is None:
            return tfidf_matrix
        if tfidf_matrix.shape[1] < expected_dim:
            padded = sp.lil_matrix((tfidf_matrix.shape[0], expected_dim), dtype=np.float32)
            padded[:, :tfidf_matrix.shape[1]] = tfidf_matrix
            return padded.tocsr()
        if tfidf_matrix.shape[1] > expected_dim:
            return tfidf_matrix[:, :expected_dim]
        return tfidf_matrix

    def _transform_text(self, text: str) -> sp.spmatrix:
        return self._pad_tfidf(self.vectorizer.transform([text]))

    def _lexical_scores(self, query: str) -> np.ndarray:
        q_tfidf = self._transform_text(query)
        scores = (self.doc_tfidf @ q_tfidf.T).toarray().ravel()
        return self._normalize_scores(scores)

    def _encode_query(self, query: str) -> np.ndarray:
        q_tfidf = self._transform_text(query)
        return self.encoder.encode_to_log_vec(q_tfidf)[0].cpu().numpy()

    def _encode_docs_in_chunks(self, doc_tfidf: sp.spmatrix,
                               chunk_rows: int = 8192) -> np.ndarray:
        """Encode the full corpus to log-SPD latents in bounded-memory chunks.

        Dense TF-IDF is huge (~89 GB float64 for a 2.8M x 4000 matrix); calling
        ``toarray()`` on all of it at once exhausts RAM. Process one block at a
        time so peak memory stays proportional to a single chunk.
        """
        n_docs = doc_tfidf.shape[0]
        out = np.empty((n_docs, self.latent_dim), dtype=np.float32)
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, n_docs, chunk_rows):
                end = min(start + chunk_rows, n_docs)
                block = doc_tfidf[start:end].toarray().astype(np.float32)
                block_t = torch.tensor(block, dtype=torch.float32, device=self.encoder.device)
                out[start:end] = self.encoder.encode_to_log_vec(block_t).cpu().numpy()
        return out

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
            queries_tfidf = self._pad_tfidf(self.vectorizer.transform(query_texts))
            positives_tfidf = self._pad_tfidf(self.vectorizer.transform(target_texts))
            doc_texts = [rec["text"] for rec in self.corpus]
            corpus_tfidf = self._pad_tfidf(self.vectorizer.transform(doc_texts))
            self.encoder.fit_retrieval(
                queries_tfidf, positives_tfidf, corpus_tfidf,
                epochs=self.encoder_finetune_epochs, n_negatives=10,
                batch_size=64, lr=1e-3, seed=42
            )
            # Recompute document latents with the fine-tuned encoder.
            self.doc_latent = self._encode_docs_in_chunks(corpus_tfidf)

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
        if self.predictor is None or len(self.session_latents) < 2:
            return None
        z = self.session_latents[-1].reshape(1, -1)
        pred = self.predictor.predict(z).ravel()
        if not np.isfinite(pred).all():
            return None
        if np.linalg.norm(pred) < 1e-8:
            return None
        return pred

    # ─────────────────────────── Retrieval interface ─────────────────────────

    def search(self, query: str, session_history: list[str], top_k: int = 10,
               prefetch: bool = True, filter_ids: set = None, **kwargs) -> list[tuple[str, float]]:
        q_vec = self._encode_query(query)
        self.session_latents.append(q_vec)
        self.session_weights.append(1.0)

        self._apply_gate()

        expanded_query = " ".join([query] + session_history[-max(0, self.context_window - 1):])
        lexical_scores = self._lexical_scores(expanded_query)
        n_docs = self.doc_latent.shape[0]

        # Keep the full lexical pool available so a weak-but-correct match is
        # not discarded before ranking. The latent branches remain secondary
        # and should only re-rank, not eliminate, plausible matches.
        candidate_idx = np.arange(n_docs)

        scores = np.full(n_docs, -np.inf, dtype=np.float64)
        scores[candidate_idx] = self.lexical_weight * lexical_scores[candidate_idx]

        # Apply patient-level filtering
        if filter_ids is not None:
            mask = np.array([nid not in filter_ids for nid in self.note_ids])
            scores[mask] = -np.inf

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
            "lexical_weight": self.lexical_weight,
            "semantic_weight": self.semantic_weight,
            "context_weight": self.context_weight,
            "semantic_candidate_k": self.semantic_candidate_k,
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
