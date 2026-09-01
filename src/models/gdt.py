#!/usr/bin/env python3
"""
src/models/gdt.py

Geodesic Diagnostic Trajectories (GDT) retriever.

GDT models the reviewer's diagnostic intent as a continuous trajectory on the
Riemannian manifold of symmetric positive definite (SPD) matrices S_d^+. It
shares the learned latent machinery with the CMA retriever (TF-IDF -> SPD
log-Euclidean encoder, JEPA next-intent predictor) but differs in how session
context is managed and scored:

  1. Diagnostic state space. Each turn's intent is the SPD matrix
     S_t = expm(vec^{-1}(e_t)) reconstructed from the log-Euclidean latent e_t.
     Geodesic displacement is measured with the affine-invariant metric
     g_S(u,v) = Tr(S^{-1} u S^{-1} v), whose associated distance is
     d_g(S_a,S_b) = || log(S_a^{-1/2} S_b S_a^{-1/2}) ||_F.

  2. Geodesic Shift Interference (GSI) gate. The instantaneous geodesic shift
     ratio (diagnostic curvature)
         kappa_t = 2 d_g(S_{t-1},S_t) / (d_g(S_{t-2},S_{t-1}) + d_g(S_{t-1},S_t) + eps)
     flags an abrupt topic pivot (kappa_t -> 2) versus steady refinement
     (kappa_t -> 0). A soft gate g_t = sigmoid((kappa_t - kappa_0)/tau) blends
     the fresh query latent e_t with the smoothed history latent h_{t-1}:
         e_t~ = (1 - g_t) h_{t-1} + g_t e_t,
     and curvature-gated lexical context attenuation excludes stale prior query
     terms from the lexical expansion once the gate is open. Proposition 1 of
     the companion manuscript bounds the stale-context interference
     ||P_xi e_t~|| <= (1-g_t)||P_xi h_{t-1}|| + g_t||P_xi e_t||.

  3. Optimal-control prefetch engine. A JEPA-style predictor f_theta forecasts
     the next latent state and the top-K_pf documents are prefetched by cosine
     similarity to the forecast, scaled by a confidence term that decays after
     abrupt pivots (uncertain forecasts fetch fewer documents), implementing a
     cognitive budget on prefetch volume.
"""

from typing import Optional

import numpy as np

from .cma import CMARetriever
from .gsi_gate import GSIGate


class GDTRetriever(CMARetriever):
    """Geodesic Diagnostic Trajectories retriever (GDT framework).

    Extends the CMA retriever with the GDT context-management mechanism: a soft
    sigmoid gate driven by the geodesic shift ratio kappa_t, gated semantic
    retrieval on the SPD manifold, and confidence-scaled optimal-control
    prefetching.
    """

    def __init__(self, corpus: list[dict],
                 curvature_threshold: float = 1.0,
                 gate_temperature: float = 0.5,
                 gate_lexical_include: float = 0.75,
                 history_beta: float = 0.7,
                 semantic_weight: float = 0.15,
                 prefetch_confidence_scale: float = 2.0,
                 **kwargs):
        # curvature_threshold doubles as the learned kappa_0 of the soft gate.
        super().__init__(corpus, curvature_threshold=curvature_threshold, **kwargs)
        self.gate_temperature = gate_temperature
        self.gate_lexical_include = gate_lexical_include
        self.history_beta = history_beta
        self.semantic_weight = semantic_weight
        self.prefetch_confidence_scale = prefetch_confidence_scale

        # Per-session gate state (in addition to CMA's session_latents/weights).
        self._history: Optional[np.ndarray] = None
        self.last_kappa: float = 1.0
        self.last_gate: float = 0.5
        self.gate_triggers: int = 0

    # ─────────────────────────── Geodesic geometry ──────────────────────────

    @staticmethod
    def _to_spd(vec: np.ndarray, spd_dim: int) -> np.ndarray:
        """Reconstruct an SPD matrix from its log-Euclidean vectorisation.

        vec is the upper-triangular (column-major) vector of log(S); we rebuild
        the symmetric matrix and exponentiate it back to S_d^+.
        """
        sym = GSIGate.vec_to_sym_matrix(np.asarray(vec, dtype=float), spd_dim)
        return GSIGate.symmetric_matrix_exp(sym)

    def _geodesic_shift_ratio(self) -> float:
        """Geodesic shift ratio kappa_t = 2 d_{t-1,t} / (d_{t-2,t-1}+d_{t-1,t}+eps).

        Returns 1.0 (steady pace) until three consecutive intent states are
        available. kappa_t -> 2 indicates an abrupt topic pivot; kappa_t -> 0
        indicates refinement within a topic.
        """
        if len(self.session_latents) < 3:
            return 1.0
        recent = self.session_latents[-3:]
        spd = [self._to_spd(v, self.spd_dim) for v in recent]
        d_prevprev = GSIGate.affine_invariant_distance(spd[0], spd[1])
        d_prev = GSIGate.affine_invariant_distance(spd[1], spd[2])
        denom = d_prevprev + d_prev
        if denom < 1e-12:
            return 1.0
        return float(2.0 * d_prev / denom)

    def _soft_gate(self, kappa: float) -> float:
        """Soft sigmoid gate g_t = sigmoid((kappa_t - kappa_0)/tau)."""
        if self.gate_temperature <= 0:
            return float(kappa > self.curvature_threshold)
        return float(1.0 / (1.0 + np.exp(-(kappa - self.curvature_threshold)
                                         / self.gate_temperature)))

    def _predict_confidence(self, kappa: float) -> float:
        """Forecast confidence p(s_hat_{t+1}): decays after abrupt pivots.

        After a large geodesic jump the next intent is uncertain, so fewer
        documents should be prefetched (cognitive budget constraint).
        """
        return float(1.0 / (1.0 + np.exp(self.prefetch_confidence_scale * (kappa - 1.0))))

    # ─────────────────────────── Contamination metric ───────────────────────

    def stale_interference(self, stale_subspace: np.ndarray) -> Optional[float]:
        """Return ||P_xi e_t~|| (norm of stale interference in the gated embedding).

        stale_subspace is a (d,) vector whose span defines the stale-topic
        subspace xi. Requires a history embedding to exist; otherwise None.
        """
        if self._history is None or len(self.session_latents) == 0:
            return None
        e_t = self.session_latents[-1]
        stale = np.asarray(stale_subspace, dtype=float)
        norm = np.linalg.norm(stale)
        if norm < 1e-12:
            return 0.0
        P_xi = np.outer(stale, stale) / (norm * norm)
        gated = (1.0 - self.last_gate) * self._history + self.last_gate * e_t
        return float(np.linalg.norm(P_xi @ gated))

    def contamination_bound(self, stale_subspace: np.ndarray) -> Optional[dict]:
        """Evaluate Proposition 1: bound vs actual interference for the gate.

        Returns a dict with the measured interference ||P_xi e_t~|| and the
        Proposition-1 upper bound (1-g_t)||P_xi h_{t-1}|| + g_t||P_xi e_t||,
        or None if no history embedding exists yet.
        """
        if self._history is None or len(self.session_latents) == 0:
            return None
        e_t = self.session_latents[-1]
        stale = np.asarray(stale_subspace, dtype=float)
        norm = np.linalg.norm(stale)
        if norm < 1e-12:
            return {"interference": 0.0, "bound": 0.0}
        P_xi = np.outer(stale, stale) / (norm * norm)
        gated = (1.0 - self.last_gate) * self._history + self.last_gate * e_t
        interference = float(np.linalg.norm(P_xi @ gated))
        bound = float((1.0 - self.last_gate) * np.linalg.norm(P_xi @ self._history)
                      + self.last_gate * np.linalg.norm(P_xi @ e_t))
        return {"interference": interference, "bound": bound,
                "gate": self.last_gate, "kappa": self.last_kappa}

    # ─────────────────────────── Retrieval interface ────────────────────────

    def search(self, query: str, session_history: list[str], top_k: int = 10,
               prefetch: bool = True, filter_ids: set = None, **kwargs) -> list[tuple[str, float]]:
        # 1. Update the diagnostic trajectory with the new intent state.
        e_t = self._encode_query(query)
        self.session_latents.append(e_t)
        self.session_weights.append(1.0)

        # 2. Geodesic shift ratio and soft gate.
        kappa = self._geodesic_shift_ratio()
        g_t = self._soft_gate(kappa)
        self.last_kappa = kappa
        self.last_gate = g_t
        if g_t > 0.5 and len(self.session_latents) >= 3:
            self.gate_triggers += 1

        # 3. Exponentially smoothed history embedding.
        h_prev = self._history
        if h_prev is None:
            h_prev = np.zeros_like(e_t)
        self._history = (self.history_beta * h_prev
                         + (1.0 - self.history_beta) * e_t)

        # 4. Curvature-gated lexical context attenuation. When the gate is open
        #    (abrupt pivot), stale prior query terms are excluded from the
        #    lexical expansion so they cannot contaminate ranking.
        if g_t >= self.gate_lexical_include:
            expanded = query
        else:
            prior = session_history[-max(0, self.context_window - 1):]
            expanded = " ".join([query] + prior)
        lexical_scores = self._lexical_scores(expanded)
        n_docs = self.doc_latent.shape[0]

        scores = self.lexical_weight * lexical_scores

        # 5. Gated semantic retrieval on the SPD manifold: score by cosine
        #    between the gated embedding e_t~ and the log-SPD doc latents.
        if self.semantic_weight > 0:
            gated = (1.0 - g_t) * h_prev + g_t * e_t
            gated_n = gated / (np.linalg.norm(gated) + 1e-9)
            scores = scores + self.semantic_weight * (self.doc_latent @ gated_n)

        # 6. Optimal-control prefetch: JEPA forecast, confidence-scaled.
        if prefetch and self.predictor is not None and len(self.session_latents) >= 2:
            forecast = self._predict_next_latent()
            if forecast is not None:
                conf = self._predict_confidence(kappa)
                f = forecast / (np.linalg.norm(forecast) + 1e-9)
                scores = scores + self.prefetch_weight * conf * (self.doc_latent @ f)

        # 7. Patient-level filtering and ranking.
        if filter_ids is not None:
            mask = np.array([nid not in filter_ids for nid in self.note_ids])
            scores[mask] = -np.inf

        ranked = np.argsort(scores)[::-1]
        return [(self.note_ids[i], float(scores[i])) for i in ranked[:top_k]]

    def reset_session(self):
        super().reset_session()
        self._history = None
        self.last_kappa = 1.0
        self.last_gate = 0.5
        self.gate_triggers = 0

    def copy_with_hyperparams(self, **overrides) -> "GDTRetriever":
        """Return a new GDT retriever sharing the fitted TF-IDF/SPD encoder/JEPA
        components but with overridden gate/search hyper-parameters."""
        params = {
            "corpus": self.corpus,
            "curvature_threshold": self.curvature_threshold,
            "gate_temperature": self.gate_temperature,
            "gate_lexical_include": self.gate_lexical_include,
            "history_beta": self.history_beta,
            "semantic_weight": self.semantic_weight,
            "prefetch_confidence_scale": self.prefetch_confidence_scale,
            "context_window": self.context_window,
            "prefetch_weight": self.prefetch_weight,
            "lexical_weight": self.lexical_weight,
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
        return GDTRetriever(**params)
