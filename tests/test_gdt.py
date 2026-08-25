import unittest

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.gdt import GDTRetriever
from src.models.gsi_gate import GSIGate


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


def make_retriever(corpus, **kwargs):
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words="english")
    vectorizer.fit([c["text"] for c in corpus])
    doc_latent = np.array([
        [0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.2, 0.0],
    ], dtype=np.float32)
    defaults = dict(
        vectorizer=vectorizer,
        encoder=DummyEncoder(),
        doc_latent=doc_latent,
        encoder_pretrain_epochs=0,
        encoder_finetune_epochs=0,
        prefetch_weight=0.0,
    )
    defaults.update(kwargs)
    return GDTRetriever(corpus, **defaults)


class GDTGeodesicTests(unittest.TestCase):
    def test_geodesic_shift_ratio_steady_steps(self):
        # Equally spaced intent states -> kappa_t ~ 1.0 (steady pace).
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        r.session_latents = [np.array([0.0, 0.0, 0.0]),
                             np.array([1.0, 0.0, 0.0]),
                             np.array([2.0, 0.0, 0.0])]
        kappa = r._geodesic_shift_ratio()
        self.assertAlmostEqual(kappa, 1.0, places=3)

    def test_geodesic_shift_ratio_abrupt_pivot(self):
        # A much larger final step -> kappa_t -> 2 (abrupt pivot).
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        r.session_latents = [np.array([0.0, 0.0, 0.0]),
                             np.array([0.5, 0.0, 0.0]),
                             np.array([4.0, 0.0, 0.0])]
        kappa = r._geodesic_shift_ratio()
        self.assertGreater(kappa, 1.5)

    def test_geodesic_shift_ratio_needs_three_states(self):
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        r.session_latents = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        self.assertEqual(r._geodesic_shift_ratio(), 1.0)

    def test_soft_gate_monotonic_in_kappa(self):
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        g0 = r._soft_gate(0.0)
        g1 = r._soft_gate(1.0)
        g2 = r._soft_gate(2.0)
        self.assertLess(g0, g1)
        self.assertLess(g1, g2)
        self.assertLess(g2, 1.0 + 1e-9)

    def test_predict_confidence_decays_with_pivot(self):
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        self.assertGreater(r._predict_confidence(0.5), r._predict_confidence(1.0))
        self.assertGreater(r._predict_confidence(1.0), r._predict_confidence(2.0))


class GDTContaminationTests(unittest.TestCase):
    def test_proposition1_bound_holds(self):
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        stale = np.array([1.0, 0.0, 0.0])          # stale-topic subspace xi
        e_t = np.array([0.2, 1.0, 0.0])            # fresh query latent
        h_prev = np.array([2.0, 0.5, 0.0])         # contaminated history
        g_t = 0.9
        r.session_latents = [e_t]
        r._history = h_prev
        r.last_gate = g_t

        bound = r.contamination_bound(stale)
        self.assertIsNotNone(bound)
        self.assertLessEqual(bound["interference"], bound["bound"] + 1e-9)

        # Manual Proposition-1 check.
        P_xi = np.outer(stale, stale) / np.dot(stale, stale)
        gated = (1 - g_t) * h_prev + g_t * e_t
        self.assertAlmostEqual(bound["interference"], float(np.linalg.norm(P_xi @ gated)), places=6)

    def test_interference_decreases_as_gate_opens(self):
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        stale = np.array([1.0, 0.0, 0.0])
        e_t = np.array([0.2, 1.0, 0.0])
        h_prev = np.array([2.0, 0.5, 0.0])
        P_xi = np.outer(stale, stale) / np.dot(stale, stale)

        gated_low = (1 - 0.1) * h_prev + 0.1 * e_t
        gated_high = (1 - 0.95) * h_prev + 0.95 * e_t
        self.assertGreater(np.linalg.norm(P_xi @ gated_low),
                           np.linalg.norm(P_xi @ gated_high))

        # ... and the retriever's metric reflects it.
        r.session_latents = [e_t]
        r._history = h_prev
        r.last_gate = 0.1
        low = r.stale_interference(stale)
        r.last_gate = 0.95
        high = r.stale_interference(stale)
        self.assertGreater(low, high)

    def test_stale_interference_none_without_history(self):
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        r.session_latents = [np.array([0.2, 1.0, 0.0])]
        self.assertIsNone(r.stale_interference(np.array([1.0, 0.0, 0.0])))


class GDTRetrievalTests(unittest.TestCase):
    def test_search_returns_ranked_docs(self):
        corpus = [
            {"note_id": "d1", "text": "pneumonia treatment guidelines"},
            {"note_id": "d2", "text": "surgical consultation and operative planning"},
            {"note_id": "d3", "text": "shock and vasopressor dosing guidance"},
        ]
        r = make_retriever(corpus, semantic_weight=0.2)
        results = r.search("pneumonia treatment", session_history=[], top_k=3, prefetch=False)
        self.assertEqual(results[0][0], "d1")

    def test_search_suppresses_stale_context_at_pivot(self):
        corpus = [
            {"note_id": "doc_target", "text": "renal dosing adjustment"},
            {"note_id": "doc_stale", "text": "sepsis"},
            {"note_id": "doc_neutral", "text": "discharge planning"},
        ]
        r = make_retriever(corpus, prefetch_weight=0.0, semantic_weight=0.0)

        # --- Steady trajectory: gate closed, history terms retained. ---
        r.reset_session()
        r._encode_query = lambda q: np.array([0.5, 0.0, 0.0])
        r.session_latents = [np.array([0.0, 0.0, 0.0]), np.array([0.25, 0.0, 0.0])]
        steady = r.search("renal", session_history=["sepsis"], top_k=2, prefetch=False)
        self.assertLess(r.last_gate, 0.75)  # gate closed
        steady_ids = [nid for nid, _ in steady]
        self.assertIn("doc_stale", steady_ids)   # context included

        # --- Abrupt pivot: gate open, stale term excluded from lexical query. ---
        r.reset_session()
        r._encode_query = lambda q: np.array([4.0, 0.0, 0.0])
        r.session_latents = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])]
        pivot = r.search("renal", session_history=["sepsis"], top_k=2, prefetch=False)
        self.assertGreater(r.last_gate, 0.75)   # gate open
        pivot_ids = [nid for nid, _ in pivot]
        self.assertNotIn("doc_stale", pivot_ids)  # stale context suppressed
        self.assertIn("doc_target", pivot_ids)

    def test_search_keeps_relevant_match_with_noisy_prefetch(self):
        corpus = [
            {"note_id": "doc_pneumonia", "text": "pneumonia treatment guidelines for clinicians"},
            {"note_id": "doc_surgery", "text": "surgical consultation and operative planning"},
            {"note_id": "doc_shock", "text": "shock and vasopressor dosing guidance"},
        ]
        r = make_retriever(corpus, predictor=NoisyPredictor(), prefetch_weight=0.4,
                           semantic_weight=0.0)
        r._encode_query = lambda q: np.array([0.0, 1.0, 0.0])
        results = r.search("pneumonia treatment", session_history=["fever"], top_k=3, prefetch=True)
        self.assertEqual(results[0][0], "doc_pneumonia")

    def test_copy_with_hyperparams_returns_gdt(self):
        corpus = [{"note_id": "d1", "text": "pneumonia treatment guidelines"}]
        r = make_retriever(corpus)
        r2 = r.copy_with_hyperparams(gate_temperature=1.5)
        self.assertIsInstance(r2, GDTRetriever)
        self.assertEqual(r2.gate_temperature, 1.5)

    def test_reset_session_clears_gate_state(self):
        r = make_retriever([{"note_id": "a", "text": "pneumonia"}])
        r.session_latents = [np.array([0.0, 0.0, 0.0]),
                             np.array([0.5, 0.0, 0.0]),
                             np.array([4.0, 0.0, 0.0])]
        r._history = np.array([1.0, 1.0, 1.0])
        r.last_gate = 0.9
        r.gate_triggers = 3
        r.reset_session()
        self.assertEqual(r.session_latents, [])
        self.assertIsNone(r._history)
        self.assertEqual(r.gate_triggers, 0)
        self.assertEqual(r.last_gate, 0.5)


class GDTGeometryRoundTripTests(unittest.TestCase):
    def test_to_spd_recovers_positive_definite_matrix(self):
        vec = np.array([0.5, 0.2, -0.1])
        P = GDTRetriever._to_spd(vec, 2)
        eigvals = np.linalg.eigh(P)[0]
        self.assertTrue(np.all(eigvals > 0))

    def test_distance_is_affine_invariant(self):
        vec_a = np.array([0.5, 0.2, -0.1])
        vec_b = np.array([1.2, -0.3, 0.4])
        Pa = GDTRetriever._to_spd(vec_a, 2)
        Pb = GDTRetriever._to_spd(vec_b, 2)
        d = GSIGate.affine_invariant_distance(Pa, Pb)
        self.assertGreaterEqual(d, 0)
        # Congruence invariance: distance unchanged under V -> A V A^T.
        A = np.array([[2.0, 1.0], [0.5, 3.0]])
        dA = GSIGate.affine_invariant_distance(A @ Pa @ A.T, A @ Pb @ A.T)
        self.assertAlmostEqual(d, dA, places=4)


if __name__ == "__main__":
    unittest.main()
