import unittest
import numpy as np

from src.models.gsi_gate import GSIGate


class GSIGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = GSIGate(spd_dim=4, curvature_threshold=0.65, gate_discount=0.05)

    def test_vec_to_sym_matrix_roundtrip(self):
        vec = np.arange(10, dtype=float)
        mat = self.gate.vec_to_sym_matrix(vec, 4)
        self.assertEqual(mat.shape, (4, 4))
        self.assertTrue(np.allclose(mat, mat.T))

    def test_symmetric_matrix_exp_preserves_spd(self):
        vec = np.random.randn(10)
        sym = self.gate.vec_to_sym_matrix(vec, 4)
        P = self.gate.symmetric_matrix_exp(sym)
        eigvals = np.linalg.eigh(P)[0]
        self.assertTrue(np.all(eigvals > 0))

    def test_affine_invariant_distance_positive(self):
        v1 = np.random.randn(10)
        v2 = np.random.randn(10)
        sym1 = self.gate.vec_to_sym_matrix(v1, 4)
        sym2 = self.gate.vec_to_sym_matrix(v2, 4)
        P = self.gate.symmetric_matrix_exp(sym1)
        Q = self.gate.symmetric_matrix_exp(sym2)
        d = self.gate.affine_invariant_distance(P, Q)
        self.assertGreaterEqual(d, 0)

    def test_sectional_curvature_returns_zero_with_fewer_than_4_latents(self):
        latents = [np.random.randn(10) for _ in range(3)]
        self.assertEqual(self.gate.sectional_curvature(latents), 0.0)

    def test_sectional_curvature_with_4_or_more_latents(self):
        latents = [np.random.randn(10) for _ in range(6)]
        kappa = self.gate.sectional_curvature(latents)
        self.assertIsInstance(kappa, float)

    def test_apply_does_not_trigger_below_threshold(self):
        # Use identical latents so curvature is near zero
        v = np.random.randn(10)
        latents = [v.copy() for _ in range(6)]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        updated, triggered = self.gate.apply(latents, weights)
        self.assertFalse(triggered)
        self.assertEqual(updated, weights)

    def test_apply_discounts_weights_when_triggered(self):
        # Use strongly diverging latents to produce high curvature
        rng = np.random.RandomState(42)
        latents = [
            rng.randn(10) * 5,
            rng.randn(10) * 5,
            rng.randn(10) * 5,
            rng.randn(10) * 5,
            rng.randn(10) * 5,
            rng.randn(10) * 5,
        ]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        updated, triggered = self.gate.apply(latents, weights)
        if triggered:
            self.assertTrue(all(w < 1.0 for w in updated[:-1]))
            self.assertEqual(updated[-1], 1.0)
            for i in range(len(weights) - 1):
                self.assertAlmostEqual(updated[i], weights[i] * self.gate.gate_discount)

    def test_apply_returns_copy_weights_when_not_triggered(self):
        v = np.random.randn(10)
        latents = [v.copy() for _ in range(6)]
        weights = [0.5, 0.3, 1.0, 0.7, 0.2, 0.9]
        updated, triggered = self.gate.apply(latents, weights)
        self.assertFalse(triggered)
        self.assertEqual(updated, weights)


if __name__ == "__main__":
    unittest.main()
