import numpy as np


class GSIGate:
    """Geodesic-Shift-Interference (GSI) gate for curvature-aware context suppression.

    The gate monitors the sectional curvature of the intent trajectory on the SPD
    manifold. When curvature exceeds a threshold, it discounts stale prior context,
    preventing latent-context pollution after abrupt topic shifts.
    """

    def __init__(self, spd_dim: int, curvature_threshold: float = 0.65,
                 gate_discount: float = 0.05):
        self.spd_dim = spd_dim
        self.curvature_threshold = curvature_threshold
        self.gate_discount = gate_discount

    @staticmethod
    def vec_to_sym_matrix(vec: np.ndarray, spd_dim: int) -> np.ndarray:
        mat = np.zeros((spd_dim, spd_dim))
        idx = 0
        for col in range(spd_dim):
            for row in range(col, spd_dim):
                mat[row, col] = vec[idx]
                if row != col:
                    mat[col, row] = vec[idx]
                idx += 1
        return mat

    @staticmethod
    def symmetric_matrix_exp(sym_mat: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(sym_mat)
        return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T

    @staticmethod
    def affine_invariant_distance(P: np.ndarray, Q: np.ndarray) -> float:
        eigvals, eigvecs = np.linalg.eigh(P)
        P_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
        M = P_inv_sqrt @ Q @ P_inv_sqrt
        M_eigvals, M_eigvecs = np.linalg.eigh(M)
        M_eigvals = np.maximum(M_eigvals, 1e-12)
        log_M = M_eigvecs @ np.diag(np.log(M_eigvals)) @ M_eigvecs.T
        return float(np.linalg.norm(log_M, ord='fro'))

    def sectional_curvature(self, latents: list[np.ndarray]) -> float:
        """Compute the sectional curvature of the intent trajectory.

        Uses the last 4 latent vectors to compute a discrete sectional curvature
        on the SPD manifold. Returns 0.0 if fewer than 4 latents are available.
        """
        if len(latents) < 4:
            return 0.0
        recent = latents[-4:]
        spd_mats = []
        for v in recent:
            sym = self.vec_to_sym_matrix(v, self.spd_dim)
            spd_mats.append(self.symmetric_matrix_exp(sym))
        D = np.zeros((4, 4))
        for i in range(4):
            for j in range(i + 1, 4):
                d = self.affine_invariant_distance(spd_mats[i], spd_mats[j])
                D[i, j] = D[j, i] = d
        a, b, c, dc = D[0, 1], D[1, 2], D[2, 3], D[0, 3]
        e, f = D[0, 2], D[1, 3]
        a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, dc ** 2
        e2, f2 = e ** 2, f ** 2
        denom = a2 * b2 + b2 * c2 + c2 * d2 + d2 * a2
        if denom < 1e-15:
            return 0.0
        return float(-3.0 * (e2 + f2 - a2 - b2 - c2 - d2) / denom)

    def apply(self, latents: list[np.ndarray],
              weights: list[float]) -> tuple[list[float], bool]:
        """Apply the GSI gate, returning (updated_weights, gate_triggered).

        If the sectional curvature exceeds the threshold, prior context weights
        are decayed by the gate discount factor.
        """
        kappa = self.sectional_curvature(latents)
        if kappa > self.curvature_threshold and len(weights) > 1:
            n = len(weights)
            updated = list(weights)
            for i in range(n - 1):
                updated[i] *= self.gate_discount
            return updated, True
        return list(weights), False
