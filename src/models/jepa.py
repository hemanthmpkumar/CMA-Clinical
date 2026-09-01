#!/usr/bin/env python3
"""
src/models/jepa.py

PyTorch / geoopt JEPA-style latent predictor for the CMA retriever.

The predictor is trained globally across historical sessions rather than fitted
on-the-fly for a single short session. It uses:

  - A 2-layer MLP whose weight matrices live on the Stiefel manifold
    (orthonormal columns), keeping the latent mapping on a well-conditioned
    non-Euclidean constraint set.
  - A slow moving-average target network, momentum-updated after each
    optimization step (EMA decay = 0.996).
  - Stop-gradient on the target branch.
  - `geoopt.optim.RiemannianAdam` to project Stiefel-parameter gradients back
    onto the tangent space and retract the weights onto the manifold.

This replaces the earlier raw-NumPy SGD implementation, which performed
unconstrained Euclidean updates and could pull the weights off the intended
non-linear parameter space.
"""

from typing import Optional, Union

import geoopt
import math
import numpy as np
import torch
import torch.nn as nn


def log_euclidean_weights(spd_dim: int, device: torch.device) -> torch.Tensor:
    """Weight vector for the Log-Euclidean distance squared.

    For two SPD matrices P, Q with log-Euclidean vectorisations v_P, v_Q
    (upper-triangular entries of log(P), log(Q)), the squared Log-Euclidean
    distance is:

      d²(P, Q) = ||log(P) - log(Q)||²_F
               = Σ_i (Δdiag_i)² + 2 · Σ_{i<j} (Δoff_ij)²

    Returns a weight vector of length spd_dim*(spd_dim+1)//2 where diagonal
    entries get weight 1 and off-diagonal entries get weight 2.
    """
    w = []
    for col in range(spd_dim):
        for row in range(col, spd_dim):
            w.append(1.0 if row == col else 2.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


class StiefelLinear(nn.Module):
    """Linear layer whose weight matrix is constrained to the Stiefel manifold.

    The Stiefel manifold St(n, k) is the set of n x k matrices with orthonormal
    columns. Riemannian optimization keeps the transformation isometric and
    avoids the parameter drift that occurs with unconstrained SGD.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Stiefel requires n >= k. We store the larger dimension as rows.
        n = max(in_features, out_features)
        k = min(in_features, out_features)
        W = torch.empty(n, k)
        nn.init.orthogonal_(W)  # produces a matrix on the Stiefel manifold
        self.W = geoopt.ManifoldParameter(W, manifold=geoopt.Stiefel())
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.transposed = in_features < out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transposed:
            # W has shape (out, in); x @ W^T -> (batch, out)
            return x @ self.W.transpose(-2, -1) + self.bias
        else:
            # W has shape (in, out); x @ W -> (batch, out)
            return x @ self.W + self.bias


class _LatentMLP(nn.Module):
    """2-layer Stiefel MLP for latent prediction."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            StiefelLinear(latent_dim, hidden_dim),
            nn.ReLU(),
            StiefelLinear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPAPredictor:
    """JEPA latent dynamics predictor implemented with PyTorch + geoopt."""

    def __init__(self,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 spd_dim: Optional[int] = None,
                 lr: float = 1e-3,
                 ema_decay: float = 0.996,
                 seed: int = 42,
                 device: Union[str, torch.device] = "cpu"):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        if spd_dim is None:
            spd_dim = int((math.isqrt(8 * latent_dim + 1) - 1) // 2)
        self.spd_dim = spd_dim

        torch.manual_seed(seed)
        torch.set_num_threads(10)
        if str(device) == "auto":
            from .spd_encoder import pick_device
            self.device = pick_device("auto")
        else:
            self.device = torch.device(device)

        self.log_euclidean_w = log_euclidean_weights(spd_dim, self.device)

        self.online = _LatentMLP(latent_dim, hidden_dim).to(self.device)
        self.target = _LatentMLP(latent_dim, hidden_dim).to(self.device)
        self._sync_target()

        self.optimizer = geoopt.optim.RiemannianAdam(self.online.parameters(), lr=lr)

    # -----------------------------------------------------------------------
    # Target network helpers
    # -----------------------------------------------------------------------
    def _sync_target(self):
        """Copy online parameters into target (used for initialization only)."""
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.copy_(p_o.data)

    def _update_target(self):
        """EMA update: target = beta * target + (1 - beta) * online."""
        beta = self.ema_decay
        with torch.no_grad():
            for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
                p_t.data.mul_(beta).add_(p_o.data, alpha=1.0 - beta)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / torch.clamp(norm, min=eps)

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict the next latent direction from the current latent."""
        self.online.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self._l2_normalize(x)
        with torch.no_grad():
            y = self.online(x)
            y = self._l2_normalize(y)
        return y.squeeze(0).cpu().numpy()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def fit(self,
            X: Union[np.ndarray, torch.Tensor],
            Y: Union[np.ndarray, torch.Tensor],
            epochs: int = 120,
            batch_size: int = 64) -> "JEPAPredictor":
        """Train online network to predict target-encoded next latents.

        Args:
            X: (N, latent_dim) current latents.
            Y: (N, latent_dim) next latents.
            epochs: number of passes over the transition data.
            batch_size: SGD mini-batch size.
        """
        X = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32, device=self.device)
        Y = torch.tensor(np.asarray(Y, dtype=float), dtype=torch.float32, device=self.device)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if Y.ndim == 1:
            Y = Y.unsqueeze(0)

        n = X.shape[0]
        if n == 0:
            return self

        # Normalize to unit length so the network predicts direction only.
        X = self._l2_normalize(X)
        Y = self._l2_normalize(Y)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        print(f"Running JEPA on device: {self.device} with {torch.get_num_threads()} CPU threads.")
        self.online.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_align = 0.0
            epoch_pred_var = 0.0
            n_batches = 0
            
            for xb, yb in loader:
                pred = self.online(xb)

                with torch.no_grad():
                    target = self.target(yb)
                    # Metric 1: Alignment between prediction and target
                    align = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
                    # Metric 2: Variance across batch to detect constant-output collapse
                    pred_var = pred.var(dim=0).mean()

                loss = ((pred - target) ** 2 * self.log_euclidean_w).sum(dim=1).mean()

                # VICReg-style variance-preservation term: reward per-dimension
                # std of predictions within the batch. Without it the online
                # network collapses to a near-constant output (pred_var -> 0).
                if pred.shape[0] >= 2:
                    std = pred.std(dim=0) + 1e-4
                    loss = loss - 0.1 * std.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target()

                epoch_loss += loss.item()
                epoch_align += align.item()
                epoch_pred_var += pred_var.item()
                n_batches += 1

            if epoch % 20 == 0 or epoch == epochs - 1:
                div = max(n_batches, 1)
                print(f"  JEPA epoch {epoch:3d}/{epochs} | log_euc={epoch_loss/div:.4f} | "
                      f"align_cos={epoch_align/div:.3f} | pred_var={epoch_pred_var/div:.5f}")

        return self
