#!/usr/bin/env python3
"""
src/models/spd_encoder.py

Trainable neural encoder that maps high-dimensional sparse TF-IDF vectors to
Symmetric Positive Definite (SPD) matrices, using the log-Euclidean metric.

Architecture:
  TF-IDF vector
      |
      v
  MLP encoder
      |
      v
  Cholesky parameter vector -> lower-triangular L with positive diagonal
      |
      v
  SPD matrix P = L L^T + eps * I
      |
      v
  Matrix logarithm log(P) (via eigen-decomposition)
      |
      v
  Upper-triangular vectorization  ->  latent vector for retrieval / JEPA

The encoder is trained as an autoencoder: reconstruct the input TF-IDF vector
from the log-SPD latent code. Once trained, the log-SPD vectors replace the
previous TruncatedSVD Euclidean projection in the CMA retriever.
"""

from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def pick_device(preferred: Union[str, torch.device] = "auto") -> torch.device:
    """Select a device whose ops are actually implemented on this platform.

    ``torch.linalg.eigh`` (used for the log-Euclidean SPD mapping) is not
    implemented on Apple's MPS backend, so we probe it and fall back to CPU
    rather than crashing at fit time.
    """
    if str(preferred) == "auto":
        candidates = ["mps", "cuda", "cpu"]
    else:
        candidates = [str(preferred)]
    for name in candidates:
        dev = torch.device(name)
        if name == "cpu":
            return dev
        try:
            if not getattr(torch.backends, name).is_available():
                continue
            probe = torch.eye(3, device=dev)
            torch.linalg.eigh(probe)
            return dev
        except (NotImplementedError, RuntimeError, AttributeError):
            continue
    return torch.device("cpu")


class SPDEncoder(nn.Module):
    """Neural TF-IDF -> SPD(log-Euclidean) encoder/decoder."""

    def __init__(self,
                 input_dim: int = 4000,
                 hidden_dim: int = 512,
                 spd_dim: int = 16,
                 dropout: float = 0.1,
                 device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.spd_dim = spd_dim
        self.n_latent = spd_dim * (spd_dim + 1) // 2
        self.device = torch.device(device)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_latent),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.to(self.device)

    # -----------------------------------------------------------------------
    # SPD construction
    # -----------------------------------------------------------------------
    def _params_to_cholesky(self, params: torch.Tensor) -> torch.Tensor:
        """Map the encoder output to a lower-triangular Cholesky factor L."""
        batch = params.shape[0]
        L = torch.zeros(batch, self.spd_dim, self.spd_dim,
                        device=params.device, dtype=params.dtype)
        idx = 0
        for col in range(self.spd_dim):
            for row in range(col, self.spd_dim):
                if row == col:
                    # Strictly positive diagonal via softplus + floor.
                    L[:, row, col] = torch.nn.functional.softplus(params[:, idx]) + 1e-3
                else:
                    L[:, row, col] = params[:, idx]
                idx += 1
        return L

    def _upper_triangular_indices(self) -> torch.Tensor:
        return torch.triu_indices(self.spd_dim, self.spd_dim, device=self.device)

    def encode_to_spd(self, x: torch.Tensor) -> torch.Tensor:
        """Return SPD matrices of shape (batch, spd_dim, spd_dim)."""
        if sp.issparse(x):
            x = torch.tensor(x.toarray(), dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(x):
            x = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        params = self.encoder(x)
        L = self._params_to_cholesky(params)
        P = L @ L.transpose(-2, -1)
        I = torch.eye(self.spd_dim, device=P.device, dtype=P.dtype).unsqueeze(0)
        return P + 1e-4 * I

    def _matrix_log(self, P: torch.Tensor) -> torch.Tensor:
        """Log-Euclidean mapping: P -> log(P). P must be symmetric positive definite."""
        # Symmetric eigen-decomposition is more stable than general svd for SPD.
        eigvals, eigvecs = torch.linalg.eigh(P)
        eigvals = eigvals.clamp(min=1e-6)
        log_eigvals = torch.log(eigvals)
        log_P = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)
        return log_P

    def _l2_normalize_rows(self, x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        return x / (torch.norm(x, dim=1, keepdim=True) + eps)

    def _encode_to_log_vec_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient-enabled version used during training. Output is L2-normalized."""
        P = self.encode_to_spd(x)
        log_P = self._matrix_log(P)
        idx = self._upper_triangular_indices()
        return self._l2_normalize_rows(log_P[:, idx[0], idx[1]])

    def encode_to_log_vec(self, x: Union[np.ndarray, sp.spmatrix, torch.Tensor]) -> torch.Tensor:
        """Return L2-normalized log-Euclidean upper-triangular vectors.

        Shape: (batch, n_latent). Gradients are disabled.
        """
        self.eval()
        if sp.issparse(x):
            x = torch.tensor(x.toarray(), dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(x):
            x = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self._encode_to_log_vec_grad(x)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def fit_retrieval(self,
                      queries_tfidf: Union[np.ndarray, sp.spmatrix],
                      positives_tfidf: Union[np.ndarray, sp.spmatrix],
                      corpus_tfidf: Union[np.ndarray, sp.spmatrix],
                      epochs: int = 30,
                      n_negatives: int = 5,
                      batch_size: int = 64,
                      lr: float = 1e-4,
                      seed: int = 42) -> "SPDEncoder":
        """Fine-tune the encoder with query-target InfoNCE contrastive loss.

        Positive pairs come from (query, target_note) pairs in the training
        vignettes; negatives are sampled from the full corpus.
        """
        torch.manual_seed(seed)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n = queries_tfidf.shape[0]
        n_corpus = corpus_tfidf.shape[0]

        print(f"  Fine-tuning SPD encoder on {n} query-target pairs "
              f"({n_negatives} negatives each)...")

        def to_tensor(mat):
            if sp.issparse(mat):
                return torch.tensor(mat.toarray(), dtype=torch.float32, device=self.device)
            return torch.tensor(np.asarray(mat, dtype=float), dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = perm[start:start + batch_size]
                q = to_tensor(queries_tfidf[batch_idx])
                pos = to_tensor(positives_tfidf[batch_idx])

                # Random negatives from the corpus.
                neg_idx = np.random.choice(n_corpus, size=len(batch_idx) * n_negatives)
                neg_mat = corpus_tfidf[neg_idx]
                neg = to_tensor(neg_mat).reshape(len(batch_idx), n_negatives, -1)

                q_log = self._encode_to_log_vec_grad(q)
                pos_log = self._encode_to_log_vec_grad(pos)
                neg_log = self._encode_to_log_vec_grad(neg.reshape(-1, neg.shape[-1]))
                neg_log = neg_log.reshape(len(batch_idx), n_negatives, -1)

                sim_pos = (q_log * pos_log).sum(dim=1, keepdim=True)  # (B, 1)
                sim_neg = (q_log.unsqueeze(1) * neg_log).sum(dim=2)    # (B, K)

                logits = torch.cat([sim_pos, sim_neg], dim=1)  # (B, 1+K)
                labels = torch.zeros(len(batch_idx), dtype=torch.long, device=self.device)
                loss = nn.functional.cross_entropy(logits / 0.1, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                div = max(n_batches, 1)
                
                # Calculate avg similarity gap between positive pairs and negative pairs
                with torch.no_grad():
                    avg_pos_sim = sim_pos.mean().item()
                    avg_neg_sim = sim_neg.mean().item()
                    sim_gap = avg_pos_sim - avg_neg_sim

                print(f"    SPD retrieval epoch {epoch:3d}/{epochs} | loss={epoch_loss/div:.4f} | "
                      f"sim_pos={avg_pos_sim:.3f} sim_neg={avg_neg_sim:.3f} (gap={sim_gap:+.3f})")

        self.eval()
        return self

    def fit(self,
            tfidf_matrix: Union[np.ndarray, sp.spmatrix],
            epochs: int = 50,
            batch_size: int = 256,
            lr: float = 1e-3,
            similarity_weight: float = 1.0,
            seed: int = 42) -> "SPDEncoder":
        """Train the encoder to reconstruct TF-IDF and preserve pairwise similarity.

        The similarity term ensures the log-SPD latent codes retain the TF-IDF
        geometry needed for retrieval, preventing collapse to a single point.
        """
        torch.manual_seed(seed)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n = tfidf_matrix.shape[0]

        # Small dense matrices (256 x 4000) run faster single-threaded; thread
        # oversubscription across the (CPU) eigen-decomposition path can stall
        # large fits on many-core machines.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        print(f"  Training SPD encoder (input_dim={self.input_dim}, "
              f"spd_dim={self.spd_dim}, latent={self.n_latent}) on {n} documents...")

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_sim = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = perm[start:start + batch_size]
                xb = tfidf_matrix[batch_idx]
                if sp.issparse(xb):
                    xb = torch.tensor(xb.toarray(), dtype=torch.float32, device=self.device)
                else:
                    xb = torch.tensor(np.asarray(xb, dtype=float), dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                log_vec = self._encode_to_log_vec_grad(xb)
                x_recon = self.decoder(log_vec)
                recon_loss = nn.functional.mse_loss(x_recon, xb)

                # Preserve pairwise TF-IDF cosine similarity in log-SPD space.
                target_sim = self._pairwise_cosine(xb)
                pred_sim = self._pairwise_cosine(log_vec)
                sim_loss = nn.functional.mse_loss(pred_sim, target_sim)

                loss = recon_loss + similarity_weight * sim_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_sim += sim_loss.item()
                n_batches += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                div = max(n_batches, 1)
                # Check for representation collapse (variance across batch dimensions)
                with torch.no_grad():
                    latent_var = log_vec.var(dim=0).mean().item()
                    
                print(f"    SPD epoch {epoch:3d}/{epochs} | total={epoch_loss/div:.4f} "
                      f"recon={epoch_recon/div:.4f} sim={epoch_sim/div:.4f} | "
                      f"latent_var={latent_var:.5f}")

        self.eval()
        return self

    @staticmethod
    def _pairwise_cosine(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Return pairwise cosine similarity matrix, scaled to [-1, 1]."""
        xn = x / (torch.norm(x, dim=1, keepdim=True) + eps)
        return xn @ xn.T

    def forward(self, x: torch.Tensor):
        """Autoencoder forward: returns (reconstruction, log-spd latent)."""
        log_vec = self.encode_to_log_vec(x)
        x_recon = self.decoder(log_vec)
        return x_recon, log_vec
