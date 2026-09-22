"""
Collaborative-filtering recommender via implicit-feedback matrix factorization.

Day 2 of the final build plan. Learns latent user/item factors from the
user×problem "solved" matrix (Alternating Least Squares, Hu-Koren-Volinsky),
so recommendations come from "users with similar solve patterns solved these
next" rather than problem content. This is the second half of the hybrid.

Falls back to TruncatedSVD if the `implicit` library is unavailable, so the
pipeline never hard-depends on a native build.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from loguru import logger

try:
    from implicit.als import AlternatingLeastSquares
    _HAVE_IMPLICIT = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_IMPLICIT = False


class CollaborativeModel:
    """Matrix-factorization CF over an implicit-feedback (solved=1) matrix."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 20,
        alpha: float = 15.0,
        random_state: int = 42,
        backend: str = "auto",
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.backend = backend if backend != "auto" else ("als" if _HAVE_IMPLICIT else "svd")
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, train: sp.csr_matrix) -> "CollaborativeModel":
        """Learn user/item factors from the training matrix (users × problems)."""
        if self.backend == "als":
            logger.info(
                f"Training ALS (factors={self.factors}, reg={self.regularization}, "
                f"iters={self.iterations}, alpha={self.alpha})"
            )
            model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=self.random_state,
            )
            # implicit treats matrix values as confidence; scale binary solves by alpha.
            # Disable BLAS threadpool per implicit's recommendation (avoids contention).
            try:
                from threadpoolctl import threadpool_limits
                with threadpool_limits(1, "blas"):
                    model.fit((train * self.alpha).astype(np.float32), show_progress=False)
            except Exception:
                model.fit((train * self.alpha).astype(np.float32), show_progress=False)
            self.user_factors = np.asarray(model.user_factors)
            self.item_factors = np.asarray(model.item_factors)
        else:
            from sklearn.decomposition import TruncatedSVD

            logger.info(f"Training TruncatedSVD fallback (factors={self.factors})")
            svd = TruncatedSVD(n_components=self.factors, random_state=self.random_state)
            self.user_factors = svd.fit_transform(train).astype(np.float32)
            self.item_factors = svd.components_.T.astype(np.float32)
        logger.success(
            f"CF trained: user_factors {self.user_factors.shape}, "
            f"item_factors {self.item_factors.shape}"
        )
        return self

    def score_all(self, user_rows: np.ndarray) -> np.ndarray:
        """Return a dense (len(user_rows) × n_problems) score matrix."""
        assert self.user_factors is not None, "call fit() first"
        return self.user_factors[user_rows] @ self.item_factors.T
