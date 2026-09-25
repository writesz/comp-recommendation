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

    # --- Serving unseen users -------------------------------------------
    #
    # ALS is transductive: `user_factors` is a learned parameter table with one
    # row per training user, not a function of a solve history. A person who
    # was not in the cohort therefore has no row, and nothing to look up.
    #
    # The way out is already inside ALS. Its user step is a closed-form ridge
    # solve given fixed item factors, so the same step can be applied to a new
    # user with Y frozen — "fold-in". For the Hu-Koren-Volinsky objective with
    # confidence c_ui = 1 + alpha * r_ui and binary r:
    #
    #     x_u = (Y'C^u Y + lambda*I)^-1 Y'C^u p(u)
    #
    # C^u is |items|x|items|, but C^u - I is zero off the user's solved set S,
    # and p(u) is zero off S too, so this collapses to
    #
    #     x_u = (Y'Y + alpha * Y_S'Y_S + lambda*I)^-1 (1 + alpha) * sum_{i in S} y_i
    #
    # Y'Y is shared by every user and precomputed once. What remains per user
    # is |S|*f^2 to accumulate and one f x f solve — sub-millisecond at f=64,
    # so this runs inside a web request.
    #
    # The assumption worth stating: fold-in takes Y as a good basis, i.e. the
    # new user's behaviour is expressible in the latent space learned from the
    # cohort. It degrades for users unlike the cohort, and says nothing at all
    # about items absent from Y.

    def _gram(self) -> np.ndarray:
        """Y'Y, cached — shared across every fold-in."""
        if getattr(self, "_gram_cache", None) is None:
            self._gram_cache = self.item_factors.T @ self.item_factors
        return self._gram_cache

    def fold_in(self, solved_cols) -> np.ndarray:
        """
        Project an unseen user onto the fixed item factors.

        Args:
            solved_cols: column indices of the problems this user has solved.

        Returns:
            A latent factor vector of length ``factors``. All-zero when the
            user has no solves inside the model's item space, which scores
            every item equally and correctly signals "no CF opinion".
        """
        assert self.item_factors is not None, "load() or fit() first"
        cols = np.unique(np.asarray(solved_cols, dtype=np.int64))
        f = self.item_factors.shape[1]
        if cols.size == 0:
            return np.zeros(f, dtype=np.float32)

        Y_S = self.item_factors[cols]                      # |S| x f

        if self.backend == "svd":
            # TruncatedSVD factors are an orthogonal basis rather than an ALS
            # solution, so the matching projection is a plain dot product.
            return Y_S.sum(axis=0).astype(np.float32)

        A = self._gram() + self.alpha * (Y_S.T @ Y_S)
        A[np.diag_indices_from(A)] += self.regularization
        b = (1.0 + self.alpha) * Y_S.sum(axis=0)
        return np.linalg.solve(A, b).astype(np.float32)

    def score_folded(self, solved_cols) -> np.ndarray:
        """Score every item for an unseen user, via fold-in."""
        return self.fold_in(solved_cols) @ self.item_factors.T

    # --- Persistence ------------------------------------------------------

    def save(self, path, problem_ids: list | None = None) -> None:
        """Persist item factors and hyperparameters for serving.

        User factors are deliberately not saved: serving folds users in on
        demand, and the cohort's own rows are of no use to the application.
        ``problem_ids`` travels with the factors so a served request can map a
        user's solved problems onto columns without a second artefact that
        could drift out of step with the model.
        """
        assert self.item_factors is not None, "call fit() first"
        payload = dict(
            item_factors=self.item_factors,
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            backend=self.backend,
        )
        if problem_ids is not None:
            assert len(problem_ids) == self.item_factors.shape[0], (
                f"problem_ids ({len(problem_ids)}) must match item factor rows "
                f"({self.item_factors.shape[0]})"
            )
            payload["problem_ids"] = np.array(problem_ids, dtype=object)
        np.savez_compressed(str(path), **payload)
        logger.success(f"CF model saved to {path} (items {self.item_factors.shape})")

    @classmethod
    def load(cls, path) -> "CollaborativeModel":
        """Load a model saved by :meth:`save`, ready to fold users in."""
        data = np.load(str(path), allow_pickle=True)
        model = cls(
            factors=int(data["factors"]),
            regularization=float(data["regularization"]),
            alpha=float(data["alpha"]),
            backend=str(data["backend"]),
        )
        model.item_factors = data["item_factors"]
        model.problem_ids = (
            list(data["problem_ids"]) if "problem_ids" in data.files else None
        )
        model.column_of = (
            {pid: i for i, pid in enumerate(model.problem_ids)}
            if model.problem_ids else {}
        )
        logger.info(f"CF model loaded from {path} (items {model.item_factors.shape})")
        return model
