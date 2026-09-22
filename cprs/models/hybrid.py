"""
Density-weighted hybrid recommender: collaborative filtering + a cold ingredient.

Day 3 of the final build plan. Blends the CF score (strong for users with rich
history) with a non-CF "cold" ingredient (content or popularity — the fallback
that carries new/sparse users, where CF has little signal). The blend weight is
a smooth function of each user's interaction density, so the model degrades
gracefully toward the cold ingredient as history shrinks — this is the cold-start
mechanism, expressed in one continuous knob rather than a hard mode switch.

    alpha(u) = n_train(u) / (n_train(u) + k0)          # -> 1 for heavy users
    hybrid   = alpha * norm(cf) + (1 - alpha) * norm(cold)

Below `coldstart_thresh` interactions we clamp alpha = 0 (pure cold ingredient),
modelling a genuinely new user for whom CF factors are unreliable.

Scores from CF (dot products, unbounded) and the cold ingredient ([0,1]-ish)
live on different scales, so each is per-user min-max normalized before blending.
"""
from __future__ import annotations

import numpy as np


def normalize_rows(M: np.ndarray) -> np.ndarray:
    """Per-row min-max normalization to [0, 1] (constant rows -> 0)."""
    mn = M.min(axis=1, keepdims=True)
    mx = M.max(axis=1, keepdims=True)
    return (M - mn) / (mx - mn + 1e-9)


class HybridModel:
    """Density-weighted blend of CF with a content/popularity cold ingredient."""

    def __init__(self, k0: float = 10.0, coldstart_thresh: int = 0, cold: str = "content"):
        assert cold in ("content", "popularity")
        self.k0 = k0
        self.coldstart_thresh = coldstart_thresh
        self.cold = cold

    def alpha(self, density: np.ndarray) -> np.ndarray:
        """CF weight per user from interaction density."""
        a = density / (density + self.k0)
        if self.coldstart_thresh > 0:
            a = np.where(density < self.coldstart_thresh, 0.0, a)
        return a.astype(np.float32)

    def score_matrix(
        self,
        cf: np.ndarray,
        content: np.ndarray,
        popularity: np.ndarray,
        density: np.ndarray,
    ) -> np.ndarray:
        """Return the (users × problems) hybrid score matrix."""
        cf_n = normalize_rows(cf)
        if self.cold == "content":
            cold_n = normalize_rows(content)
        else:  # popularity: same vector for every user
            cold_n = normalize_rows(popularity[None, :])  # (1 × P), broadcasts in blend
        a = self.alpha(density)[:, None]
        return (a * cf_n + (1.0 - a) * cold_n).astype(np.float32)
