"""
Online cross-platform skill estimator.

Realizes the "unify on difficulty, assume solvability, correct live" idea. A
user's skill is tracked on the unified [0,1] difficulty scale (validated to be
cross-platform-consistent at r=0.77) by a stochastic-
approximation quantile tracker: each solved problem nudges the estimate toward
the q-th percentile of the difficulties the user solves. This is:

  - Online: O(1) per solve, so it corrects live as new outcomes arrive.
  - Convergent on solve-only data: the update self-limits (unlike a naive
    wins-only Elo that would drift upward without bound).
  - Cross-platform: difficulties from any platform update the same estimate, so
    AtCoder history can seed a Codeforces skill prior for a cold user.

Update rule (Robbins-Monro quantile estimation for percentile q):
    theta <- theta + lr * (q - 1[b < theta])
so an above-estimate solve raises theta by lr*q and a below-estimate solve
lowers it by lr*(1-q); the fixed point is the user's q-th percentile difficulty.
"""
from __future__ import annotations

import numpy as np


class OnlineSkillEstimator:
    """Live skill estimate on the unified difficulty scale via online quantile tracking."""

    def __init__(self, q: float = 0.75, lr: float = 0.02, init: float = 0.3):
        self.q = q
        self.lr = lr
        self.theta = float(init)
        self.n = 0

    def update(self, difficulty: float) -> float:
        """Incorporate one solved problem's normalized difficulty; return new estimate."""
        if difficulty is None or np.isnan(difficulty):
            return self.theta
        self.theta += self.lr * (self.q - (1.0 if difficulty < self.theta else 0.0))
        self.theta = float(min(1.0, max(0.0, self.theta)))
        self.n += 1
        return self.theta

    def seed(self, difficulties: list[float]) -> "OnlineSkillEstimator":
        """Batch-initialize from a history (e.g. another platform), then stay online."""
        vals = [d for d in difficulties if d is not None and not np.isnan(d)]
        if vals:
            self.theta = float(np.percentile(vals, self.q * 100))
            self.n = len(vals)
        return self

    def predict_solvable(self, difficulty: float, scale: float = 8.0) -> float:
        """P(user solves a problem of this difficulty) under a logistic skill model."""
        return float(1.0 / (1.0 + np.exp(-scale * (self.theta - difficulty))))
