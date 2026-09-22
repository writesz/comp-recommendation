"""
Offline content-based scorer aligned to the interaction-matrix problem space.

Day 2 of the final build plan. This mirrors the deployed content engine's
scoring formula (topic-gap + difficulty-fit + popularity) but is deterministic
and vectorized over the matrix's fixed problem ordering, so it can serve both as
a fair content-only *baseline* and as the content ingredient of the Day-3 hybrid.

Formula per (user u, problem p):
    score = 0.4 * topic_gap + 0.4 * difficulty_fit + 0.1 * popularity
  - topic_gap:     mean over p's tags of the user's per-tag weakness weight
                   1/(1+solves_in_tag)  (unseen tags weigh most)
  - difficulty_fit gaussian around the user's level (75th pct of solved
                   difficulty, nudged +stretch); 0 if difficulty unknown
  - popularity:    log-scaled solve count, shared across users
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from loguru import logger

DATASET = Path(__file__).resolve().parent.parent / "data" / "cprs_unified_tagged.json"


class ContentScorer:
    """Deterministic content scorer over a fixed problem ordering."""

    def __init__(self, problem_ids: list[str], stretch: float = 0.1, sigma: float = 0.15):
        self.problem_ids = problem_ids
        self.stretch = stretch
        self.sigma = sigma
        self.n_problems = len(problem_ids)

        with open(DATASET) as f:
            by_id = {p["cprs_id"]: p for p in json.load(f)}

        # Aligned per-problem feature arrays.
        self.difficulty = np.full(self.n_problems, np.nan, dtype=np.float32)
        popularity = np.zeros(self.n_problems, dtype=np.float32)
        tag_vocab: dict[str, int] = {}
        rows, cols = [], []
        for j, pid in enumerate(problem_ids):
            p = by_id.get(pid, {})
            d = p.get("difficulty_normalized")
            if d is not None:
                self.difficulty[j] = float(d)
            popularity[j] = min(1.0, np.log1p(p.get("solve_count") or 0) / 12.0)
            for t in p.get("tags_unified", []) or []:
                k = tag_vocab.setdefault(t, len(tag_vocab))
                rows.append(j)
                cols.append(k)

        self.popularity = popularity
        n_tags = max(1, len(tag_vocab))
        self.tag_matrix = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.n_problems, n_tags),
        )
        self.tags_per_problem = np.asarray(self.tag_matrix.sum(axis=1)).ravel()
        self.tags_per_problem[self.tags_per_problem == 0] = 1.0  # avoid /0
        self._diff_filled = np.nan_to_num(self.difficulty, nan=-10.0)  # -> diff_fit ~0
        logger.info(
            f"ContentScorer: {self.n_problems} problems, {n_tags} tags, "
            f"{np.isnan(self.difficulty).sum()} without difficulty"
        )

    def score_all(self, train: sp.csr_matrix, user_rows: np.ndarray) -> np.ndarray:
        """Return a dense (len(user_rows) × n_problems) content score matrix."""
        sub = train[user_rows]

        # Topic gap: per-user per-tag solve counts -> weakness weights -> per-problem mean.
        tag_counts = (sub @ self.tag_matrix).toarray()            # (U × n_tags)
        weakness = 1.0 / (1.0 + tag_counts)                       # unseen tags -> 1.0
        topic_gap = (weakness @ self.tag_matrix.T) / self.tags_per_problem  # (U × P)

        # Difficulty fit: gaussian around each user's level.
        centers = self._user_difficulty_levels(sub)              # (U,)
        z = (self._diff_filled[None, :] - centers[:, None]) / self.sigma
        diff_fit = np.exp(-0.5 * z * z).astype(np.float32)

        return (0.4 * topic_gap + 0.4 * diff_fit + 0.1 * self.popularity[None, :]).astype(np.float32)

    def score_all_match(self, train: sp.csr_matrix, user_rows: np.ndarray,
                        stretch: float = 0.05) -> np.ndarray:
        """
        Difficulty-match + *familiar*-category scorer (the "assume they'd solve it"
        bet): recommend problems at the user's level in categories they already
        practise — the opposite of the pedagogical content scorer, and aligned
        with how people actually pick their next problem.
        """
        sub = train[user_rows]
        # Familiarity: prefer tags the user has solved a lot (not unseen ones).
        tag_counts = (sub @ self.tag_matrix).toarray()                  # (U × n_tags)
        familiar = (tag_counts @ self.tag_matrix.T) / self.tags_per_problem  # (U × P)
        familiar /= (familiar.max(axis=1, keepdims=True) + 1e-9)

        centers = self._user_difficulty_levels(sub, stretch=stretch)
        z = (self._diff_filled[None, :] - centers[:, None]) / self.sigma
        diff_fit = np.exp(-0.5 * z * z).astype(np.float32)

        return (0.5 * diff_fit + 0.5 * familiar + 0.05 * self.popularity[None, :]).astype(np.float32)

    def _user_difficulty_levels(self, sub: sp.csr_matrix, stretch: float | None = None) -> np.ndarray:
        """75th-percentile of each user's solved-problem difficulty, nudged by stretch."""
        centers = np.full(sub.shape[0], 0.3, dtype=np.float32)  # default for no-difficulty users
        indptr, indices = sub.indptr, sub.indices
        for i in range(sub.shape[0]):
            cols = indices[indptr[i]:indptr[i + 1]]
            diffs = self.difficulty[cols]
            diffs = diffs[~np.isnan(diffs)]
            if diffs.size:
                centers[i] = np.percentile(diffs, 75)
        s = self.stretch if stretch is None else stretch
        return np.minimum(centers + s, 1.0)
