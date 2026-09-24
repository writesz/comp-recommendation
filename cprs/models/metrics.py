"""Top-N ranking metrics for offline recommender evaluation.

Extracted from ``scripts/evaluate.py`` so the metric definitions have a single home
and can be tested directly against hand-worked examples (``tests/test_metrics.py``).

All metrics follow the definitions in the report's Evaluation Methodology section. For a
user with held-out relevant set R and a top-K list L (best first):

    HitRate@K   = 1 if L n R is non-empty else 0
    Precision@K = |L n R| / K
    Recall@K    = |L n R| / |R|
    MRR@K       = 1 / (rank of first hit), 0 if no hit inside K
    nDCG@K      = DCG@K / IDCG@K, binary relevance,
                  IDCG@K = sum_{i=1..min(|R|,K)} 1/log2(i+1)

Note that Recall@K is computed against the full held-out set, so it is bounded above by
K/|R| when a user has more held-out items than K -- this is the standard convention and
is why Recall@10 cannot reach 1.0 for a user with 20 held-out solves.
"""
from __future__ import annotations

import numpy as np

METRICS = ("hit", "precision", "recall", "mrr", "ndcg")


def idcg_table(k_list):
    """Cumulative ideal-DCG lookup: ``table[k][j]`` is IDCG@k for j+1 relevant items."""
    return {k: np.cumsum(1.0 / np.log2(np.arange(2, k + 2))) for k in k_list}


def top_k_indices(scores: np.ndarray, kmax: int) -> np.ndarray:
    """Indices of the ``kmax`` highest scores, best first.

    Uses argpartition for O(n) selection followed by an O(kmax log kmax) sort of the
    selected slice, rather than sorting all n candidates.
    """
    if kmax >= scores.size:
        return np.argsort(-scores)
    top = np.argpartition(-scores, kmax)[:kmax]
    return top[np.argsort(-scores[top])]


def rank_metrics(scores, seen, relevant, k_list, idcg=None):
    """Compute all ranking metrics for one user.

    Args:
        scores:   score vector over every candidate item.
        seen:     item indices to mask out (the user's training solves); ranking a user's
                  known items would inflate every metric, so they are removed first.
        relevant: held-out item indices that count as hits.
        k_list:   cut-offs to report at.
        idcg:     optional precomputed :func:`idcg_table` for ``k_list``.

    Returns:
        dict mapping ``"<metric>@<k>"`` to a float.
    """
    k_list = list(k_list)
    kmax = max(k_list)
    idcg = idcg or idcg_table(k_list)

    relevant = np.asarray(relevant, dtype=np.int64)
    if relevant.size == 0:
        raise ValueError("a user with no held-out items cannot be scored")

    scores = np.asarray(scores, dtype=float).copy()
    if seen is not None and len(seen) > 0:
        scores[np.asarray(seen, dtype=np.int64)] = -np.inf

    top = top_k_indices(scores, kmax)
    rel_set = set(relevant.tolist())
    hit_flags = np.array([1.0 if p in rel_set else 0.0 for p in top])
    first_hit = int(np.argmax(hit_flags)) if hit_flags.any() else -1

    out = {}
    for k in k_list:
        hk = hit_flags[:k]
        n_hit = hk.sum()
        out[f"hit@{k}"] = 1.0 if n_hit > 0 else 0.0
        out[f"precision@{k}"] = n_hit / k
        out[f"recall@{k}"] = n_hit / relevant.size
        out[f"mrr@{k}"] = 1.0 / (first_hit + 1) if 0 <= first_hit < k else 0.0
        dcg = (hk / np.log2(np.arange(2, k + 2))).sum()
        out[f"ndcg@{k}"] = dcg / idcg[k][min(relevant.size, k) - 1]
    return out
