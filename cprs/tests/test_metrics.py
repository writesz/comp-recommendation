"""Ranking metrics checked against hand-worked examples.

The metric code is the part of the pipeline where a silent error would be least
visible: a wrong nDCG denominator or an off-by-one in the rank of the first hit would
shift every model by the same factor and so would not disturb the *ordering* that the
report's conclusions rest on. These tests pin the definitions to arithmetic computed by
hand, independently of the implementation.
"""
import math

import numpy as np
import pytest

from models.metrics import idcg_table, rank_metrics, top_k_indices

K = [5, 10, 20]


def scores_ranking(order, n=50):
    """Build a score vector whose descending order is exactly ``order`` first."""
    s = np.zeros(n)
    for rank, item in enumerate(order):
        s[item] = 100.0 - rank
    return s


def test_top_k_indices_orders_by_descending_score():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k_indices(s, 3).tolist() == [1, 3, 2]


def test_top_k_indices_handles_k_larger_than_candidates():
    s = np.array([0.2, 0.8])
    assert top_k_indices(s, 10).tolist() == [1, 0]


def test_perfect_ranking_scores_one():
    """Relevant items in the top positions -> every metric maxes out."""
    rel = [7, 3, 11]
    s = scores_ranking([7, 3, 11])
    m = rank_metrics(s, seen=[], relevant=rel, k_list=K)
    assert m["hit@5"] == 1.0
    assert m["recall@5"] == 1.0
    assert m["mrr@5"] == 1.0
    assert m["ndcg@5"] == pytest.approx(1.0)
    assert m["precision@5"] == pytest.approx(3 / 5)  # bounded by |R|/K


def test_no_relevant_items_in_top_k_scores_zero():
    s = scores_ranking([0, 1, 2, 3, 4])
    m = rank_metrics(s, seen=[], relevant=[40, 41], k_list=[5])
    assert m["hit@5"] == 0.0
    assert m["precision@5"] == 0.0
    assert m["recall@5"] == 0.0
    assert m["mrr@5"] == 0.0
    assert m["ndcg@5"] == 0.0


def test_mrr_is_reciprocal_rank_of_first_hit():
    # relevant item placed third -> MRR = 1/3
    s = scores_ranking([0, 1, 9])
    m = rank_metrics(s, seen=[], relevant=[9], k_list=[5, 10])
    assert m["mrr@5"] == pytest.approx(1 / 3)


def test_mrr_is_zero_when_first_hit_falls_outside_k():
    # single relevant item at rank 7: inside K=10, outside K=5
    s = scores_ranking([0, 1, 2, 3, 4, 5, 9])
    m = rank_metrics(s, seen=[], relevant=[9], k_list=[5, 10])
    assert m["mrr@5"] == 0.0
    assert m["mrr@10"] == pytest.approx(1 / 7)


def test_ndcg_matches_hand_computed_value():
    """One relevant item at rank 2, one held-out item total.

    DCG  = 1/log2(3)
    IDCG = 1/log2(2) = 1
    """
    s = scores_ranking([0, 9])
    m = rank_metrics(s, seen=[], relevant=[9], k_list=[5])
    assert m["ndcg@5"] == pytest.approx(1.0 / math.log2(3))


def test_ndcg_denominator_uses_min_of_relevant_and_k():
    """With |R| > K the ideal list is only K long, so a perfect top-K scores 1.0."""
    rel = list(range(20, 30))  # 10 relevant items
    s = scores_ranking(rel[:5])  # top 5 all relevant
    m = rank_metrics(s, seen=[], relevant=rel, k_list=[5])
    assert m["ndcg@5"] == pytest.approx(1.0)


def test_recall_is_bounded_by_k_over_relevant_count():
    """A user with more held-out items than K cannot reach recall 1.0 -- documented
    convention, and the reason Recall@10 looks low in the results table."""
    rel = list(range(20, 40))  # 20 relevant
    s = scores_ranking(rel[:10])
    m = rank_metrics(s, seen=[], relevant=rel, k_list=[10])
    assert m["recall@10"] == pytest.approx(10 / 20)


def test_seen_items_are_masked_and_cannot_be_recommended():
    """Training solves must never appear in the ranking; if they leaked, a model that
    simply echoed a user's history would score near-perfectly."""
    s = scores_ranking([1, 2, 3, 9])
    m = rank_metrics(s, seen=[1, 2, 3], relevant=[9], k_list=[5])
    # with 1,2,3 masked, item 9 is now top-ranked
    assert m["mrr@5"] == pytest.approx(1.0)


def test_masking_does_not_mutate_caller_scores():
    s = scores_ranking([1, 9])
    before = s.copy()
    rank_metrics(s, seen=[1], relevant=[9], k_list=[5])
    assert np.array_equal(s, before)


def test_empty_relevant_set_is_rejected():
    with pytest.raises(ValueError):
        rank_metrics(np.zeros(10), seen=[], relevant=[], k_list=[5])


def test_metrics_are_monotone_in_k():
    """Hit, precision-mass and recall cannot decrease as the cut-off widens."""
    rng = np.random.default_rng(0)
    s = rng.random(200)
    rel = rng.choice(200, size=8, replace=False)
    m = rank_metrics(s, seen=[], relevant=rel, k_list=K)
    assert m["hit@5"] <= m["hit@10"] <= m["hit@20"]
    assert m["recall@5"] <= m["recall@10"] <= m["recall@20"]


def test_idcg_table_matches_closed_form():
    t = idcg_table([5])[5]
    assert t[0] == pytest.approx(1.0)
    assert t[1] == pytest.approx(1.0 + 1 / math.log2(3))
