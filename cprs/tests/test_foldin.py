"""
Tests for serving unseen users from a trained CF model.

ALS learns a user-factor table, not a function, so the application cannot score
anyone outside the training cohort by lookup. `fold_in` re-applies the ALS user
step with the item factors held fixed. These tests pin the properties that
makes it safe to serve: it recovers what training learned, it is invariant to
things it should be invariant to, and it degrades to "no opinion" rather than
to nonsense.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from models.collaborative import CollaborativeModel


@pytest.fixture(scope="module")
def synthetic():
    """A small matrix with three clear user archetypes."""
    rng = np.random.default_rng(7)
    n_users, n_items = 90, 60
    M = np.zeros((n_users, n_items))
    for u in range(n_users):
        block = u % 3                     # which third of the catalogue they solve
        lo, hi = block * 20, block * 20 + 20
        picks = rng.choice(np.arange(lo, hi), size=12, replace=False)
        M[u, picks] = 1
    return sp.csr_matrix(M)


@pytest.fixture(scope="module")
def model(synthetic):
    return CollaborativeModel(factors=8, iterations=30, random_state=0).fit(synthetic)


def test_fold_in_returns_right_shape(model):
    assert model.fold_in([1, 2, 3]).shape == (model.item_factors.shape[1],)


def test_no_history_gives_no_opinion(model):
    """An empty history must score every item equally, not arbitrarily."""
    x = model.fold_in([])
    assert np.allclose(x, 0.0)
    scores = model.score_folded([])
    assert np.allclose(scores, scores[0])


def test_fold_in_recovers_trained_factors(synthetic, model):
    """
    Fold-in is the ALS user step, so on a converged model it should reproduce
    the factors training learned for that same user.
    """
    sims = []
    for u in range(synthetic.shape[0]):
        cols = synthetic.indices[synthetic.indptr[u]:synthetic.indptr[u + 1]]
        trained = model.user_factors[u]
        folded = model.fold_in(cols)
        denom = np.linalg.norm(trained) * np.linalg.norm(folded)
        if denom:
            sims.append(trained @ folded / denom)
    assert np.mean(sims) > 0.95, f"mean cosine only {np.mean(sims):.3f}"


def test_duplicate_history_is_idempotent(model):
    """Solving a problem twice is still one solve."""
    a = model.fold_in([4, 5, 6])
    b = model.fold_in([4, 5, 5, 6, 6, 6])
    assert np.allclose(a, b)


def test_order_does_not_matter(model):
    assert np.allclose(model.fold_in([1, 9, 3]), model.fold_in([3, 1, 9]))


def test_folded_user_ranks_own_archetype_highest(synthetic, model):
    """
    A user who only solves the middle block should be scored towards that
    block — otherwise fold-in is not recovering anything meaningful.
    """
    cols = list(range(20, 32))
    scores = model.score_folded(cols)
    unseen = [c for c in range(60) if c not in cols]
    top = max(unseen, key=lambda c: scores[c])
    assert 20 <= top < 40, f"top unseen item {top} came from the wrong block"


def test_more_history_moves_the_estimate(model):
    """Fold-in must actually respond to evidence."""
    a = model.fold_in([0])
    b = model.fold_in(list(range(0, 15)))
    assert not np.allclose(a, b)


def test_save_load_round_trip(model, tmp_path):
    ids = [f"cf:{i}" for i in range(model.item_factors.shape[0])]
    path = tmp_path / "m.npz"
    model.save(path, problem_ids=ids)

    loaded = CollaborativeModel.load(path)
    assert np.allclose(loaded.item_factors, model.item_factors)
    assert loaded.factors == model.factors
    assert loaded.alpha == model.alpha
    assert loaded.regularization == model.regularization
    assert loaded.problem_ids == ids
    assert loaded.column_of["cf:5"] == 5


def test_loaded_model_folds_in_identically(model, tmp_path):
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = CollaborativeModel.load(path)
    cols = [2, 7, 11]
    assert np.allclose(loaded.fold_in(cols), model.fold_in(cols))


def test_save_rejects_mismatched_problem_ids(model, tmp_path):
    with pytest.raises(AssertionError):
        model.save(tmp_path / "bad.npz", problem_ids=["cf:1", "cf:2"])


def test_load_without_problem_ids_is_usable(model, tmp_path):
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = CollaborativeModel.load(path)
    assert loaded.problem_ids is None
    assert loaded.column_of == {}
    assert loaded.fold_in([1, 2]).shape == (model.factors,)


def test_fold_in_before_load_is_rejected():
    with pytest.raises(AssertionError):
        CollaborativeModel().fold_in([1, 2])
