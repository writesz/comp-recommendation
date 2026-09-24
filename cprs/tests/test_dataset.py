"""Invariants of the interaction dataset and the temporal split.

These run against the *committed* artefacts rather than synthetic fixtures, because the
claims they protect are claims about the data the report actually reports on. A leak
between train and test, or a k-core that never reached its fixed point, would inflate
every model at once and so would not show up as a suspicious ordering.

Skipped automatically if the derived artefacts have not been built.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from scripts.build_interactions import iterative_filter

DATA = Path(__file__).resolve().parents[1] / "data"
INTER = DATA / "interactions"

MIN_USER = 5
MIN_PROBLEM = 5

needs_data = pytest.mark.skipif(
    not (INTER / "matrix.npz").exists(),
    reason="derived interaction artefacts not built; run scripts/build_interactions.py",
)


@pytest.fixture(scope="module")
def artefacts():
    return {
        "full": sp.load_npz(INTER / "matrix.npz").tocsr(),
        "train": sp.load_npz(INTER / "train_matrix.npz").tocsr(),
        "mappings": json.loads((INTER / "mappings.json").read_text()),
        "test": json.loads((INTER / "test.json").read_text()),
        "meta": json.loads((INTER / "split_meta.json").read_text()),
    }


# --------------------------------------------------------------- k-core properties

def test_iterative_filter_reaches_a_fixed_point():
    """One pass is not enough: removing a user can drop a problem below threshold and
    vice versa, so the filter must iterate until stable."""
    events = [("u1", f"p{i}", i) for i in range(5)]          # u1 has 5
    events += [("u2", f"p{i}", i) for i in range(5)]          # p0..p4 have 2 solvers
    events += [("u3", "p0", 99)]                              # u3 has 1 -> dropped
    out = iterative_filter(events, min_user=5, min_problem=2)
    users = {u for u, _, _ in out}
    assert "u3" not in users
    # re-running must be a no-op: the output is already a fixed point
    assert iterative_filter(out, min_user=5, min_problem=2) == out


def test_iterative_filter_can_empty_a_dataset_that_supports_no_core():
    events = [("u1", "p1", 0), ("u2", "p2", 1)]
    assert iterative_filter(events, min_user=5, min_problem=5) == []


@needs_data
def test_committed_matrix_satisfies_the_k_core_thresholds(artefacts):
    m = artefacts["full"]
    per_user = np.asarray(m.sum(axis=1)).ravel()
    per_item = np.asarray(m.sum(axis=0)).ravel()
    assert per_user.min() >= MIN_USER
    assert per_item.min() >= MIN_PROBLEM


@needs_data
def test_matrix_is_binary_implicit_feedback(artefacts):
    """Solved = 1; a repeated solve must not become a 2 and silently act as a weight."""
    assert set(np.unique(artefacts["full"].data).tolist()) == {1}


# ------------------------------------------------------------ split integrity

@needs_data
def test_no_test_item_appears_in_training(artefacts):
    """The central anti-leakage property. If violated, every model could score a
    held-out item it had already been shown."""
    train, mappings, test = artefacts["train"], artefacts["mappings"], artefacts["test"]
    p_idx = {p: j for j, p in enumerate(mappings["problems"])}
    leaked = 0
    for row, user in enumerate(mappings["users"]):
        held = {p_idx[p] for p in test.get(user, []) if p in p_idx}
        if not held:
            continue
        trained = set(train.indices[train.indptr[row]:train.indptr[row + 1]].tolist())
        leaked += len(held & trained)
    assert leaked == 0


@needs_data
def test_train_and_test_partition_the_full_matrix(artefacts):
    full, train = artefacts["full"], artefacts["train"]
    n_test = sum(len(v) for v in artefacts["test"].values())
    assert train.nnz + n_test == full.nnz


@needs_data
def test_every_user_has_both_training_and_test_items(artefacts):
    """The report evaluates all 2,459 users; a user with an empty test set would be
    silently dropped from the averages."""
    train, mappings, test = artefacts["train"], artefacts["mappings"], artefacts["test"]
    per_user = np.diff(train.indptr)
    assert per_user.min() >= 1
    with_test = sum(1 for u in mappings["users"] if test.get(u))
    assert with_test == len(mappings["users"]) == artefacts["meta"]["users_with_test"]


@needs_data
def test_holdout_size_respects_the_configured_cap(artefacts):
    meta = artefacts["meta"]
    cap, frac = meta["holdout_n"], meta["holdout_frac"]
    full = artefacts["full"]
    per_user_total = np.asarray(full.sum(axis=1)).ravel()
    for row, user in enumerate(artefacts["mappings"]["users"]):
        n_held = len(artefacts["test"].get(user, []))
        assert n_held <= cap
        assert n_held <= max(1, int(per_user_total[row] * frac))


@needs_data
def test_reported_split_metadata_matches_the_artefacts(artefacts):
    meta, full, train = artefacts["meta"], artefacts["full"], artefacts["train"]
    assert (meta["n_users"], meta["n_problems"]) == full.shape
    assert meta["train_interactions"] == train.nnz
    assert meta["test_interactions"] == sum(len(v) for v in artefacts["test"].values())
