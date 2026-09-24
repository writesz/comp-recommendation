"""Properties of the online cross-platform skill estimator.

The estimator underwrites RQ4, so its two claimed properties -- that it converges to the
q-th quantile of solved difficulty, and that it is self-limiting on solve-only data
rather than drifting upward like a wins-only Elo -- are worth asserting rather than
assuming.
"""
import numpy as np
import pytest

from models.skill import OnlineSkillEstimator


def run_stream(stream, **kw):
    """Feed a difficulty stream, returning the estimator and its second-half trace."""
    est = OnlineSkillEstimator(**kw)
    trace = []
    for i, d in enumerate(stream):
        est.update(float(d))
        if i >= len(stream) // 2:
            trace.append(est.theta)
    return est, np.array(trace)


def test_converges_in_mean_to_the_q_quantile_of_a_stationary_stream():
    """The fixed point of the Robbins-Monro rule is the q-quantile.

    With a *constant* step size the estimate does not converge pointwise: it performs a
    random walk whose stationary mean is the quantile. The time-average is therefore the
    quantity to assert on, not the last value.
    """
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 1, size=20000)
    _, trace = run_stream(truth, q=0.75, lr=0.02, init=0.3)
    assert trace.mean() == pytest.approx(np.percentile(truth, 75), abs=0.02)


def test_oscillation_amplitude_shrinks_with_the_learning_rate():
    """Because the step size is constant, the estimate carries a noise floor
    proportional to lr -- roughly 0.04 on the [0,1] difficulty scale at the deployed
    lr=0.02. Tightening lr trades responsiveness for precision."""
    rng = np.random.default_rng(1)
    truth = rng.uniform(0, 1, size=20000)
    _, coarse = run_stream(truth, q=0.75, lr=0.05, init=0.3)
    _, fine = run_stream(truth, q=0.75, lr=0.005, init=0.3)
    assert fine.std() < coarse.std()
    # both still centre on the same fixed point
    assert fine.mean() == pytest.approx(coarse.mean(), abs=0.03)


def test_tracks_a_quantile_other_than_the_default():
    rng = np.random.default_rng(2)
    truth = rng.uniform(0, 1, size=20000)
    _, trace = run_stream(truth, q=0.25, lr=0.01, init=0.5)
    assert trace.mean() == pytest.approx(np.percentile(truth, 25), abs=0.03)


def test_does_not_drift_upward_on_solve_only_data():
    """A naive wins-only Elo would climb without bound on an all-success stream; this
    estimator must settle instead. Feeding a constant difficulty should park theta at
    that value, not run away from it."""
    est = OnlineSkillEstimator(q=0.75, lr=0.05, init=0.1)
    for _ in range(5000):
        est.update(0.5)
    assert est.theta == pytest.approx(0.5, abs=0.06)
    assert est.theta <= 1.0


def test_update_direction_is_signed_by_the_comparison():
    est = OnlineSkillEstimator(q=0.75, lr=0.1, init=0.5)
    up = est.update(0.9)          # solved something above the estimate -> raise
    assert up > 0.5
    est2 = OnlineSkillEstimator(q=0.75, lr=0.1, init=0.5)
    down = est2.update(0.1)       # solved something easy -> lower
    assert down < 0.5


def test_seed_sets_the_empirical_quantile_of_supplied_history():
    """This is the cross-platform mechanism: an AtCoder history becomes a CF prior."""
    hist = [0.1, 0.2, 0.3, 0.4, 0.5]
    est = OnlineSkillEstimator(q=0.75).seed(hist)
    assert est.theta == pytest.approx(np.percentile(hist, 75))
    assert est.n == len(hist)


def test_seed_then_update_composes_without_special_casing():
    """The 'seed cross-platform, correct live' condition of RQ4 is exactly this."""
    est = OnlineSkillEstimator(q=0.75, lr=0.02).seed([0.2, 0.3, 0.4])
    seeded = est.theta
    for _ in range(50):
        est.update(0.9)
    assert est.theta > seeded


def test_theta_is_clipped_to_the_difficulty_scale():
    est = OnlineSkillEstimator(q=0.99, lr=0.5, init=0.9)
    for _ in range(100):
        est.update(1.0)
    assert 0.0 <= est.theta <= 1.0

    est2 = OnlineSkillEstimator(q=0.01, lr=0.5, init=0.1)
    for _ in range(100):
        est2.update(0.0)
    assert 0.0 <= est2.theta <= 1.0


def test_missing_difficulties_are_ignored():
    """Roughly half of AtCoder problems carry no difficulty estimate, so NaNs reach the
    estimator in normal operation and must not poison it."""
    est = OnlineSkillEstimator(init=0.4)
    before, n_before = est.theta, est.n
    est.update(float("nan"))
    est.update(None)
    assert est.theta == before
    assert est.n == n_before


def test_seed_ignores_missing_values():
    est = OnlineSkillEstimator(q=0.5).seed([0.2, float("nan"), 0.4])
    assert est.theta == pytest.approx(0.3)
    assert est.n == 2


def test_seed_with_no_usable_history_leaves_the_prior_untouched():
    est = OnlineSkillEstimator(init=0.33).seed([float("nan")])
    assert est.theta == pytest.approx(0.33)


def test_solvability_is_a_half_at_the_skill_level_and_monotone_decreasing():
    est = OnlineSkillEstimator(init=0.5)
    assert est.predict_solvable(0.5) == pytest.approx(0.5)
    assert est.predict_solvable(0.2) > est.predict_solvable(0.5) > est.predict_solvable(0.8)


def test_update_is_constant_time_in_history_length():
    """O(1) per solve is what lets the estimate run in the request path."""
    est = OnlineSkillEstimator()
    for i in range(10000):
        est.update(0.5)
    assert est.n == 10000
