"""
Regression tests for weak-topic selection.

The original rule scored any never-attempted tag at 1.5, above the worst
possible demonstrated weakness (1.3). Because each platform carries its own
tag vocabulary, that declared every foreign-platform tag to be a user's
weakest area and collapsed recommendations onto a single platform.
"""
import pytest

from models.recommender import RecommenderEngine, TopicMastery, UserProfile


@pytest.fixture
def engine(tmp_path):
    """A tiny corpus with a lopsided tag vocabulary, as the real one has."""
    import json
    problems = []
    # 'dp' and 'graphs' are common; 'design' is a rare foreign-platform tag.
    for i in range(40):
        problems.append({
            "cprs_id": f"cf:{i}", "platform": "codeforces", "name": f"CF {i}",
            "url": "", "difficulty_normalized": 0.5,
            "tags_unified": ["dp"] if i % 2 else ["graphs"], "solve_count": 100,
        })
    for i in range(2):
        problems.append({
            "cprs_id": f"lc:{i}", "platform": "leetcode", "name": f"LC {i}",
            "url": "", "difficulty_normalized": 0.5,
            "tags_unified": ["design"], "solve_count": 100,
        })
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(problems))
    return RecommenderEngine(str(p))


def profile_with(**topics):
    prof = UserProfile(handle="u", platform="codeforces")
    prof.topic_mastery = {
        name: TopicMastery(topic=name, problems_attempted=a, problems_solved=s,
                           solve_rate=(s / a if a else 0.0))
        for name, (a, s) in topics.items()
    }
    return prof


def test_demonstrated_weakness_outranks_never_attempted():
    """The core regression: failing at something beats never trying it."""
    eng_profile = profile_with(dp=(20, 2))       # 10% solve rate, well evidenced
    from models.recommender import RecommenderEngine
    # score directly, no corpus needed for the mastery branch
    class Stub(RecommenderEngine):
        def __init__(self):
            self.all_tags = ["dp", "design"]
            self._tag_prevalence = {"dp": 40, "design": 40}
    scores = Stub()._topic_scores(eng_profile)
    assert scores["dp"] > scores["design"]


def test_weak_topic_ranking_prefers_evidence(engine):
    prof = profile_with(dp=(20, 2), graphs=(2, 0))
    scores = engine._topic_scores(prof)
    # both are failing, but dp has ten times the evidence
    assert scores["dp"] > scores["graphs"]


def test_rare_unexplored_tags_are_discounted(engine):
    """A tag on two problems is not a meaningful gap."""
    prof = profile_with(dp=(10, 8), graphs=(10, 8))
    scores = engine._topic_scores(prof)
    assert scores["design"] < engine.UNEXPLORED_PRIOR


def test_unexplored_still_surfaces_when_nothing_is_weak(engine):
    """Exploration must not be switched off — it's the cross-platform value."""
    prof = profile_with(dp=(30, 30), graphs=(30, 30))   # perfect solve rate
    weak = engine._find_weak_topics(prof)
    assert "design" in weak


def test_other_prefixed_tags_are_never_targets(engine):
    prof = profile_with(**{"other:interactive": (10, 1), "dp": (10, 5)})
    scores = engine._topic_scores(prof)
    assert "other:interactive" not in scores


def test_unattempted_mastery_entries_ignored(engine):
    prof = profile_with(dp=(0, 0))
    scores = engine._topic_scores(prof)
    assert "dp" not in scores or scores.get("dp", 0) == engine.UNEXPLORED_PRIOR


def test_topic_weights_never_zero(engine):
    prof = profile_with(dp=(10, 10))
    w = engine._compute_topic_weights(prof, ["dp", "design", "unknown_tag"])
    assert all(v >= 0.1 for v in w.values())


def test_recommendations_are_not_monopolised_by_foreign_tags(engine):
    """
    A user with only Codeforces history must not have every slot taken by the
    two LeetCode problems carrying an unseen tag.
    """
    prof = profile_with(dp=(20, 14), graphs=(20, 15))
    prof.difficulty_level = 0.5
    recs = engine.recommend(prof, n=10)
    platforms = {r.platform for r in recs}
    assert "codeforces" in platforms
    lc = sum(1 for r in recs if r.platform == "leetcode")
    assert lc <= 2      # the corpus only has 2, and they must not crowd out CF
