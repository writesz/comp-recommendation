"""Tests for cross-platform contest normalisation and analysis."""
import time

import pytest

from models import contests as C


DAY = 86400
NOW = int(time.time())


def rec(delta, old=1500, ts=None, platform="codeforces", place=100, rated=True, **kw):
    """Build one ContestRecord with a given rating delta."""
    return C.ContestRecord(
        platform=platform,
        name="Round",
        timestamp=ts if ts is not None else NOW - 10 * DAY,
        place=place,
        old_rating=old,
        new_rating=old + delta,
        delta=delta,
        rated=rated,
        **kw,
    )


# --- normalisation ----------------------------------------------------------

def test_from_codeforces_computes_delta():
    out = C.from_codeforces([
        {"contestName": "Round 1", "ratingUpdateTimeSeconds": NOW, "rank": 50,
         "oldRating": 1500, "newRating": 1580},
    ])
    assert len(out) == 1
    assert out[0].delta == 80
    assert out[0].platform == "codeforces"
    assert out[0].place == 50


def test_from_atcoder_marks_unrated_and_parses_time():
    out = C.from_atcoder([
        {"IsRated": True, "Place": 3, "OldRating": 2000, "NewRating": 2100,
         "Performance": 2500, "ContestName": "ABC 1", "EndTime": "2024-05-01T23:00:00+09:00"},
        {"IsRated": False, "Place": 9, "OldRating": 2100, "NewRating": 2100,
         "Performance": 0, "ContestName": "Unrated", "EndTime": "2024-06-01T23:00:00+09:00"},
    ])
    assert out[0].delta == 100 and out[0].rated is True
    assert out[1].rated is False and out[1].delta is None
    assert out[0].timestamp > 0


def test_from_atcoder_survives_bad_timestamp():
    out = C.from_atcoder([
        {"IsRated": True, "OldRating": 1, "NewRating": 2, "EndTime": "not-a-date"},
    ])
    assert out[0].timestamp == 0


def test_from_leetcode_reconstructs_deltas_in_order():
    payload = {"history": [
        {"attended": True, "rating": 1600, "ranking": 900, "problemsSolved": 3,
         "totalProblems": 4, "contest": {"title": "W2", "startTime": NOW}},
        {"attended": True, "rating": 1500, "ranking": 1200, "problemsSolved": 2,
         "totalProblems": 4, "contest": {"title": "W1", "startTime": NOW - DAY}},
    ]}
    out = C.from_leetcode(payload)
    # sorted oldest first
    assert [r.name for r in out] == ["W1", "W2"]
    assert out[0].delta is None          # no prior rating to diff against
    assert out[1].delta == 100
    assert out[1].problems_solved == 3


def test_from_leetcode_skips_unattended():
    payload = {"history": [
        {"attended": False, "rating": 1500, "contest": {"title": "X", "startTime": NOW}},
    ]}
    assert C.from_leetcode(payload) == []


# --- analysis ---------------------------------------------------------------

def test_analyse_empty():
    out = C.analyse([])
    assert out["summary"] == {} and out["timeline"] == [] and out["insights"] == []


def test_debut_contest_excluded_from_net_change():
    """A debut delta measured from rating 0 must not inflate all-time change."""
    records = [
        C.ContestRecord("codeforces", "Debut", NOW - 5 * DAY, old_rating=0,
                        new_rating=1400, delta=1400),
        rec(50, old=1400, ts=NOW - 4 * DAY),
        rec(-20, old=1450, ts=NOW - 3 * DAY),
    ]
    s = C.analyse(records)["summary"]
    assert s["net_change"] == 30           # not 1430
    assert s["current_rating"] == 1430


def test_peak_is_tracked_with_date():
    records = [
        rec(0, old=1500, ts=NOW - 5 * DAY),
        rec(0, old=1900, ts=NOW - 4 * DAY),
        rec(0, old=1600, ts=NOW - 3 * DAY),
    ]
    s = C.analyse(records)["summary"]
    assert s["peak_rating"] == 1900
    assert s["current_rating"] == 1600
    assert s["peak_date"]


def test_positive_rate_and_volatility():
    records = [rec(d, ts=NOW - (10 - i) * DAY) for i, d in enumerate([10, -5, 20, -5])]
    s = C.analyse(records)["summary"]
    assert s["positive_rate"] == 0.5
    assert s["volatility"] > 0


def test_timeline_is_sorted_oldest_first():
    records = [rec(5, ts=NOW - DAY), rec(5, ts=NOW - 9 * DAY)]
    tl = C.analyse(records)["timeline"]
    assert tl[0]["timestamp"] < tl[1]["timestamp"]


def test_records_without_timestamp_are_dropped():
    records = [rec(5, ts=0), rec(5, ts=NOW - DAY)]
    assert len(C.analyse(records)["timeline"]) == 1


def test_best_and_worst_rounds():
    records = [rec(d, ts=NOW - (10 - i) * DAY) for i, d in enumerate([5, 90, -70, 3])]
    out = C.analyse(records)
    assert out["best"]["delta"] == 90
    assert out["worst"]["delta"] == -70


# --- insights ---------------------------------------------------------------

def titles(out):
    return " | ".join(i["title"] for i in out["insights"])


def test_no_rated_contests_insight():
    records = [C.ContestRecord("atcoder", "Unrated", NOW - DAY, rated=False)]
    out = C.analyse(records)
    assert "No rated contests yet" in titles(out)


def test_pace_insight_does_not_contradict_improving_trend():
    """
    Gaining rating overall while gaining it more slowly than before must not
    produce a 'form is down' claim next to an 'improving' trend.
    """
    early = [rec(60, old=1500 + i * 60, ts=NOW - (40 - i) * DAY) for i in range(10)]
    late = [rec(5, old=2100 + i * 5, ts=NOW - (10 - i) * DAY) for i in range(10)]
    out = C.analyse(early + late)
    assert out["summary"]["trend"] == "improving"      # recent sum is positive
    assert "form is down" not in titles(out).lower()
    assert "rate of progress has slowed" in titles(out).lower()


def test_faster_progress_insight():
    early = [rec(2, old=1500, ts=NOW - (40 - i) * DAY) for i in range(10)]
    late = [rec(40, old=1600 + i * 40, ts=NOW - (10 - i) * DAY) for i in range(10)]
    out = C.analyse(early + late)
    assert "climbing faster" in titles(out).lower()


def test_plateau_insight():
    records = [rec(1, old=1500, ts=NOW - (10 - i) * DAY) for i in range(8)]
    assert "plateaued" in titles(C.analyse(records)).lower()


def test_below_peak_insight():
    records = [
        rec(0, old=2000, ts=NOW - 5 * DAY),
        rec(-300, old=2000, ts=NOW - 4 * DAY),
    ]
    assert "below your peak" in titles(C.analyse(records)).lower()


def test_inactivity_insight():
    records = [rec(5, old=1500, ts=NOW - (400 - i) * DAY) for i in range(5)]
    assert "no contest in" in titles(C.analyse(records)).lower()


def test_leetcode_problems_insight():
    records = [
        C.ContestRecord("leetcode", f"W{i}", NOW - (10 - i) * DAY, old_rating=1500,
                        new_rating=1505, delta=5, problems_solved=2, problems_total=4)
        for i in range(6)
    ]
    assert "of 4 problems" in titles(C.analyse(records))


def test_insights_are_well_formed():
    records = [rec(d, ts=NOW - (20 - i) * DAY) for i, d in enumerate([50, -30, 80, -10, 5, 60, -70, 20])]
    for i in C.analyse(records)["insights"]:
        assert i["kind"] in ("good", "watch", "info")
        assert i["title"] and i["detail"]


# --- merge ------------------------------------------------------------------

def test_merge_pools_only_comparable_things():
    a = C.analyse([rec(10, ts=NOW - 3 * DAY), rec(-5, ts=NOW - 2 * DAY)])
    b = C.analyse([rec(20, platform="leetcode", ts=NOW - DAY)])
    m = C.merge({"codeforces": a, "leetcode": b})
    assert m["total_contests"] == 3
    assert m["platforms"] == 2
    assert 0 <= m["positive_rate"] <= 1
    assert m["first_date"] <= m["last_date"]


def test_merge_empty():
    m = C.merge({})
    assert m["total_contests"] == 0 and m["first_date"] is None


# --- difficulty calibration -------------------------------------------------

def _analysis(deltas, platform="codeforces", start=1500, performances=None, ts_offset=1):
    recs = []
    rating = start
    for i, d in enumerate(deltas):
        perf = performances[i] if performances else None
        recs.append(C.ContestRecord(
            platform=platform, name=f"R{i}", timestamp=NOW - (ts_offset + len(deltas) - i) * DAY,
            old_rating=rating, new_rating=rating + d, delta=d, performance=perf, place=50,
        ))
        rating += d
    return C.analyse(recs)


def test_calibration_none_without_rated_history():
    assert C.difficulty_calibration({"summary": {}, "timeline": []}, "codeforces") is None


def test_calibration_maps_cf_rating_onto_unified_scale():
    a = _analysis([10] * 10, start=1600)
    cal = C.difficulty_calibration(a, "codeforces")
    # rating ends ~1700; CF scale is 800..3500
    assert 0.25 < cal["level"] < 0.40
    assert cal["source"] == "codeforces"
    assert cal["rated_contests"] == 10


def test_calibration_confidence_grows_with_contests():
    few = C.difficulty_calibration(_analysis([5] * 2), "codeforces")
    many = C.difficulty_calibration(_analysis([5] * 20), "codeforces")
    assert few["confidence"] < many["confidence"]
    assert many["confidence"] == 1.0


def test_calibration_leetcode_is_discounted():
    cf = C.difficulty_calibration(_analysis([5] * 20), "codeforces")
    lc = C.difficulty_calibration(_analysis([5] * 20, platform="leetcode"), "leetcode")
    assert lc["confidence"] < cf["confidence"]


def test_calibration_decays_when_stale():
    fresh = C.difficulty_calibration(_analysis([5] * 20, ts_offset=1), "codeforces")
    stale = C.difficulty_calibration(_analysis([5] * 20, ts_offset=900), "codeforces")
    assert stale["confidence"] < fresh["confidence"]


def test_calibration_prefers_recent_performance_when_available():
    a = _analysis([5] * 10, platform="atcoder", start=1200,
                  performances=[2400] * 10)
    cal = C.difficulty_calibration(a, "atcoder")
    assert "performance" in cal["basis"]
    # performance (2400) should dominate the smoothed rating (~1250)
    assert cal["rating"] == 2400


def test_stretch_reflects_form():
    up = C.difficulty_calibration(_analysis([40] * 10), "codeforces")
    down = C.difficulty_calibration(_analysis([-40] * 10), "codeforces")
    assert up["stretch"] > down["stretch"]


def test_stretch_tightens_when_volatile():
    swingy = C.difficulty_calibration(
        _analysis([300, -300] * 6), "codeforces")
    assert swingy["stretch"] <= 0.06


# --- blending ---------------------------------------------------------------

def test_blend_without_contests_falls_back_to_history():
    b = C.blend_calibrations([], 0.42)
    assert b["level"] == 0.42
    assert b["confidence"] == 0.0
    assert b["sources"] == []


def test_blend_full_confidence_ignores_history():
    cal = C.difficulty_calibration(_analysis([5] * 20), "codeforces")
    b = C.blend_calibrations([cal], history_level=0.99)
    assert b["confidence"] == 1.0
    assert abs(b["level"] - cal["level"]) < 1e-6


def test_blend_partial_confidence_mixes_with_history():
    cal = C.difficulty_calibration(_analysis([5] * 2), "codeforces")
    b = C.blend_calibrations([cal], history_level=0.9)
    assert cal["level"] < b["level"] < 0.9


def test_blend_combines_multiple_platforms():
    cf = C.difficulty_calibration(_analysis([5] * 10), "codeforces")
    lc = C.difficulty_calibration(_analysis([5] * 10, platform="leetcode"), "leetcode")
    b = C.blend_calibrations([cf, lc], history_level=0.5)
    assert set(b["sources"]) == {"codeforces", "leetcode"}
    assert b["explanation"]


def test_blend_level_stays_in_range():
    for hist in (0.0, 0.5, 1.0):
        cal = C.difficulty_calibration(_analysis([5] * 3), "codeforces")
        b = C.blend_calibrations([cal], hist)
        assert 0.0 <= b["level"] <= 1.0
