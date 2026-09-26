"""
Cross-platform contest performance analysis.

Codeforces, AtCoder and LeetCode each expose contest history in a different
shape. This module normalises them into a single `ContestRecord` timeline and
derives the things a practising competitor actually wants to know: where the
rating is going, how consistent the results are, and which concrete habits the
record suggests changing.

The insight rules are deliberately conservative — each one only fires when
there is enough history to support it, so a user with three contests is not
told a story the data cannot carry.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Optional


@dataclass
class ContestRecord:
    """One rated (or unrated) contest appearance, normalised across platforms."""
    platform: str
    name: str
    timestamp: int                      # unix seconds
    place: Optional[int] = None
    old_rating: Optional[int] = None
    new_rating: Optional[int] = None
    delta: Optional[int] = None
    performance: Optional[int] = None
    rated: bool = True
    problems_solved: Optional[int] = None
    problems_total: Optional[int] = None

    @property
    def date(self) -> str:
        return datetime.fromtimestamp(self.timestamp, timezone.utc).strftime("%Y-%m-%d")

    def as_dict(self) -> dict:
        d = {
            "platform": self.platform,
            "name": self.name,
            "timestamp": self.timestamp,
            "date": self.date,
            "place": self.place,
            "old_rating": self.old_rating,
            "new_rating": self.new_rating,
            "delta": self.delta,
            "performance": self.performance,
            "rated": self.rated,
        }
        if self.problems_solved is not None:
            d["problems_solved"] = self.problems_solved
            d["problems_total"] = self.problems_total
        return d


# --- Normalisation ----------------------------------------------------------

def from_codeforces(history: list) -> list:
    """Normalise the Codeforces user.rating payload."""
    out = []
    for h in history:
        old, new = h.get("oldRating"), h.get("newRating")
        out.append(ContestRecord(
            platform="codeforces",
            name=h.get("contestName", "Contest"),
            timestamp=int(h.get("ratingUpdateTimeSeconds", 0)),
            place=h.get("rank"),
            old_rating=old,
            new_rating=new,
            delta=(new - old) if old is not None and new is not None else None,
            rated=True,
        ))
    return out


def from_atcoder(history: list) -> list:
    """Normalise the AtCoder users/<id>/history/json payload."""
    out = []
    for h in history:
        old, new = h.get("OldRating"), h.get("NewRating")
        rated = bool(h.get("IsRated"))
        end = h.get("EndTime", "")
        try:
            ts = int(datetime.fromisoformat(end).timestamp())
        except (ValueError, TypeError):
            ts = 0
        out.append(ContestRecord(
            platform="atcoder",
            name=h.get("ContestName") or h.get("ContestNameEn") or "Contest",
            timestamp=ts,
            place=h.get("Place"),
            old_rating=old,
            new_rating=new,
            delta=(new - old) if rated and old is not None and new is not None else None,
            performance=h.get("Performance") or None,
            rated=rated,
        ))
    return out


def from_codechef(history: list) -> list:
    """
    Normalise the CodeChef rating history scraped from the profile page.

    Like LeetCode, CodeChef publishes only the rating *after* each contest,
    so deltas are reconstructed by walking the timeline in order. Unlike
    LeetCode, its ratings sit on the same Elo-like scale as its problem
    difficulties, so they calibrate directly.
    """
    out = []
    previous = None
    for h in sorted(history, key=lambda x: x.get("end_date") or ""):
        new = h.get("rating")
        end = h.get("end_date") or ""
        try:
            ts = int(datetime.strptime(end, "%Y-%m-%d %H:%M:%S").timestamp())
        except (ValueError, TypeError):
            ts = 0
        out.append(ContestRecord(
            platform="codechef",
            name=h.get("name") or "Contest",
            timestamp=ts,
            place=h.get("rank"),
            old_rating=previous,
            new_rating=new,
            delta=(new - previous) if new is not None and previous is not None else None,
            rated=True,
        ))
        previous = new
    return out


def from_leetcode(payload: dict) -> list:
    """
    Normalise the LeetCode contest history.

    LeetCode reports only the rating *after* each contest, so deltas are
    reconstructed by walking the timeline in order.
    """
    out = []
    previous = None
    attended = [h for h in payload.get("history", []) if h.get("attended")]
    for h in sorted(attended, key=lambda x: x["contest"]["startTime"]):
        rating = h.get("rating")
        new = round(rating) if rating is not None else None
        out.append(ContestRecord(
            platform="leetcode",
            name=h["contest"].get("title", "Contest"),
            timestamp=int(h["contest"].get("startTime", 0)),
            place=h.get("ranking"),
            old_rating=previous,
            new_rating=new,
            delta=(new - previous) if new is not None and previous is not None else None,
            rated=True,
            problems_solved=h.get("problemsSolved"),
            problems_total=h.get("totalProblems"),
        ))
        previous = new
    return out


# --- Analysis ---------------------------------------------------------------

def _trend(deltas: list) -> str:
    if not deltas:
        return "stable"
    total = sum(deltas)
    if total > 15:
        return "improving"
    if total < -15:
        return "declining"
    return "stable"


def analyse(records: list, recent_window: int = 10) -> dict:
    """
    Summarise a contest timeline and derive improvement points.

    Returns a dict with a summary block, the timeline itself, and a list of
    plain-language insights.
    """
    records = sorted([r for r in records if r.timestamp], key=lambda r: r.timestamp)
    if not records:
        return {"summary": {}, "timeline": [], "insights": [], "best": None, "worst": None}

    rated = [r for r in records if r.rated and r.delta is not None]
    # The debut contest reports a delta measured from a rating of zero, which
    # would otherwise make "all-time change" equal the current rating.
    deltas = [r.delta for r in rated if r.old_rating]
    rated_with_rating = [r for r in records if r.new_rating is not None]

    current = rated_with_rating[-1].new_rating if rated_with_rating else None
    peak_rec = max(rated_with_rating, key=lambda r: r.new_rating) if rated_with_rating else None

    established = [r for r in rated if r.old_rating]
    recent = established[-recent_window:]
    earlier = established[:-recent_window]
    recent_deltas = [r.delta for r in recent]

    best = max(established, key=lambda r: r.delta) if established else None
    worst = min(established, key=lambda r: r.delta) if established else None

    summary = {
        "contests": len(records),
        "rated_contests": len(rated),
        "current_rating": current,
        "peak_rating": peak_rec.new_rating if peak_rec else None,
        "peak_contest": peak_rec.name if peak_rec else None,
        "peak_date": peak_rec.date if peak_rec else None,
        "net_change": sum(deltas) if deltas else 0,
        "recent_change": sum(recent_deltas) if recent_deltas else 0,
        "recent_window": len(recent),
        "avg_delta": round(mean(deltas), 1) if deltas else 0,
        "volatility": round(pstdev(deltas), 1) if len(deltas) > 1 else 0,
        "positive_rate": round(sum(1 for d in deltas if d > 0) / len(deltas), 3) if deltas else 0,
        "best_place": min((r.place for r in records if r.place), default=None),
        "trend": _trend(recent_deltas),
        "first_date": records[0].date,
        "last_date": records[-1].date,
    }

    insights = _insights(records, rated, deltas, recent, earlier, summary)

    return {
        "summary": summary,
        "timeline": [r.as_dict() for r in records],
        "best": best.as_dict() if best else None,
        "worst": worst.as_dict() if worst else None,
        "insights": insights,
    }


def _insights(records, rated, deltas, recent, earlier, summary) -> list:
    """
    Derive improvement points. Each insight is {kind, title, detail} where kind
    is one of good / watch / info, so the UI can colour them.
    """
    out = []
    if not rated:
        out.append({
            "kind": "info",
            "title": "No rated contests yet",
            "detail": "Contest results are the clearest signal of progress — "
                      "entering a few rated rounds will make this report much more useful.",
        })
        return out

    # Pace: the recent window measured against the user's own history.
    # Phrased as pace rather than direction, because a user can be gaining
    # rating overall (an "improving" trend) while still gaining it more slowly
    # than they used to — saying "form is down" there would contradict the
    # trend shown beside it.
    if len(recent) >= 3 and len(earlier) >= 3:
        recent_avg = mean(r.delta for r in recent)
        earlier_avg = mean(r.delta for r in earlier)
        shift = recent_avg - earlier_avg
        if shift > 5:
            out.append({
                "kind": "good",
                "title": "You're climbing faster than you used to",
                "detail": f"Across your last {len(recent)} contests you average "
                          f"{recent_avg:+.1f} rating per round, against {earlier_avg:+.1f} "
                          f"over everything before that. Whatever changed recently is working.",
            })
        elif shift < -5:
            direction = "still gaining, but" if recent_avg > 0 else "now losing rating and"
            out.append({
                "kind": "watch",
                "title": "Your rate of progress has slowed",
                "detail": f"You're {direction} averaging {recent_avg:+.1f} per round over your "
                          f"last {len(recent)} contests, against {earlier_avg:+.1f} historically. "
                          f"Rating gets harder to win as it rises, so some slowdown is normal — "
                          f"but it's worth reviewing contest frequency and problem mix.",
            })

    # Plateau: enough contests, but little net movement.
    if len(recent) >= 5 and abs(summary["recent_change"]) < 25:
        out.append({
            "kind": "watch",
            "title": "You've plateaued",
            "detail": f"Only {summary['recent_change']:+d} rating across your last "
                      f"{len(recent)} contests. A plateau usually means the problems you can "
                      f"already solve are no longer stretching you — target the topics in your "
                      f"weak list rather than more of the same.",
        })

    # Consistency.
    if summary["volatility"] and len(deltas) >= 8:
        if summary["volatility"] > 80:
            out.append({
                "kind": "watch",
                "title": "Results are volatile",
                "detail": f"Your per-contest swing is ±{summary['volatility']:.0f} rating. "
                          f"Large swings usually point to inconsistent early-problem speed — "
                          f"one slow start costs more than one hard problem gains.",
            })
        elif summary["volatility"] < 35:
            out.append({
                "kind": "good",
                "title": "Very consistent results",
                "detail": f"Your per-contest swing is only ±{summary['volatility']:.0f} rating. "
                          f"You reliably perform at your level — the next gain will come from "
                          f"attempting harder problems, not from steadier play.",
            })

    # Win rate.
    pr = summary["positive_rate"]
    if len(deltas) >= 6:
        if pr >= 0.6:
            out.append({
                "kind": "good",
                "title": f"You gain rating in {round(pr * 100)}% of contests",
                "detail": "A positive majority means you are competing below your true level. "
                          "Consider moving up a division or entering harder rounds.",
            })
        elif pr <= 0.35:
            out.append({
                "kind": "watch",
                "title": f"You lose rating in {round((1 - pr) * 100)}% of contests",
                "detail": "Losing more often than not usually means the round difficulty is "
                          "above your current solving speed. Practice at your level to rebuild, "
                          "rather than entering every round.",
            })

    # Distance from peak.
    if summary["current_rating"] and summary["peak_rating"]:
        gap = summary["peak_rating"] - summary["current_rating"]
        if gap > 100:
            out.append({
                "kind": "watch",
                "title": f"{gap} points below your peak",
                "detail": f"You peaked at {summary['peak_rating']} on {summary['peak_date']}. "
                          f"Getting back there is usually about restoring contest frequency "
                          f"and reviewing the rounds where you lost the most.",
            })
        elif gap == 0 and len(rated) >= 5:
            out.append({
                "kind": "good",
                "title": "You're at your peak rating",
                "detail": f"{summary['current_rating']} is the highest you've been. "
                          f"Keep the current routine.",
            })

    # LeetCode-specific: solve count within the round.
    lc = [r for r in records if r.problems_solved is not None and r.problems_total]
    if len(lc) >= 5:
        solved_avg = mean(r.problems_solved for r in lc)
        total = max(r.problems_total for r in lc)
        if solved_avg < total - 1:
            out.append({
                "kind": "info",
                "title": f"You average {solved_avg:.1f} of {total} problems in LeetCode contests",
                "detail": "The last problem in a round is usually where the rating is. "
                          "Practising one difficulty tier above your comfort zone moves this number.",
            })

    # Cadence: are they actually competing?
    if len(records) >= 2:
        span_days = (records[-1].timestamp - records[0].timestamp) / 86400
        if span_days > 0:
            per_month = len(records) / (span_days / 30.4)
            last_gap = (datetime.now(timezone.utc).timestamp() - records[-1].timestamp) / 86400
            if last_gap > 60:
                out.append({
                    "kind": "watch",
                    "title": f"No contest in {int(last_gap)} days",
                    "detail": "Rating decays in relevance once you stop competing. "
                              "Re-entering regularly is the single fastest way to make this "
                              "report meaningful again.",
                })
            elif per_month < 1 and len(records) >= 4:
                out.append({
                    "kind": "info",
                    "title": f"You compete about {per_month:.1f} times a month",
                    "detail": "Competitors who improve fastest tend to enter 2–4 rated rounds "
                              "a month — enough signal to learn from, not so much that there's "
                              "no time to review.",
                })

    return out


def merge(per_platform: dict) -> dict:
    """
    Combine per-platform analyses into a cross-platform view.

    Ratings are not comparable across platforms, so nothing is averaged — the
    merged view reports each platform's standing side by side and pools only
    the things that are genuinely comparable (contest counts, cadence, form).
    """
    all_records = []
    for analysis in per_platform.values():
        all_records.extend(analysis.get("timeline", []))
    all_records.sort(key=lambda r: r["timestamp"])

    total = len(all_records)
    rated = [r for r in all_records if r.get("delta") is not None]
    improving = sum(
        1 for a in per_platform.values() if a.get("summary", {}).get("trend") == "improving"
    )
    declining = sum(
        1 for a in per_platform.values() if a.get("summary", {}).get("trend") == "declining"
    )

    return {
        "total_contests": total,
        "platforms": len(per_platform),
        "positive_rate": (
            round(sum(1 for r in rated if r["delta"] > 0) / len(rated), 3) if rated else 0
        ),
        "first_date": all_records[0]["date"] if all_records else None,
        "last_date": all_records[-1]["date"] if all_records else None,
        "platforms_improving": improving,
        "platforms_declining": declining,
    }


# --- Difficulty calibration -------------------------------------------------
#
# Contest rating and practice history answer different questions, and the
# recommender should take each from the source that actually knows.
#
# Rating is an Elo estimate: by construction a competitor rated R has roughly
# an even chance on a problem rated R, solved alone and under time pressure.
# The 75th percentile of a user's *solved* problems is a much weaker proxy —
# it counts problems solved with an editorial open, upsolved after the round,
# or simply whatever difficulty band the user happens to grind, so it drifts
# with practice habits rather than tracking ability.
#
# Contests are far too sparse to say anything per topic (a handful of problems
# per round against a taxonomy of ~100 tags), so topic targeting still comes
# from solve history. Contests set *how hard*; history sets *what*.

# LeetCode contest ratings are not on a problem-difficulty scale — LeetCode
# only labels problems Easy/Medium/Hard — so this range is an approximation
# calibrated against the platform's published rating distribution, and is
# reported with lower confidence than the Codeforces and AtCoder mappings.
LC_RATING_MIN = 1300
LC_RATING_MAX = 3000

_INACTIVE_DAYS = 180


def difficulty_calibration(analysis: dict, platform: str) -> Optional[dict]:
    """
    Derive a target difficulty on the unified [0, 1] scale from contest results.

    Returns {level, stretch, confidence, basis, rating, source} or None when
    there is no usable rated history.
    """
    from models.unified_schema import (
        normalize_cf_difficulty, normalize_ac_difficulty,
        normalize_cc_difficulty,
    )

    summary = analysis.get("summary") or {}
    rated = summary.get("rated_contests") or 0
    if not rated or summary.get("current_rating") is None:
        return None

    timeline = analysis.get("timeline") or []

    # Prefer recent *performance* ratings where the platform reports them
    # (AtCoder does): performance describes how the user actually did in that
    # round, so it tracks current form faster than the smoothed rating.
    performances = [
        t["performance"] for t in timeline[-5:]
        if t.get("performance") and t.get("rated")
    ]
    if len(performances) >= 3:
        rating = mean(performances)
        basis = f"average performance across your last {len(performances)} rated rounds"
    else:
        rating = summary["current_rating"]
        basis = "your current contest rating"

    if platform == "codeforces":
        level = normalize_cf_difficulty(int(rating))
    elif platform == "atcoder":
        level = normalize_ac_difficulty(float(rating))
    elif platform == "codechef":
        # CodeChef rates problems on the same scale it rates users, so the
        # normaliser that maps a problem rating maps a user rating too.
        level = normalize_cc_difficulty(float(rating))
    else:
        clamped = max(LC_RATING_MIN, min(LC_RATING_MAX, rating))
        level = (clamped - LC_RATING_MIN) / (LC_RATING_MAX - LC_RATING_MIN)

    if level is None:
        return None

    # Confidence grows with the number of rated rounds and decays once the
    # record goes stale — an old rating is a claim about a past self.
    confidence = min(1.0, rated / 8.0)
    if platform == "leetcode":
        confidence *= 0.7          # approximate rating->difficulty mapping
    last = timeline[-1]["timestamp"] if timeline else 0
    days_idle = (datetime.now(timezone.utc).timestamp() - last) / 86400 if last else 9e9
    if days_idle > _INACTIVE_DAYS:
        confidence *= max(0.3, 1.0 - (days_idle - _INACTIVE_DAYS) / 730)

    # Form decides how far above their level to aim. Stretching someone who is
    # already sliding compounds the problem; consolidation serves them better.
    trend = summary.get("trend")
    volatility = summary.get("volatility") or 0
    if trend == "improving":
        stretch, note = 0.12, "you're gaining rating, so these aim a little above your level"
    elif trend == "declining":
        stretch, note = 0.04, "you're losing rating, so these stay close to your level to rebuild"
    else:
        stretch, note = 0.08, "your rating is steady, so these sit just above your level"
    if volatility > 80:
        stretch = min(stretch, 0.06)
        note += "; your results swing a lot, so the range is kept tight"

    return {
        "level": round(float(level), 4),
        "stretch": stretch,
        "confidence": round(float(confidence), 3),
        "rating": int(rating),
        "basis": basis,
        "note": note,
        "source": platform,
        "rated_contests": rated,
    }


def blend_calibrations(calibrations: list, history_level: float) -> dict:
    """
    Combine per-platform contest calibrations with the history-derived level.

    Each calibration is weighted by its own confidence; whatever confidence is
    left over falls back to the practice-history estimate, so a user with no
    contest record is scored exactly as before.
    """
    usable = [c for c in calibrations if c]
    if not usable:
        return {
            "level": history_level,
            "stretch": 0.1,
            "confidence": 0.0,
            "sources": [],
            "explanation": "calibrated from the difficulty of problems you've solved",
        }

    total_conf = sum(c["confidence"] for c in usable)
    weight = min(1.0, total_conf)
    contest_level = sum(c["level"] * c["confidence"] for c in usable) / (total_conf or 1)

    level = weight * contest_level + (1 - weight) * history_level
    best = max(usable, key=lambda c: c["confidence"])

    labels = {"codeforces": "Codeforces", "atcoder": "AtCoder",
              "codechef": "CodeChef", "leetcode": "LeetCode"}
    names = ", ".join(labels.get(c["source"], c["source"]) for c in usable)

    return {
        "level": round(level, 4),
        "stretch": best["stretch"],
        "confidence": round(weight, 3),
        "sources": [c["source"] for c in usable],
        "contest_level": round(contest_level, 4),
        "history_level": round(history_level, 4),
        "explanation": (
            f"calibrated from your {names} contest record — {best['basis']}, "
            f"and {best['note']}"
        ),
    }
