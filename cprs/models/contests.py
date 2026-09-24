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
