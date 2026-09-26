"""
Content-based recommendation engine for competitive programming problems.

Given a user's submission history (from Codeforces), the recommender:
1. Builds a skill profile (per-topic mastery + difficulty level)
2. Identifies weak topics and appropriate difficulty range
3. Scores all problems in the unified dataset
4. Returns ranked recommendations, optionally cross-platform

Scoring formula:
    score = topic_gap_weight * difficulty_fit * freshness_bonus

Where:
- topic_gap_weight: higher for topics the user is weak in
- difficulty_fit: gaussian around the user's current level (slightly above)
- freshness_bonus: prefer problems the user hasn't seen
"""
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


def _codechef_epoch(sub: dict) -> float:
    """
    Sort key for a CodeChef submission.

    The widget renders times as "08:36 PM 10/06/26" (day/month/two-digit year)
    rather than a unix timestamp. Unparseable or missing times sort oldest so
    that a markup change degrades ordering rather than raising.
    """
    raw = (sub or {}).get("time")
    if not raw:
        return 0.0
    try:
        return datetime.strptime(raw.strip(), "%I:%M %p %d/%m/%y").timestamp()
    except (ValueError, TypeError):
        return 0.0


class TopicMastery(BaseModel):
    """User's mastery level for a single topic."""
    topic: str
    problems_attempted: int = 0
    problems_solved: int = 0
    solve_rate: float = 0.0
    avg_difficulty: float = 0.0
    max_difficulty: float = 0.0


class UserProfile(BaseModel):
    """A user's skill profile derived from submission history."""
    handle: str
    platform: str = "codeforces"
    rating: Optional[int] = None
    total_solved: int = 0
    total_attempted: int = 0
    overall_solve_rate: float = 0.0
    difficulty_level: float = 0.0  # normalized 0-1
    topic_mastery: dict = Field(default_factory=dict)  # topic -> TopicMastery
    solved_problem_ids: set = Field(default_factory=set)

    class Config:
        arbitrary_types_allowed = True


class Recommendation(BaseModel):
    """A single problem recommendation with explanation."""
    cprs_id: str
    platform: str
    name: str
    url: str
    difficulty_normalized: Optional[float] = None
    tags: list = Field(default_factory=list)
    score: float = 0.0
    reasons: list = Field(default_factory=list)


class RecentPerformance(BaseModel):
    """Analysis of user's recent problem-solving performance."""
    last_n: int = 0
    solved: int = 0
    attempted: int = 0
    solve_rate: float = 0.0
    avg_difficulty: float = 0.0
    topics_practiced: list = Field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable


class RecommenderEngine:
    """Content-based recommender for competitive programming problems."""

    def __init__(self, dataset_path: str):
        """Load the unified dataset."""
        with open(dataset_path) as f:
            self.problems = json.load(f)
        logger.info(f"Loaded {len(self.problems)} problems")

        # Index by cprs_id for fast lookup
        self.problem_index = {p["cprs_id"]: p for p in self.problems}

        # Collect all tags
        all_tags = set()
        self._tag_prevalence = Counter()
        for p in self.problems:
            tags = p.get("tags_unified", [])
            all_tags.update(tags)
            self._tag_prevalence.update(tags)
        self.all_tags = sorted(all_tags)

    def build_user_profile(self, submissions: list, handle: str, rating: Optional[int] = None) -> UserProfile:
        """
        Build a user profile from Codeforces submissions.

        Args:
            submissions: list of CF submission dicts from the API
            handle: user's CF handle
            rating: user's current rating (optional)
        """
        profile = UserProfile(handle=handle, rating=rating)

        # Track per-problem best verdict
        problem_verdicts = {}  # (contestId, index) -> best_verdict
        problem_tags = {}
        problem_ratings = {}

        for sub in submissions:
            problem = sub.get("problem", {})
            contest_id = problem.get("contestId")
            index = problem.get("index", "")
            verdict = sub.get("verdict", "")
            key = (contest_id, index)

            tags = problem.get("tags", [])
            rating_val = problem.get("rating")

            if key not in problem_verdicts:
                problem_verdicts[key] = verdict
                problem_tags[key] = tags
                if rating_val:
                    problem_ratings[key] = rating_val
            elif verdict == "OK":
                problem_verdicts[key] = "OK"

        # Compute stats
        profile.total_attempted = len(problem_verdicts)
        solved_keys = {k for k, v in problem_verdicts.items() if v == "OK"}
        profile.total_solved = len(solved_keys)
        profile.overall_solve_rate = (
            profile.total_solved / profile.total_attempted
            if profile.total_attempted > 0 else 0.0
        )

        # Build solved set
        profile.solved_problem_ids = {
            f"cf:{k[0]}{k[1]}" for k in solved_keys if k[0]
        }

        # Per-topic mastery
        topic_stats = defaultdict(lambda: {
            "attempted": 0, "solved": 0, "difficulties": []
        })

        from models.unified_schema import unify_tags

        for key, verdict in problem_verdicts.items():
            tags = unify_tags(problem_tags.get(key, []))
            rating_val = problem_ratings.get(key)
            is_solved = verdict == "OK"

            for tag in tags:
                topic_stats[tag]["attempted"] += 1
                if is_solved:
                    topic_stats[tag]["solved"] += 1
                if rating_val:
                    from models.unified_schema import normalize_cf_difficulty
                    norm_diff = normalize_cf_difficulty(rating_val)
                    if norm_diff is not None:
                        topic_stats[tag]["difficulties"].append(norm_diff)

        profile.topic_mastery = {}
        for topic, stats in topic_stats.items():
            diffs = stats["difficulties"]
            profile.topic_mastery[topic] = TopicMastery(
                topic=topic,
                problems_attempted=stats["attempted"],
                problems_solved=stats["solved"],
                solve_rate=stats["solved"] / stats["attempted"] if stats["attempted"] > 0 else 0.0,
                avg_difficulty=float(np.mean(diffs)) if diffs else 0.0,
                max_difficulty=float(max(diffs)) if diffs else 0.0,
            )

        # Overall difficulty level
        solved_diffs = []
        for key in solved_keys:
            if key in problem_ratings:
                from models.unified_schema import normalize_cf_difficulty
                nd = normalize_cf_difficulty(problem_ratings[key])
                if nd is not None:
                    solved_diffs.append(nd)

        if solved_diffs:
            # Use 75th percentile of solved difficulties as user level
            profile.difficulty_level = float(np.percentile(solved_diffs, 75))
        elif rating:
            from models.unified_schema import normalize_cf_difficulty
            profile.difficulty_level = normalize_cf_difficulty(rating) or 0.3

        return profile

    def build_atcoder_profile(self, submissions: list, handle: str) -> UserProfile:
        """
        Build a user profile from AtCoder submissions.
        AtCoder submissions from kenkoooo API have different structure than CF.
        """
        from models.unified_schema import normalize_ac_difficulty

        profile = UserProfile(handle=handle, platform="atcoder")

        # Load difficulty models for normalization
        import json as _json
        models_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "ac_models.json"
        ac_models = {}
        if models_path.exists():
            with open(models_path) as f:
                ac_models = _json.load(f)

        # Track per-problem best result
        problem_results = {}  # problem_id -> best_result
        problem_diffs = {}

        for sub in submissions:
            pid = sub.get("problem_id", "")
            result = sub.get("result", "")
            key = pid

            if key not in problem_results:
                problem_results[key] = result
                if pid in ac_models and "difficulty" in ac_models[pid]:
                    problem_diffs[key] = ac_models[pid]["difficulty"]
            elif result == "AC":
                problem_results[key] = "AC"

        profile.total_attempted = len(problem_results)
        solved_keys = {k for k, v in problem_results.items() if v == "AC"}
        profile.total_solved = len(solved_keys)
        profile.overall_solve_rate = (
            profile.total_solved / profile.total_attempted
            if profile.total_attempted > 0 else 0.0
        )

        profile.solved_problem_ids = {f"ac:{k}" for k in solved_keys}

        # Difficulty level from solved problems
        solved_diffs = []
        for key in solved_keys:
            if key in problem_diffs:
                nd = normalize_ac_difficulty(problem_diffs[key])
                if nd is not None:
                    solved_diffs.append(nd)

        if solved_diffs:
            profile.difficulty_level = float(np.percentile(solved_diffs, 75))

        return profile

    def build_codechef_profile(self, submissions: list, handle: str) -> UserProfile:
        """
        Build a user profile from CodeChef submissions.

        CodeChef differs from the other three sources in one useful way: the
        submission widget reports every verdict, not just accepted ones, so
        attempted-but-unsolved problems are directly observable rather than
        inferred. That makes `overall_solve_rate` a genuine success rate here,
        and it is the same negative signal the evaluation notes CF lacks.

        Difficulty comes from the unified catalogue rather than a side-car
        model file, because CodeChef publishes its per-problem rating in the
        catalogue endpoint itself.
        """
        profile = UserProfile(handle=handle, platform="codechef")

        cc_index = {
            p["platform_id"]: p
            for p in self.problems
            if p["platform"] == "codechef"
        }

        # Per-problem best verdict: accepted wins over any number of failures.
        problem_solved: dict = {}
        for sub in submissions:
            code = sub.get("problem_code", "")
            if not code:
                continue
            problem_solved[code] = problem_solved.get(code, False) or bool(
                sub.get("accepted")
            )

        profile.total_attempted = len(problem_solved)
        solved_codes = {c for c, ok in problem_solved.items() if ok}
        profile.total_solved = len(solved_codes)
        profile.overall_solve_rate = (
            profile.total_solved / profile.total_attempted
            if profile.total_attempted > 0 else 0.0
        )
        profile.solved_problem_ids = {f"cc:{c}" for c in solved_codes}

        # Topic mastery over the catalogue problems we can resolve. Attempts
        # count every problem touched; solves only those accepted — so an
        # unsolved attempt lowers mastery rather than being invisible.
        topic_stats = defaultdict(lambda: {
            "attempted": 0, "solved": 0, "difficulties": []
        })
        solved_diffs = []
        for code, ok in problem_solved.items():
            matched = cc_index.get(code)
            if not matched:
                continue
            diff = matched.get("difficulty_normalized")
            for tag in matched.get("tags_unified", []):
                topic_stats[tag]["attempted"] += 1
                if ok:
                    topic_stats[tag]["solved"] += 1
                    if diff is not None:
                        topic_stats[tag]["difficulties"].append(diff)
            if ok and diff is not None:
                solved_diffs.append(diff)

        profile.topic_mastery = {}
        for topic, stats in topic_stats.items():
            diffs = stats["difficulties"]
            profile.topic_mastery[topic] = TopicMastery(
                topic=topic,
                problems_attempted=stats["attempted"],
                problems_solved=stats["solved"],
                solve_rate=(stats["solved"] / stats["attempted"]
                            if stats["attempted"] > 0 else 0.0),
                avg_difficulty=float(np.mean(diffs)) if diffs else 0.0,
                max_difficulty=float(max(diffs)) if diffs else 0.0,
            )

        if solved_diffs:
            profile.difficulty_level = float(np.percentile(solved_diffs, 75))

        return profile

    def build_leetcode_profile(
        self, submissions: list, user_info: dict, username: str
    ) -> UserProfile:
        """
        Build a user profile from LeetCode data.

        Args:
            submissions: recent AC submissions from recentAcSubmissionList
            user_info: matchedUser dict from userPublicProfile
            username: LC username
        """
        from models.unified_schema import unify_tags, normalize_lc_difficulty

        profile = UserProfile(handle=username, platform="leetcode")

        # Extract solve counts from profile
        ac_stats = user_info.get("submitStatsGlobal", {}).get("acSubmissionNum", [])
        for entry in ac_stats:
            if entry["difficulty"] == "All":
                profile.total_solved = entry["count"]
                profile.total_attempted = entry["submissions"]

        profile.overall_solve_rate = (
            profile.total_solved / profile.total_attempted
            if profile.total_attempted > 0 else 0.0
        )

        # Build solved set from recent submissions by matching titleSlug to our dataset
        lc_problem_index = {}
        for p in self.problems:
            if p["platform"] == "leetcode":
                # cprs_id is like "lc:123" — also index by slug from url
                slug = p.get("url", "").rstrip("/").split("/")[-1]
                if slug:
                    lc_problem_index[slug] = p

        topic_stats = defaultdict(lambda: {
            "attempted": 0, "solved": 0, "difficulties": []
        })

        for sub in submissions:
            slug = sub.get("titleSlug", "")
            matched = lc_problem_index.get(slug)
            if matched:
                profile.solved_problem_ids.add(matched["cprs_id"])
                tags = matched.get("tags_unified", [])
                diff = matched.get("difficulty_normalized")
                for tag in tags:
                    topic_stats[tag]["attempted"] += 1
                    topic_stats[tag]["solved"] += 1
                    if diff is not None:
                        topic_stats[tag]["difficulties"].append(diff)

        # Build topic mastery
        profile.topic_mastery = {}
        for topic, stats in topic_stats.items():
            diffs = stats["difficulties"]
            profile.topic_mastery[topic] = TopicMastery(
                topic=topic,
                problems_attempted=stats["attempted"],
                problems_solved=stats["solved"],
                solve_rate=stats["solved"] / stats["attempted"] if stats["attempted"] > 0 else 0.0,
                avg_difficulty=float(np.mean(diffs)) if diffs else 0.0,
                max_difficulty=float(max(diffs)) if diffs else 0.0,
            )

        # Difficulty level from matched solved problems
        solved_diffs = [
            lc_problem_index[sub.get("titleSlug", "")]["difficulty_normalized"]
            for sub in submissions
            if sub.get("titleSlug", "") in lc_problem_index
            and lc_problem_index[sub["titleSlug"]].get("difficulty_normalized") is not None
        ]
        if solved_diffs:
            profile.difficulty_level = float(np.percentile(solved_diffs, 75))
        else:
            # Estimate from profile difficulty distribution
            diff_map = {"Easy": 0.15, "Medium": 0.4, "Hard": 0.7}
            weighted = []
            for entry in ac_stats:
                if entry["difficulty"] in diff_map and entry["count"] > 0:
                    weighted.extend([diff_map[entry["difficulty"]]] * min(entry["count"], 100))
            if weighted:
                profile.difficulty_level = float(np.percentile(weighted, 75))

        return profile

    def merge_profiles(self, profiles: list) -> UserProfile:
        """
        Merge profiles from multiple platforms into a single unified profile.
        """
        if not profiles:
            return UserProfile(handle="unknown")
        if len(profiles) == 1:
            return profiles[0]

        merged = UserProfile(
            handle=profiles[0].handle,
            platform="multi",
        )

        # Combine solved sets
        for p in profiles:
            merged.solved_problem_ids.update(p.solved_problem_ids)

        # Combine stats
        merged.total_solved = sum(p.total_solved for p in profiles)
        merged.total_attempted = sum(p.total_attempted for p in profiles)
        merged.overall_solve_rate = (
            merged.total_solved / merged.total_attempted
            if merged.total_attempted > 0 else 0.0
        )

        # Use highest rating
        ratings = [p.rating for p in profiles if p.rating]
        if ratings:
            merged.rating = max(ratings)

        # Weighted average difficulty level
        weights = [p.total_solved for p in profiles]
        total_w = sum(weights) or 1
        merged.difficulty_level = sum(
            p.difficulty_level * w for p, w in zip(profiles, weights)
        ) / total_w

        # Merge topic mastery
        topic_combined = defaultdict(lambda: {
            "attempted": 0, "solved": 0, "difficulties": []
        })
        for p in profiles:
            for topic, m in p.topic_mastery.items():
                if isinstance(m, dict):
                    m = TopicMastery(**m)
                topic_combined[topic]["attempted"] += m.problems_attempted
                topic_combined[topic]["solved"] += m.problems_solved
                if m.avg_difficulty > 0:
                    topic_combined[topic]["difficulties"].append(m.avg_difficulty)

        merged.topic_mastery = {}
        for topic, stats in topic_combined.items():
            diffs = stats["difficulties"]
            merged.topic_mastery[topic] = TopicMastery(
                topic=topic,
                problems_attempted=stats["attempted"],
                problems_solved=stats["solved"],
                solve_rate=stats["solved"] / stats["attempted"] if stats["attempted"] > 0 else 0.0,
                avg_difficulty=float(np.mean(diffs)) if diffs else 0.0,
                max_difficulty=float(max(diffs)) if diffs else 0.0,
            )

        return merged

    def analyze_recent_performance(
        self, submissions: list, platform: str = "codeforces", last_n: int = 50
    ) -> RecentPerformance:
        """
        Analyze performance on the last N submissions.
        Detects trends (improving/declining) and recent topic focus.
        """
        from models.unified_schema import unify_tags, normalize_cf_difficulty, normalize_ac_difficulty

        if not submissions:
            return RecentPerformance()

        # Sort by time (most recent first)
        if platform == "codeforces":
            sorted_subs = sorted(submissions, key=lambda s: s.get("creationTimeSeconds", 0), reverse=True)
        elif platform == "leetcode":
            sorted_subs = sorted(submissions, key=lambda s: int(s.get("timestamp", 0)), reverse=True)
        elif platform == "codechef":
            sorted_subs = sorted(submissions, key=_codechef_epoch, reverse=True)
        else:
            sorted_subs = sorted(submissions, key=lambda s: s.get("epoch_second", 0), reverse=True)

        # Deduplicate by problem (keep best verdict per problem)
        seen = set()
        recent = []
        for sub in sorted_subs:
            if platform == "codeforces":
                pid = (sub.get("problem", {}).get("contestId"), sub.get("problem", {}).get("index"))
            elif platform == "leetcode":
                pid = sub.get("titleSlug", "")
            elif platform == "codechef":
                pid = sub.get("problem_code", "")
            else:
                pid = sub.get("problem_id", "")

            if pid not in seen:
                seen.add(pid)
                recent.append(sub)
                if len(recent) >= last_n:
                    break

        # Build LC problem index for tag/difficulty lookup
        lc_problem_index = {}
        if platform == "leetcode":
            for p in self.problems:
                if p["platform"] == "leetcode":
                    slug = p.get("url", "").rstrip("/").split("/")[-1]
                    if slug:
                        lc_problem_index[slug] = p

        cc_problem_index = {}
        if platform == "codechef":
            cc_problem_index = {
                p["platform_id"]: p
                for p in self.problems
                if p["platform"] == "codechef"
            }

        # Analyze
        solved = 0
        difficulties = []
        topics = Counter()

        for sub in recent:
            if platform == "codeforces":
                is_solved = sub.get("verdict") == "OK"
                problem = sub.get("problem", {})
                tags = unify_tags(problem.get("tags", []))
                rating = problem.get("rating")
                if rating:
                    difficulties.append(normalize_cf_difficulty(rating) or 0)
            elif platform == "leetcode":
                is_solved = True  # recentAcSubmissionList only returns AC
                slug = sub.get("titleSlug", "")
                matched = lc_problem_index.get(slug)
                if matched:
                    tags = matched.get("tags_unified", [])
                    diff = matched.get("difficulty_normalized")
                    if diff is not None:
                        difficulties.append(diff)
                else:
                    tags = []
            elif platform == "codechef":
                # CodeChef reports every verdict, so an unsolved attempt is
                # observed rather than assumed — unlike the LeetCode feed,
                # which only ever returns accepted submissions.
                is_solved = bool(sub.get("accepted"))
                matched = cc_problem_index.get(sub.get("problem_code", ""))
                if matched:
                    tags = matched.get("tags_unified", [])
                    diff = matched.get("difficulty_normalized")
                    if diff is not None:
                        difficulties.append(diff)
                else:
                    tags = []
            else:
                is_solved = sub.get("result") == "AC"
                tags = []
                difficulties.append(0)

            if is_solved:
                solved += 1
            for t in tags:
                topics[t] += 1

        # Trend detection: compare first half vs second half difficulty
        trend = "stable"
        if len(difficulties) >= 10:
            mid = len(difficulties) // 2
            recent_avg = np.mean(difficulties[:mid])
            older_avg = np.mean(difficulties[mid:])
            diff = recent_avg - older_avg
            if diff > 0.05:
                trend = "improving"
            elif diff < -0.05:
                trend = "declining"

        return RecentPerformance(
            last_n=len(recent),
            solved=solved,
            attempted=len(recent),
            solve_rate=solved / len(recent) if recent else 0.0,
            avg_difficulty=float(np.mean(difficulties)) if difficulties else 0.0,
            topics_practiced=[t for t, _ in topics.most_common(10)],
            trend=trend,
        )

    def recommend(
        self,
        profile: UserProfile,
        n: int = 20,
        platforms: Optional[list] = None,
        target_topics: Optional[list] = None,
        difficulty_stretch: float = 0.1,
        diversity_weight: float = 0.3,
        cf_affinity: Optional[dict] = None,
    ) -> list:
        """
        Generate recommendations for a user.

        Args:
            profile: user's skill profile
            n: number of recommendations to return
            platforms: filter to specific platforms (None = all)
            target_topics: focus on specific topics (None = auto-detect weak areas)
            difficulty_stretch: how much harder than current level (0.1 = 10% harder)
            diversity_weight: balance between targeting weaknesses vs exploration
            cf_affinity: optional {cprs_id: collaborative-filtering score}. Where a
                problem has one it replaces the topic-gap term, because the offline
                evaluation found CF the stronger affinity signal by an order of
                magnitude (nDCG@10 0.0959 against 0.0012 for content). CF covers
                only the problems that appear in the interaction matrix, so the
                topic-gap term still carries everything else. Both paths feed the
                same weighted sum, so the two populations stay comparable.

                Note this is deliberately not the offline `hybrid` model, which
                blends the two proportionally to interaction density. Bucketed
                analysis (scripts/analyse_hybrid_crossover.py) showed that blend
                never beats CF at any history size, and is worst exactly where it
                was meant to help: for users with under ten solves it halves
                nDCG@10, because a small alpha hands most of the weight to a
                signal that scores at chance. Falling back to content only when
                CF has nothing to say is the behaviour the evidence supports.
        """
        # Determine target difficulty range
        target_center = profile.difficulty_level + difficulty_stretch
        target_center = min(target_center, 1.0)
        difficulty_sigma = 0.15  # gaussian spread

        # Determine weak topics (if not specified)
        if target_topics is None:
            target_topics = self._find_weak_topics(profile)

        # Build topic weights (higher = more important to practice)
        topic_weights = self._compute_topic_weights(profile, target_topics)

        # CF scores are unbounded dot products while the topic-gap term is
        # [0, 1]; min-max them onto the same scale so one weight means the same
        # thing on both paths.
        cf_lo = cf_span = None
        if cf_affinity:
            vals = np.fromiter(cf_affinity.values(), dtype=float)
            cf_lo = float(vals.min())
            cf_span = float(vals.max()) - cf_lo or 1.0

        # Score all problems
        scored = []
        for problem in self.problems:
            cprs_id = problem["cprs_id"]

            # Skip already solved
            if cprs_id in profile.solved_problem_ids:
                continue

            # Platform filter
            if platforms and problem["platform"] not in platforms:
                continue

            # Skip problems without difficulty
            diff = problem.get("difficulty_normalized")
            if diff is None:
                continue

            # Skip premium problems
            if problem.get("is_premium", False):
                continue

            tags = problem.get("tags_unified", [])
            if not tags:
                continue

            # --- Scoring ---
            # 1. Difficulty fit (gaussian around target)
            diff_score = math.exp(-0.5 * ((diff - target_center) / difficulty_sigma) ** 2)

            # 2. Affinity: CF where we have it, topic gap everywhere else.
            matching_topics = []
            cf_raw = cf_affinity.get(cprs_id) if cf_affinity else None
            if cf_raw is not None:
                topic_score = (cf_raw - cf_lo) / cf_span
                from_cf = True
            else:
                from_cf = False
                topic_score = 0.0
                for tag in tags:
                    w = topic_weights.get(tag, 0.0)
                    if w > 0:
                        topic_score += w
                        matching_topics.append(tag)
                if len(tags) > 0:
                    topic_score /= len(tags)  # normalize by number of tags

            # 3. Popularity bonus (slight preference for well-tested problems)
            solve_count = problem.get("solve_count") or 0
            pop_score = min(1.0, math.log1p(solve_count) / 12.0)  # cap at ~160K solves

            # Combined score
            score = (
                0.4 * topic_score +
                0.4 * diff_score +
                0.1 * pop_score +
                0.1 * np.random.random() * diversity_weight  # slight randomness for diversity
            )

            # Build reasons
            reasons = []
            if from_cf:
                reasons.append("Solved next by people with a similar solve history")
            if matching_topics:
                reasons.append(f"Targets weak topics: {', '.join(matching_topics[:3])}")
            if abs(diff - target_center) < difficulty_sigma:
                reasons.append(f"Good difficulty match ({diff:.2f} vs target {target_center:.2f})")
            elif diff > target_center:
                reasons.append(f"Stretching challenge ({diff:.2f} vs level {profile.difficulty_level:.2f})")

            scored.append(Recommendation(
                cprs_id=cprs_id,
                platform=problem["platform"],
                name=problem["name"],
                url=problem["url"],
                difficulty_normalized=diff,
                tags=tags,
                score=score,
                reasons=reasons,
            ))

        # Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate: don't recommend too many from same contest
        seen_contests = Counter()
        filtered = []
        for rec in scored:
            # Extract contest from cprs_id
            parts = rec.cprs_id.split(":")
            contest = parts[1][:4] if len(parts) > 1 else ""
            if seen_contests[contest] >= 2:
                continue
            seen_contests[contest] += 1
            filtered.append(rec)
            if len(filtered) >= n:
                break

        return filtered

    # A topic the user has never attempted is unknown, not weak. Ranking it
    # above a topic they demonstrably struggle with confuses absence of
    # evidence with evidence of absence — and in a cross-platform system it
    # does real damage, because each platform has its own tag vocabulary. A
    # Codeforces specialist has never touched `design` or `hash_table`, so
    # treating unseen tags as maximal weakness declares every LeetCode-only
    # tag their weakest area and floods the results with one platform.
    #
    # Unexplored topics still deserve a place — surfacing them is the whole
    # point of going cross-platform — but as exploration ranked below proven
    # weakness, not above it.
    UNEXPLORED_PRIOR = 0.45
    MIN_ATTEMPTS_FOR_CONFIDENCE = 8

    def _topic_scores(self, profile: UserProfile) -> dict:
        """
        Score every candidate topic by how much practice it deserves.

        Demonstrated weakness is scaled by how much evidence supports it, so a
        topic failed twice does not outrank one failed twenty times.
        """
        mastery = profile.topic_mastery
        scores = {}

        for topic, m in mastery.items():
            # `other:` tags are unmapped platform-specific labels rather than
            # taxonomy entries, so they are not meaningful practice targets.
            if topic.startswith("other:"):
                continue
            if isinstance(m, dict):
                m = TopicMastery(**m)
            attempted = m.problems_attempted
            if attempted <= 0:
                continue
            weakness = 1.0 - m.solve_rate
            confidence = min(1.0, attempted / self.MIN_ATTEMPTS_FOR_CONFIDENCE)
            # Half the score is the weakness itself, half is how sure we are
            # of it; an unconvincing sample cannot reach the top on its own.
            scores[topic] = weakness * (0.5 + 0.5 * confidence)

        # Unexplored topics, weighted by how well represented the topic is in
        # the corpus — a tag on four problems is not a meaningful gap.
        seen = {t for t, m in mastery.items()
                if (m.get("problems_attempted", 0) if isinstance(m, dict)
                    else m.problems_attempted) > 0}
        if self._tag_prevalence:
            busiest = max(self._tag_prevalence.values())
            for tag in self.all_tags:
                if tag in seen or tag.startswith("other:"):
                    continue
                share = self._tag_prevalence.get(tag, 0) / busiest
                scores[tag] = self.UNEXPLORED_PRIOR * min(1.0, share * 4)

        return scores

    def _find_weak_topics(self, profile: UserProfile, top_n: int = 5) -> list:
        """The topics most worth practising, strongest evidence first."""
        scores = self._topic_scores(profile)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [t for t, _ in ranked[:top_n]]

    def _compute_topic_weights(self, profile: UserProfile, target_topics: list) -> dict:
        """Importance weight per target topic, on the same evidence scale."""
        scores = self._topic_scores(profile)
        return {t: max(0.1, scores.get(t, self.UNEXPLORED_PRIOR)) for t in target_topics}
