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
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


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
        for p in self.problems:
            all_tags.update(p.get("tags_unified", []))
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

            # 2. Topic gap score (how much do these tags target weak areas?)
            topic_score = 0.0
            matching_topics = []
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

    def _find_weak_topics(self, profile: UserProfile, top_n: int = 5) -> list:
        """Find user's weakest topics based on solve rate and coverage."""
        # All topics the user has encountered
        mastery = profile.topic_mastery

        # Score each topic: lower = weaker
        topic_scores = {}
        for topic, m in mastery.items():
            if isinstance(m, dict):
                m = TopicMastery(**m)
            # Penalize low solve rate and low attempt count
            weakness = 1.0 - m.solve_rate
            # Boost importance if they've barely tried it
            if m.problems_attempted < 5:
                weakness += 0.3
            topic_scores[topic] = weakness

        # Also add topics they haven't tried at all
        for tag in self.all_tags:
            if tag not in mastery and not tag.startswith("other:"):
                topic_scores[tag] = 1.5  # highest priority — never attempted

        # Return weakest topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])
        return [t for t, _ in sorted_topics[:top_n]]

    def _compute_topic_weights(self, profile: UserProfile, target_topics: list) -> dict:
        """Compute importance weights for each topic."""
        weights = {}
        mastery = profile.topic_mastery

        for topic in target_topics:
            m = mastery.get(topic)
            if m is None:
                weights[topic] = 1.0  # never tried, high priority
            else:
                if isinstance(m, dict):
                    m = TopicMastery(**m)
                # Weight inversely proportional to mastery
                weights[topic] = max(0.1, 1.0 - m.solve_rate)

        return weights
