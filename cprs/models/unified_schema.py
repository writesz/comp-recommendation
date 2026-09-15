"""
Unified problem schema for cross-platform competitive programming problems.

This is the core contribution of the CPRS dataset: normalizing problems from
Codeforces, AtCoder, and LeetCode into a single comparable representation.

Normalization challenges addressed:
1. Difficulty scales differ (CF: 800-3500, AtCoder: -1000 to 4000+, LC: Easy/Medium/Hard)
2. Tag taxonomies are platform-specific
3. Problem identifiers are incompatible
4. Solve rate semantics vary
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"
    LEETCODE = "leetcode"


class UnifiedProblem(BaseModel):
    """A problem normalized to the unified CPRS schema."""

    # Identity
    platform: Platform
    platform_id: str = Field(description="Original ID on the platform")
    cprs_id: str = Field(description="Unified ID: {platform}:{platform_id}")
    url: str

    # Metadata
    name: str
    contest_id: Optional[str] = None

    # Difficulty — normalized to 0.0-1.0 scale
    difficulty_raw: Optional[float] = Field(
        default=None, description="Original difficulty value from platform"
    )
    difficulty_normalized: Optional[float] = Field(
        default=None, description="Normalized difficulty [0.0, 1.0]"
    )

    # Tags — mapped to unified taxonomy
    tags_original: List[str] = Field(default_factory=list)
    tags_unified: List[str] = Field(default_factory=list)

    # Solve statistics
    solve_count: Optional[int] = None
    acceptance_rate: Optional[float] = None

    # Flags
    is_premium: bool = False


# --- Difficulty normalization ---

# CF ratings range from 800 to 3500
CF_RATING_MIN = 800
CF_RATING_MAX = 3500

# AtCoder difficulty ranges from roughly -1000 to 4500
AC_DIFF_MIN = -500  # clip low outliers
AC_DIFF_MAX = 4000

# LeetCode has 3 levels
LC_DIFFICULTY_MAP = {"Easy": 0.2, "Medium": 0.5, "Hard": 0.85}


def normalize_cf_difficulty(rating: Optional[int] = None) -> Optional[float]:
    """Normalize Codeforces rating (800-3500) to [0, 1]."""
    if rating is None:
        return None
    clamped = max(CF_RATING_MIN, min(CF_RATING_MAX, rating))
    return (clamped - CF_RATING_MIN) / (CF_RATING_MAX - CF_RATING_MIN)


def normalize_ac_difficulty(difficulty: Optional[float] = None) -> Optional[float]:
    """Normalize AtCoder difficulty estimate to [0, 1]."""
    if difficulty is None:
        return None
    clamped = max(AC_DIFF_MIN, min(AC_DIFF_MAX, difficulty))
    return (clamped - AC_DIFF_MIN) / (AC_DIFF_MAX - AC_DIFF_MIN)


def normalize_lc_difficulty(level: str) -> Optional[float]:
    """Map LeetCode difficulty label to [0, 1]."""
    return LC_DIFFICULTY_MAP.get(level)


# --- Tag unification ---
# Maps platform-specific tags to a unified taxonomy.
# This is a key contribution: enabling cross-platform topic comparison.

UNIFIED_TAG_MAP = {
    # Data structures
    "implementation": "implementation",
    "brute force": "brute_force",
    "constructive algorithms": "constructive",
    "math": "math",
    "number theory": "number_theory",
    "combinatorics": "combinatorics",
    "geometry": "geometry",

    # Graph theory
    "graphs": "graphs",
    "graph": "graphs",
    "shortest paths": "shortest_paths",
    "trees": "trees",
    "tree": "trees",
    "dfs and similar": "dfs",
    "depth-first search": "dfs",
    "breadth-first search": "bfs",

    # Dynamic programming
    "dp": "dynamic_programming",
    "dynamic programming": "dynamic_programming",

    # Greedy
    "greedy": "greedy",
    "sortings": "sorting",
    "sorting": "sorting",

    # Strings
    "strings": "strings",
    "string": "strings",
    "string matching": "string_matching",
    "hashing": "hashing",

    # Binary search
    "binary search": "binary_search",
    "two pointers": "two_pointers",

    # Data structures
    "data structures": "data_structures",
    "stack": "stack",
    "queue": "queue",
    "heap (priority queue)": "heap",
    "segment tree": "segment_tree",
    "union find": "union_find",
    "disjoint set union": "union_find",
    "hash table": "hash_table",
    "linked list": "linked_list",
    "array": "array",
    "matrix": "matrix",

    # Advanced
    "flows": "network_flow",
    "fft": "fft",
    "games": "game_theory",
    "probabilities": "probability",
    "bitmasks": "bitmask",
    "bit manipulation": "bitmask",
    "divide and conquer": "divide_and_conquer",
    "backtracking": "backtracking",
    "sliding window": "sliding_window",
    "trie": "trie",
    "recursion": "recursion",
    "simulation": "simulation",
    "design": "design",
    "prefix sum": "prefix_sum",
    "monotonic stack": "monotonic_stack",
}


def unify_tags(tags: list[str]) -> list[str]:
    """Map platform-specific tags to the unified taxonomy."""
    unified = set()
    for tag in tags:
        key = tag.lower().strip()
        if key in UNIFIED_TAG_MAP:
            unified.add(UNIFIED_TAG_MAP[key])
        else:
            # Keep unmapped tags with a prefix for transparency
            unified.add(f"other:{key}")
    return sorted(unified)
