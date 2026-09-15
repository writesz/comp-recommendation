"""
Fetch problem and user data from the Codeforces API.

Endpoints used:
- problemset.problems — all problems with tags, ratings, solve counts
- user.status — a user's submission history
- user.info — user profile (rating, rank)
- user.rating — rating change history
"""
import time
from typing import Any, Optional

import requests
from loguru import logger


BASE_URL = "https://codeforces.com/api"
REQUEST_DELAY = 0.5  # seconds between requests (CF rate limit: 1 req/sec)


def _get(method: str, params: Optional[dict] = None) -> dict:
    """Make a GET request to the Codeforces API with rate limiting."""
    url = f"{BASE_URL}/{method}"
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise RuntimeError(f"CF API error: {data.get('comment', 'unknown')}")
    return data["result"]


def fetch_all_problems() -> list:
    """
    Fetch all Codeforces problems with metadata.
    Returns list of dicts with keys:
        contestId, index, name, type, rating, tags, solvedCount
    """
    logger.info("Fetching all Codeforces problems...")
    result = _get("problemset.problems")
    problems = result["problems"]
    stats = {(s["contestId"], s["index"]): s["solvedCount"] for s in result["problemStatistics"]}

    for p in problems:
        key = (p.get("contestId"), p.get("index"))
        p["solvedCount"] = stats.get(key, 0)

    logger.info(f"Fetched {len(problems)} problems")
    return problems


def fetch_user_submissions(handle: str, count: Optional[int] = None) -> list:
    """Fetch submission history for a user."""
    logger.info(f"Fetching submissions for user '{handle}'...")
    params = {"handle": handle}
    if count:
        params["count"] = count
    submissions = _get("user.status", params)
    logger.info(f"Fetched {len(submissions)} submissions for '{handle}'")
    return submissions


def fetch_user_info(handle: str) -> dict:
    """Fetch user profile info (rating, rank, etc.)."""
    result = _get("user.info", {"handles": handle})
    return result[0]


def fetch_user_rating_history(handle: str) -> list:
    """Fetch rating change history for a user."""
    return _get("user.rating", {"handle": handle})
