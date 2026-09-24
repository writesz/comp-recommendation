"""
Fetch problem and contest data from AtCoder via the kenkoooo API.

Endpoints used:
- /resources/problems.json — all problems
- /resources/problem-models.json — difficulty estimates
- /resources/contests.json — contest metadata
- /atcoder-api/v3/user/submissions — user submissions (paginated)
"""
import time
from typing import Any, Optional

import requests
from loguru import logger


BASE_URL = "https://kenkoooo.com/atcoder"
REQUEST_DELAY = 1.0  # kenkoooo asks for polite usage


def _get(path: str, params: Optional[dict] = None) -> Any:
    """Make a GET request to the kenkoooo AtCoder API."""
    url = f"{BASE_URL}/{path}"
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_problems() -> list:
    """Fetch all AtCoder problems."""
    logger.info("Fetching all AtCoder problems...")
    problems = _get("resources/problems.json")
    logger.info(f"Fetched {len(problems)} problems")
    return problems


def fetch_difficulty_models() -> dict:
    """
    Fetch difficulty estimates for problems.
    Returns dict mapping problem_id -> {difficulty, discrimination, ...}
    """
    logger.info("Fetching AtCoder difficulty models...")
    models = _get("resources/problem-models.json")
    logger.info(f"Fetched difficulty models for {len(models)} problems")
    return models


def fetch_contests() -> list:
    """Fetch all AtCoder contests."""
    logger.info("Fetching AtCoder contests...")
    contests = _get("resources/contests.json")
    logger.info(f"Fetched {len(contests)} contests")
    return contests


def fetch_user_submissions(user_id: str, from_second: int = 0) -> list:
    """
    Fetch submissions for a user, paginated by unix timestamp.
    The API returns up to 500 submissions at a time.
    """
    logger.info(f"Fetching AtCoder submissions for '{user_id}'...")
    all_subs = []
    cursor = from_second

    while True:
        subs = _get("atcoder-api/v3/user/submissions", {
            "user": user_id,
            "from_second": cursor,
        })
        if not subs:
            break
        all_subs.extend(subs)
        cursor = subs[-1]["epoch_second"] + 1
        logger.debug(f"  fetched {len(all_subs)} submissions so far...")

    logger.info(f"Fetched {len(all_subs)} total submissions for '{user_id}'")
    return all_subs


def fetch_user_contest_history(user_id: str) -> list:
    """
    Fetch a user's rated contest history.

    Served by AtCoder itself rather than the kenkoooo mirror, which has no
    rating-history endpoint. Each entry carries the contest, the placement,
    the rating before/after and the performance for that round.
    """
    url = f"https://atcoder.jp/users/{user_id}/history/json"
    logger.info(f"Fetching AtCoder contest history for '{user_id}'...")
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    history = resp.json()
    logger.info(f"Fetched {len(history)} contest entries for '{user_id}'")
    return history
