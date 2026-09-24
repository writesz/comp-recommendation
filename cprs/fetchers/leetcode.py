"""
Fetch problem data from LeetCode's GraphQL API.

LeetCode does not have a public REST API, so we use the GraphQL endpoint
that powers their website. User submission data requires authentication
and is not publicly accessible, so we focus on problem metadata.
"""
import time
from typing import Any, Dict, List

import requests
from loguru import logger


URL = "https://leetcode.com/graphql"
REQUEST_DELAY = 1.0
BATCH_SIZE = 100  # max per request


PROBLEMSET_QUERY = """
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      frontendQuestionId: questionFrontendId
      title
      titleSlug
      difficulty
      acRate
      topicTags {
        name
        slug
      }
      isPaidOnly
    }
  }
}
"""


def _graphql(query: str, variables: dict) -> dict:
    """Make a GraphQL request to LeetCode."""
    time.sleep(REQUEST_DELAY)
    resp = requests.post(
        URL,
        json={"query": query, "variables": variables},
        headers={
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com/problemset/",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"LeetCode GraphQL error: {data['errors']}")
    return data["data"]


USER_PROFILE_QUERY = """
query userPublicProfile($username: String!) {
  matchedUser(username: $username) {
    username
    submitStatsGlobal {
      acSubmissionNum {
        difficulty
        count
        submissions
      }
    }
  }
}
"""

RECENT_AC_QUERY = """
query recentAcSubmissionList($username: String!, $limit: Int!) {
  recentAcSubmissionList(username: $username, limit: $limit) {
    id
    title
    titleSlug
    timestamp
  }
}
"""


def fetch_user_profile(username: str) -> dict:
    """
    Fetch a LeetCode user's public profile (solve counts by difficulty).
    Returns dict with username and acSubmissionNum breakdown.
    """
    logger.info(f"Fetching LeetCode profile for {username}...")
    data = _graphql(USER_PROFILE_QUERY, {"username": username})
    user = data.get("matchedUser")
    if not user:
        raise RuntimeError(f"LeetCode user '{username}' not found")
    return user


def fetch_user_submissions(username: str, limit: int = 200) -> list:
    """
    Fetch a user's recent accepted submissions from LeetCode.
    Returns list of dicts with: id, title, titleSlug, timestamp.
    """
    logger.info(f"Fetching recent LC submissions for {username} (limit={limit})...")
    data = _graphql(RECENT_AC_QUERY, {"username": username, "limit": limit})
    subs = data.get("recentAcSubmissionList")
    if subs is None:
        raise RuntimeError(f"Could not fetch submissions for '{username}'")
    logger.info(f"Fetched {len(subs)} recent LC submissions for {username}")
    return subs


def fetch_all_problems() -> list:
    """
    Fetch all LeetCode problems (paginated).
    Returns list of dicts with: frontendQuestionId, title, titleSlug,
    difficulty, acRate, topicTags, isPaidOnly
    """
    logger.info("Fetching all LeetCode problems...")
    all_problems = []
    skip = 0

    while True:
        result = _graphql(PROBLEMSET_QUERY, {
            "categorySlug": "",
            "limit": BATCH_SIZE,
            "skip": skip,
            "filters": {},
        })
        questions = result["problemsetQuestionList"]["questions"]
        total = result["problemsetQuestionList"]["total"]

        if not questions:
            break

        all_problems.extend(questions)
        skip += len(questions)
        logger.debug(f"  fetched {len(all_problems)}/{total} problems...")

        if len(all_problems) >= total:
            break

    logger.info(f"Fetched {len(all_problems)} LeetCode problems")
    return all_problems


CONTEST_QUERY = """
query userContestRankingInfo($username: String!) {
  userContestRanking(username: $username) {
    attendedContestsCount
    rating
    globalRanking
    totalParticipants
    topPercentage
  }
  userContestRankingHistory(username: $username) {
    attended
    rating
    ranking
    trendDirection
    problemsSolved
    totalProblems
    finishTimeInSeconds
    contest { title startTime }
  }
}
"""


def fetch_user_contest_history(username: str) -> dict:
    """
    Fetch a user's contest rating summary and per-contest history.

    Returns {"summary": {...}, "history": [...]} where history holds only the
    contests the user actually attended.
    """
    logger.info(f"Fetching LeetCode contest history for {username}...")
    data = _graphql(CONTEST_QUERY, {"username": username})
    history = [h for h in (data.get("userContestRankingHistory") or []) if h.get("attended")]
    logger.info(f"Fetched {len(history)} attended contests for {username}")
    return {"summary": data.get("userContestRanking") or {}, "history": history}
