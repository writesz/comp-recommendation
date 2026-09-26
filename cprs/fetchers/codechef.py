"""
Fetch problem and user data from CodeChef.

CodeChef's official developer API (api.codechef.com) was closed to new
registrations and now answers `401 unauthorized for this resource scope`, so
we use the same public JSON endpoints that serve codechef.com itself:

- /api/list/problems                       — paginated problem catalogue
- /api/contests/PRACTICE/problems/<CODE>   — per-problem detail, tags, statement
- /users/<handle>                          — profile HTML (rating + contest history)
- /recent/user?user_handle=<h>&page=<n>    — paginated submission history (HTML rows)

Two things distinguish this fetcher from the other three:

1. CodeChef rate-limits far more aggressively than Codeforces or kenkoooo —
   a naive half-second loop is throttled to HTTP 429 within about a dozen
   requests. Every call therefore goes through _get(), which backs off
   exponentially and retries.
2. Problem *statements* are served as structured JSON by the detail endpoint
   rather than hidden behind the anti-scraping protection that limits AtCoder
   statement collection, so the auto-tagger can reach the whole catalogue.

Submission history and the profile are HTML rather than JSON; the parsers
below pull out only the fields CPRS needs and are deliberately tolerant of
markup drift, returning what they can rather than raising.
"""
import json
import re
import time
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

BASE_URL = "https://www.codechef.com"
REQUEST_DELAY = 2.0  # CodeChef throttles hard; see module docstring
MAX_RETRIES = 5
PAGE_SIZE = 100  # max accepted by /api/list/problems

# The catalogue endpoint is far more tolerant than the profile and detail
# pages, which is what actually throttles: paginating the problem list at
# half a second a page is accepted, while the same rate on /users/<handle>
# is throttled within a dozen requests.
CATALOGUE_DELAY = 0.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9",
}

# difficulty_rating uses these as "no estimate" / "unrated" markers
DIFFICULTY_SENTINELS = (-1, 0, 9999)


class RateLimited(RuntimeError):
    """Raised when CodeChef keeps returning 429 after every retry."""


def _get(path: str, params: Optional[dict] = None, as_json: bool = True,
         delay: float = REQUEST_DELAY) -> Any:
    """GET with exponential backoff on 429. Returns parsed JSON or raw text."""
    url = path if path.startswith("http") else f"{BASE_URL}/{path.lstrip('/')}"
    backoff = delay
    for attempt in range(MAX_RETRIES):
        time.sleep(backoff)
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            backoff *= 2
            logger.debug(f"429 from {url}; backing off to {backoff:.1f}s "
                         f"(attempt {attempt + 1}/{MAX_RETRIES})")
            continue
        resp.raise_for_status()
        if not as_json:
            return resp.text
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(f"CodeChef returned non-JSON for {url}: {e}") from e
    raise RateLimited(f"rate limited after {MAX_RETRIES} attempts: {url}")


# --- Problem catalogue ------------------------------------------------------

def fetch_all_problems(limit: Optional[int] = None,
                       delay: float = CATALOGUE_DELAY) -> list:
    """
    Fetch the CodeChef problem catalogue.

    Returns list of dicts with keys:
        code, name, difficulty_rating, total_submissions,
        successful_submissions, distinct_successful_submissions, contest_code
    """
    logger.info("Fetching all CodeChef problems...")
    all_problems: list = []
    page = 0
    total: Optional[int] = None

    while True:
        result = _get("api/list/problems", {
            "sort_by": "difficulty_rating",
            "sort_order": "asc",
            "search": "",
            "limit": PAGE_SIZE,
            "page": page,
        }, delay=delay)
        rows = result.get("data") or []
        if total is None:
            total = int(result.get("count") or 0)
            logger.info(f"CodeChef reports {total} problems")
        if not rows:
            break

        all_problems.extend(rows)
        logger.debug(f"  fetched {len(all_problems)}/{total} problems...")
        page += 1

        if limit and len(all_problems) >= limit:
            all_problems = all_problems[:limit]
            break
        if total and len(all_problems) >= total:
            break

    logger.info(f"Fetched {len(all_problems)} CodeChef problems")
    return all_problems


def fetch_problem_detail(code: str, delay: float = REQUEST_DELAY) -> dict:
    """
    Fetch one problem's detail: tags, difficulty and full statement.

    `computed_tags` is CodeChef's own coarse taxonomy; `user_tags` is
    crowd-sourced and mixes genuine topics with setter usernames, so callers
    should map it through the unified taxonomy and discard what does not land.
    """
    return _get(f"api/contests/PRACTICE/problems/{code}", delay=delay)


def extract_statement(detail: dict) -> str:
    """
    Pull clean statement text out of a problem-detail payload.

    Prefers the structured `problemComponents.statement`, falling back to the
    full `body`. HTML and LaTeX-ish markup are stripped down to plain text so
    the result can go straight into the tagger's TF-IDF vectoriser.
    """
    components = detail.get("problemComponents") or {}
    raw = components.get("statement") or detail.get("body") or ""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "lxml").get_text(separator=" ", strip=True)
    text = re.sub(r"\$+[^$]*\$+", " ", text)  # drop inline maths
    return re.sub(r"\s+", " ", text).strip()


# --- User data --------------------------------------------------------------

_RATING_RE = re.compile(r"rating-number[^>]*>\s*([0-9]+)\s*<")
_ALL_RATING_RE = re.compile(r"var all_rating\s*=\s*(\[.*?\]);", re.S)


def fetch_user_info(handle: str) -> dict:
    """
    Fetch a CodeChef user's public profile.

    Returns {handle, rating, highest_rating, global_rank, n_contests}.
    Missing fields come back as None rather than raising, because CodeChef
    renders a partial profile for users who have never competed.
    """
    logger.info(f"Fetching CodeChef profile for '{handle}'...")
    html = _get(f"users/{handle}", as_json=False)

    ratings = [int(m) for m in _RATING_RE.findall(html)]
    history = _parse_rating_history(html)
    highest = max((h["rating"] for h in history), default=None)

    rank_m = re.search(r"rating-ranks.*?<strong>\s*([0-9]+)\s*<", html, re.S)
    return {
        "handle": handle,
        "rating": ratings[0] if ratings else None,
        "highest_rating": highest,
        "global_rank": int(rank_m.group(1)) if rank_m else None,
        "n_contests": len(history),
    }


def _parse_rating_history(html: str) -> list:
    """Extract the `all_rating` JS array embedded in the profile page."""
    m = _ALL_RATING_RE.search(html)
    if not m:
        return []
    try:
        raw = json.loads(m.group(1))
    except ValueError:
        logger.warning("CodeChef all_rating array present but unparseable")
        return []

    out = []
    for e in raw:
        try:
            out.append({
                "code": e.get("code"),
                "name": e.get("name"),
                "rating": int(e["rating"]),
                "rank": int(e["rank"]) if e.get("rank") else None,
                "end_date": e.get("end_date"),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch_user_contest_history(handle: str) -> list:
    """Fetch a user's rated contest history (one entry per contest attended)."""
    logger.info(f"Fetching CodeChef contest history for '{handle}'...")
    html = _get(f"users/{handle}", as_json=False)
    history = _parse_rating_history(html)
    logger.info(f"Fetched {len(history)} contest entries for '{handle}'")
    return history


_VERDICT_ACCEPTED = "accepted"


_CODE_RE = re.compile(r"^[A-Za-z0-9_]{2,}$")


def _parse_submission_rows(content_html: str) -> list:
    """
    Parse one page of the recent-submissions widget.

    Row layout is [timestamp, problem, verdict, language, view]. The problem
    code lives in the second cell's `title` attribute; it is wrapped in a
    /problems/ link only for problems that have reached the practice section,
    so reading the link alone silently drops the majority of rows. The verdict
    is the `title` of a span inside the third cell ("accepted", "wrong
    answer", "compilation error", ...) — the cell's own title carries the
    partial score, e.g. "(100)", not the verdict.
    """
    soup = BeautifulSoup(content_html, "lxml")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue  # header and footer rows

        code = (cells[1].get("title") or cells[1].get_text(strip=True) or "").strip()
        if not code:
            link = cells[1].find("a", href=re.compile(r"^/problems/"))
            code = link["href"].rsplit("/", 1)[-1] if link else ""
        if not _CODE_RE.match(code):
            continue

        verdict_span = cells[2].find("span", title=True)
        verdict = (verdict_span["title"].strip().lower() if verdict_span else "")

        out.append({
            "problem_code": code,
            "verdict": verdict,
            "accepted": verdict.startswith(_VERDICT_ACCEPTED),
            "time": cells[0].get("title"),
            "language": cells[3].get("title"),
        })
    return out


def fetch_user_submissions(handle: str, max_pages: Optional[int] = None) -> list:
    """
    Fetch a user's submission history by walking the paginated widget.

    Unlike the AtCoder fetcher — where one request returns 500 submissions —
    CodeChef serves roughly a dozen rows of HTML per request, so a full
    history costs one request per page. `max_pages` bounds that walk for
    users with very long histories.

    Returns list of dicts: {problem_code, verdict, accepted, time, language}.
    """
    logger.info(f"Fetching CodeChef submissions for '{handle}'...")
    first = _get("recent/user", {"user_handle": handle, "page": 0})
    max_page = int(first.get("max_page") or 0)
    pages = max_page + 1
    if max_pages is not None:
        pages = min(pages, max_pages)

    subs = _parse_submission_rows(first.get("content") or "")
    for page in range(1, pages):
        payload = _get("recent/user", {"user_handle": handle, "page": page})
        rows = _parse_submission_rows(payload.get("content") or "")
        if not rows:
            break
        subs.extend(rows)
        if page % 20 == 0:
            logger.debug(f"  {handle}: {len(subs)} submissions over {page} pages...")

    logger.info(f"Fetched {len(subs)} submissions for '{handle}' "
                f"over {pages} page(s) (max_page={max_page})")
    return subs


def solved_problem_codes(submissions: list) -> set:
    """Distinct problem codes the user has an accepted submission for."""
    return {s["problem_code"] for s in submissions if s.get("accepted")}
