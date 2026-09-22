"""
Fetch Codeforces user×problem interaction data for collaborative filtering.

Day 1 of the final build plan. Collaborative filtering needs a user×problem
matrix, and offline ranking metrics need real held-out solves — this script
produces the raw interaction events those depend on.

Pipeline:
1. Pull the list of rated CF users (user.ratedList), sample N handles with a
   fixed seed (reproducible).
2. For each handle, fetch submission history (user.status).
3. Keep only ACCEPTED submissions for problems present in our unified dataset,
   recording the *earliest* accepted time per (user, problem).
4. Write one JSON object per solved (user, problem) event to a JSONL file,
   incrementally and resumably (re-running skips already-fetched handles).

Usage:
    python -m scripts.fetch_interactions --num-users 3000 --seed 42
    python -m scripts.fetch_interactions --num-users 20   # quick validation run
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetchers.codeforces import fetch_user_submissions, _get

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
UNIFIED_CSV = DATA_DIR / "cprs_unified_tagged.csv"


def load_cf_problem_ids() -> set[str]:
    """Load the set of valid Codeforces problem ids (cprs_id) from the unified dataset."""
    ids: set[str] = set()
    with open(UNIFIED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("platform") == "codeforces":
                ids.add(row["cprs_id"])
    logger.info(f"Loaded {len(ids)} Codeforces problem ids from unified dataset")
    return ids


def sample_handles(num_users: int, seed: int) -> list[str]:
    """Fetch the rated-user list and sample `num_users` handles reproducibly."""
    logger.info("Fetching rated user list (user.ratedList, activeOnly)...")
    users = _get("user.ratedList", {"activeOnly": "true", "includeRetired": "false"})
    handles = [u["handle"] for u in users if "handle" in u]
    logger.info(f"Rated-user pool: {len(handles)} handles")
    rng = random.Random(seed)
    rng.shuffle(handles)
    sampled = handles[:num_users]
    logger.info(f"Sampled {len(sampled)} handles (seed={seed})")
    return sampled


def load_done_handles(path: Path) -> set[str]:
    """Return handles already recorded in a progress file (for resumability)."""
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    return done


def extract_solved(submissions: list, valid_ids: set[str]) -> dict[str, int]:
    """
    From a user's submissions, return {problem_id: earliest_accepted_epoch}
    for accepted submissions whose problem is in the unified dataset.
    """
    solved: dict[str, int] = {}
    for sub in submissions:
        if sub.get("verdict") != "OK":
            continue
        prob = sub.get("problem", {})
        contest_id = prob.get("contestId")
        index = prob.get("index")
        if contest_id is None or index is None:
            continue
        pid = f"cf:{contest_id}{index}"
        if pid not in valid_ids:
            continue
        t = sub.get("creationTimeSeconds", 0)
        if pid not in solved or t < solved[pid]:
            solved[pid] = t
    return solved


def main(num_users: int, seed: int, request_delay: float, max_retries: int) -> None:
    INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    events_path = INTERACTIONS_DIR / "cf_submissions.jsonl"
    progress_path = INTERACTIONS_DIR / "cf_fetched_handles.txt"

    valid_ids = load_cf_problem_ids()
    handles = sample_handles(num_users, seed)
    done = load_done_handles(progress_path)
    todo = [h for h in handles if h not in done]
    logger.info(f"{len(done)} handles already done, {len(todo)} to fetch")

    n_events = 0
    n_failed = 0
    with open(events_path, "a") as ev_f, open(progress_path, "a") as prog_f:
        for handle in tqdm(todo, desc="users"):
            subs: Optional[list] = None
            for attempt in range(max_retries):
                try:
                    subs = fetch_user_submissions(handle)
                    break
                except Exception as e:  # private profile, rate limit, transient errors
                    wait = request_delay * (2 ** attempt)
                    logger.debug(f"{handle}: attempt {attempt + 1} failed ({e}); retry in {wait:.1f}s")
                    time.sleep(wait)
            if subs is None:
                n_failed += 1
                prog_f.write(handle + "\n")
                prog_f.flush()
                continue

            solved = extract_solved(subs, valid_ids)
            for pid, t in solved.items():
                ev_f.write(json.dumps({"handle": handle, "problem_id": pid, "solve_time": t}) + "\n")
            n_events += len(solved)
            ev_f.flush()
            prog_f.write(handle + "\n")
            prog_f.flush()

    logger.success(
        f"Done. Wrote {n_events} solved events this run "
        f"({n_failed} handles failed/skipped). Events file: {events_path}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch CF user×problem interaction data")
    ap.add_argument("--num-users", type=int, default=3000, help="number of users to sample")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (reproducibility)")
    ap.add_argument("--request-delay", type=float, default=0.5, help="base backoff delay (s)")
    ap.add_argument("--max-retries", type=int, default=3, help="retries per user on failure")
    args = ap.parse_args()
    main(args.num_users, args.seed, args.request_delay, args.max_retries)
