"""
Build a multi-platform user cohort by handle-matching CF users against AtCoder.

Cross-platform value analysis (promoted headline thrust). There is no directory
linking a user's Codeforces handle to their AtCoder handle, so we use the common
practice that competitive programmers reuse the same handle across judges: probe
each already-fetched CF handle on AtCoder (kenkoooo). Handles with an AtCoder
account and accepted solves form the cross-platform cohort used to test whether
merging a user's AtCoder history improves their (Codeforces-side) recommendations
— i.e. whether cross-platform data cures single-platform cold-start.

Handle-matching is a heuristic (a shared handle need not be the same person, and
some people use different handles) — documented as a dataset limitation.

Outputs (data/interactions/):
- ac_submissions.jsonl : {handle, problem_id "ac:...", solve_time} accepted solves
- cf_ac_cohort.tsv     : one row per probed handle "handle\t<n_ac_solves>" (resumable)

Usage:
    python -m scripts.fetch_crossplatform
    python -m scripts.fetch_crossplatform --limit 20   # quick validation
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetchers.atcoder import fetch_user_submissions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
UNIFIED_CSV = DATA_DIR / "cprs_unified_tagged.csv"


def load_ac_problem_ids() -> set[str]:
    ids = {
        f"ac:{r['platform_id']}"
        for r in csv.DictReader(open(UNIFIED_CSV))
        if r["platform"] == "atcoder"
    }
    logger.info(f"Loaded {len(ids)} AtCoder problem ids from unified dataset")
    return ids


def load_cf_handles() -> list[str]:
    path = INTERACTIONS_DIR / "cf_fetched_handles.txt"
    handles = [h.strip() for h in path.read_text().splitlines() if h.strip()]
    logger.info(f"{len(handles)} CF handles to probe against AtCoder")
    return handles


def load_probed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()}


def extract_solved(subs: list, valid_ids: set[str]) -> dict[str, int]:
    """{problem_id: earliest AC epoch} for accepted solves present in the unified dataset."""
    solved: dict[str, int] = {}
    for s in subs:
        if s.get("result") != "AC":
            continue
        pid = f"ac:{s.get('problem_id', '')}"
        if pid not in valid_ids:
            continue
        t = s.get("epoch_second", 0)
        if pid not in solved or t < solved[pid]:
            solved[pid] = t
    return solved


def main(limit: Optional[int], request_delay: float, max_retries: int) -> None:
    INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    events_path = INTERACTIONS_DIR / "ac_submissions.jsonl"
    cohort_path = INTERACTIONS_DIR / "cf_ac_cohort.tsv"

    valid_ids = load_ac_problem_ids()
    handles = load_cf_handles()
    if limit:
        handles = handles[:limit]
    probed = load_probed(cohort_path)
    todo = [h for h in handles if h not in probed]
    logger.info(f"{len(probed)} already probed, {len(todo)} to go")

    n_cross = 0
    n_events = 0
    with open(events_path, "a") as ev_f, open(cohort_path, "a") as co_f:
        for handle in tqdm(todo, desc="probe"):
            subs: Optional[list] = None
            for attempt in range(max_retries):
                try:
                    subs = fetch_user_submissions(handle)
                    break
                except Exception as e:
                    wait = request_delay * (2 ** attempt)
                    logger.debug(f"{handle}: attempt {attempt+1} failed ({e}); retry in {wait:.1f}s")
                    time.sleep(wait)
            if subs is None:
                co_f.write(f"{handle}\t-1\n")  # probe failed
                co_f.flush()
                continue

            solved = extract_solved(subs, valid_ids)
            for pid, t in solved.items():
                ev_f.write(json.dumps({"handle": handle, "problem_id": pid, "solve_time": t}) + "\n")
            ev_f.flush()
            co_f.write(f"{handle}\t{len(solved)}\n")
            co_f.flush()
            n_events += len(solved)
            if solved:
                n_cross += 1

    logger.success(
        f"Done. {n_cross} cross-platform users found this run, {n_events} AtCoder solves written."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Handle-match CF users against AtCoder")
    ap.add_argument("--limit", type=int, default=None, help="probe only the first N handles")
    ap.add_argument("--request-delay", type=float, default=1.0, help="base backoff delay (s)")
    ap.add_argument("--max-retries", type=int, default=3, help="retries per handle")
    args = ap.parse_args()
    main(args.limit, args.request_delay, args.max_retries)
