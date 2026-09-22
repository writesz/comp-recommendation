"""
Build a sparse user×problem implicit-feedback matrix from raw CF interaction events.

Day 1 of the final build plan. Reads the solved events produced by
`fetch_interactions.py`, applies iterative min-interaction filtering (a k-core
prune so both users and problems clear a support threshold), and writes a
binary implicit-feedback matrix (solved = 1) plus index mappings and a filtered
events file (with solve times) for the train/test split step.

Usage:
    python -m scripts.build_interactions --min-user 5 --min-problem 5
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from loguru import logger
from rich.console import Console
from rich.table import Table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"

console = Console()


def load_events(path: Path) -> list[tuple[str, str, int]]:
    """Load (handle, problem_id, solve_time), deduping to earliest time per (user, problem)."""
    best: dict[tuple[str, str], int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["handle"], r["problem_id"])
            t = int(r.get("solve_time", 0))
            if key not in best or t < best[key]:
                best[key] = t
    events = [(u, p, t) for (u, p), t in best.items()]
    logger.info(f"Loaded {len(events)} unique (user, problem) events from {path.name}")
    return events


def iterative_filter(
    events: list[tuple[str, str, int]], min_user: int, min_problem: int
) -> list[tuple[str, str, int]]:
    """Prune users/problems below support thresholds until stable (k-core)."""
    cur = events
    it = 0
    while True:
        it += 1
        user_cnt: dict[str, int] = defaultdict(int)
        prob_cnt: dict[str, int] = defaultdict(int)
        for u, p, _ in cur:
            user_cnt[u] += 1
            prob_cnt[p] += 1
        keep = [
            (u, p, t)
            for (u, p, t) in cur
            if user_cnt[u] >= min_user and prob_cnt[p] >= min_problem
        ]
        logger.info(f"filter pass {it}: {len(cur)} -> {len(keep)} events")
        if len(keep) == len(cur):
            return keep
        cur = keep


def build_matrix(events: list[tuple[str, str, int]]):
    """Build a CSR implicit-feedback matrix and the user/problem index lists."""
    users = sorted({u for u, _, _ in events})
    problems = sorted({p for _, p, _ in events})
    u_idx = {u: i for i, u in enumerate(users)}
    p_idx = {p: j for j, p in enumerate(problems)}

    rows = np.fromiter((u_idx[u] for u, _, _ in events), dtype=np.int32, count=len(events))
    cols = np.fromiter((p_idx[p] for _, p, _ in events), dtype=np.int32, count=len(events))
    vals = np.ones(len(events), dtype=np.float32)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(users), len(problems)))
    return mat, users, problems


def report(mat, users: list[str], problems: list[str]) -> None:
    nnz = mat.nnz
    n_u, n_p = mat.shape
    density = nnz / (n_u * n_p)
    per_user = np.asarray(mat.sum(axis=1)).ravel()
    per_prob = np.asarray(mat.sum(axis=0)).ravel()

    t = Table(title="Interaction matrix")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("users", f"{n_u:,}")
    t.add_row("problems", f"{n_p:,}")
    t.add_row("interactions (nnz)", f"{nnz:,}")
    t.add_row("density", f"{density:.5%}")
    t.add_row("solves/user  (mean/median/min/max)",
              f"{per_user.mean():.1f} / {np.median(per_user):.0f} / {per_user.min():.0f} / {per_user.max():.0f}")
    t.add_row("solves/problem (mean/median/min/max)",
              f"{per_prob.mean():.1f} / {np.median(per_prob):.0f} / {per_prob.min():.0f} / {per_prob.max():.0f}")
    console.print(t)


def main(min_user: int, min_problem: int) -> None:
    events_path = INTERACTIONS_DIR / "cf_submissions.jsonl"
    events = load_events(events_path)
    events = iterative_filter(events, min_user, min_problem)
    if not events:
        logger.error("No events survived filtering — lower thresholds or fetch more data.")
        sys.exit(1)

    mat, users, problems = build_matrix(events)
    report(mat, users, problems)

    sp.save_npz(INTERACTIONS_DIR / "matrix.npz", mat)
    (INTERACTIONS_DIR / "mappings.json").write_text(
        json.dumps({"users": users, "problems": problems})
    )
    with open(INTERACTIONS_DIR / "events_filtered.jsonl", "w") as f:
        for u, p, t in events:
            f.write(json.dumps({"handle": u, "problem_id": p, "solve_time": t}) + "\n")

    logger.success(
        f"Saved matrix.npz ({mat.shape[0]}×{mat.shape[1]}, {mat.nnz} nnz), "
        f"mappings.json, events_filtered.jsonl to {INTERACTIONS_DIR}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build user×problem interaction matrix")
    ap.add_argument("--min-user", type=int, default=5, help="min solves for a user to be kept")
    ap.add_argument("--min-problem", type=int, default=5, help="min solvers for a problem to be kept")
    args = ap.parse_args()
    main(args.min_user, args.min_problem)
