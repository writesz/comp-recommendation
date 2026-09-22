"""
Time-based train/test split of the user×problem interaction matrix.

Day 1 of the final build plan. For each user we hold out their most recent
solves (by solve_time) as the test set — this is a leave-last-N temporal split,
which mimics real deployment (predict what a user solves *next*) and avoids the
leakage of a random split. The rest form the training matrix that CF/hybrid
models learn from.

Held-out size per user: max(1, min(--holdout-n, floor(--holdout-frac * n_solves)))
so heavy users cap at --holdout-n and light users still keep most solves for training.

Outputs:
- train_matrix.npz   — training interactions (same shape as full matrix)
- test.json          — {handle: [held-out problem_ids]} ground truth for evaluation
- split_meta.json    — parameters + summary stats

Usage:
    python -m scripts.split_interactions --holdout-n 10 --holdout-frac 0.2
"""
import argparse
import json
import sys
from collections import defaultdict
from math import floor
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from loguru import logger
from rich.console import Console

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"

console = Console()


def load_filtered_events() -> dict[str, list[tuple[str, int]]]:
    """Return {handle: [(problem_id, solve_time), ...]} from the filtered events file."""
    by_user: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with open(INTERACTIONS_DIR / "events_filtered.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_user[r["handle"]].append((r["problem_id"], int(r["solve_time"])))
    return by_user


def main(holdout_n: int, holdout_frac: float) -> None:
    mappings = json.loads((INTERACTIONS_DIR / "mappings.json").read_text())
    users, problems = mappings["users"], mappings["problems"]
    u_idx = {u: i for i, u in enumerate(users)}
    p_idx = {p: j for j, p in enumerate(problems)}

    by_user = load_filtered_events()

    train_rows, train_cols = [], []
    test: dict[str, list[str]] = {}
    n_test_events = 0

    for u in users:
        solves = sorted(by_user[u], key=lambda x: x[1])  # ascending by time
        n = len(solves)
        k = max(1, min(holdout_n, floor(holdout_frac * n)))
        k = min(k, n - 1)  # always leave >=1 train interaction
        train_part = solves[: n - k]
        test_part = solves[n - k:]
        for pid, _ in train_part:
            train_rows.append(u_idx[u])
            train_cols.append(p_idx[pid])
        held = [pid for pid, _ in test_part]
        if held:
            test[u] = held
            n_test_events += len(held)

    train = sp.csr_matrix(
        (np.ones(len(train_rows), dtype=np.float32), (train_rows, train_cols)),
        shape=(len(users), len(problems)),
    )

    sp.save_npz(INTERACTIONS_DIR / "train_matrix.npz", train)
    (INTERACTIONS_DIR / "test.json").write_text(json.dumps(test))
    meta = {
        "holdout_n": holdout_n,
        "holdout_frac": holdout_frac,
        "n_users": len(users),
        "n_problems": len(problems),
        "train_interactions": int(train.nnz),
        "test_interactions": n_test_events,
        "users_with_test": len(test),
        "avg_heldout_per_user": round(n_test_events / max(1, len(test)), 2),
    }
    (INTERACTIONS_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))

    console.print(meta)
    logger.success(
        f"Saved train_matrix.npz ({train.nnz} train nnz), test.json "
        f"({n_test_events} held-out solves over {len(test)} users), split_meta.json"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Time-based train/test split")
    ap.add_argument("--holdout-n", type=int, default=10, help="max held-out solves per user")
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="fraction of solves held out (capped by --holdout-n)")
    args = ap.parse_args()
    main(args.holdout_n, args.holdout_frac)
