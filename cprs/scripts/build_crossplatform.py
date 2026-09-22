"""
Build a merged Codeforces+AtCoder interaction matrix over the existing user set.

Cross-platform value analysis. Reuses the 2,459-user CF matrix and its temporal
train/test split, and augments it with each user's AtCoder solves as extra
problem columns. This lets us ask, on an unchanged CF-side test set, whether a
user's cross-platform history improves their Codeforces recommendations —
i.e. whether cross-platform data cures single-platform cold-start.

AtCoder columns are kept only if solved by >= --min-ac-solvers cohort users, so
their learned item factors rest on a real collaborative bridge (not one user).
AtCoder solves are training-only signal (never prediction targets), so all of a
user's AtCoder solves go into train.

Outputs (data/interactions/):
- merged_train.npz     : 2459 × (n_cf + n_ac) implicit-feedback train matrix
- xplat_mappings.json  : {users, problems (cf then ac), n_cf}
- cohort.json          : {user_row: n_ac_solves} for users with kept AtCoder solves

Usage:
    python -m scripts.build_crossplatform --min-ac-solvers 3
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
console = Console()


def main(min_ac_solvers: int) -> None:
    cf_train = sp.load_npz(INTERACTIONS_DIR / "train_matrix.npz").tocsr()
    mappings = json.loads((INTERACTIONS_DIR / "mappings.json").read_text())
    users, cf_problems = mappings["users"], mappings["problems"]
    u_idx = {u: i for i, u in enumerate(users)}
    n_users, n_cf = cf_train.shape

    # Gather AtCoder solves for users in our CF user set.
    ac_by_user: dict[int, set[str]] = defaultdict(set)
    ac_solver_count: dict[str, int] = defaultdict(int)
    seen_pairs: set[tuple[int, str]] = set()
    with open(INTERACTIONS_DIR / "ac_submissions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            row = u_idx.get(r["handle"])
            if row is None:
                continue  # AtCoder user not in the filtered CF user set
            pid = r["problem_id"]
            if (row, pid) in seen_pairs:
                continue
            seen_pairs.add((row, pid))
            ac_by_user[row].add(pid)
            ac_solver_count[pid] += 1

    # Keep AtCoder problems with enough cohort solvers to form a collaborative bridge.
    kept_ac = sorted(p for p, c in ac_solver_count.items() if c >= min_ac_solvers)
    ac_col = {p: n_cf + j for j, p in enumerate(kept_ac)}
    logger.info(
        f"AtCoder: {len(ac_solver_count)} problems solved by cohort; "
        f"{len(kept_ac)} kept (>= {min_ac_solvers} solvers)"
    )

    # Build merged matrix: CF train (unchanged) + AtCoder augmentation columns.
    rows, cols = list(cf_train.nonzero()[0]), list(cf_train.nonzero()[1])
    cohort: dict[int, int] = {}
    for row, pids in ac_by_user.items():
        kept = [p for p in pids if p in ac_col]
        if not kept:
            continue
        cohort[row] = len(kept)
        for p in kept:
            rows.append(row)
            cols.append(ac_col[p])

    n_ac = len(kept_ac)
    merged = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_cf + n_ac),
    )

    sp.save_npz(INTERACTIONS_DIR / "merged_train.npz", merged)
    (INTERACTIONS_DIR / "xplat_mappings.json").write_text(
        json.dumps({"users": users, "problems": cf_problems + kept_ac, "n_cf": n_cf})
    )
    (INTERACTIONS_DIR / "cohort.json").write_text(
        json.dumps({str(k): v for k, v in cohort.items()})
    )

    ac_counts = np.array(list(cohort.values())) if cohort else np.array([0])
    logger.success(
        f"merged_train.npz: {n_users} users × {n_cf + n_ac} problems "
        f"({n_cf} CF + {n_ac} AC); cohort = {len(cohort)} users "
        f"(AtCoder solves/user: mean {ac_counts.mean():.1f}, median {np.median(ac_counts):.0f}, "
        f"max {ac_counts.max()})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build merged CF+AtCoder matrix")
    ap.add_argument("--min-ac-solvers", type=int, default=3,
                    help="min cohort solvers for an AtCoder problem to be kept")
    args = ap.parse_args()
    main(args.min_ac_solvers)
