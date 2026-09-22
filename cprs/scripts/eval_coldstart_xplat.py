"""
Decisive cross-platform cold-start test on *genuinely* CF-cold users.

The main cross-platform eval used experienced users as proxy cold-start users,
which unfairly deflates non-CF signals (their held-out solves are advanced, rare
problems). This script instead evaluates users who are truly new to Codeforces —
few total CF solves — but who have real AtCoder history. Their raw CF histories
survive in cf_submissions.jsonl (fetched before the >=5-solve filter), so no new
fetching is needed. Crucially these users are held out of the CF/AtCoder training
matrices (they were filtered out), so there is no leakage.

For each cold user we temporally split their CF solves and compare, on held-out
CF solves, the signals available to a new user:
  popularity : non-personalized CF popularity
  cf_foldin  : ALS fold-in from their few early CF solves
  cross_nbr  : AtCoder-neighborhood transfer (recommend CF problems that
               AtCoder-similar cohort users solved) — uses NO CF history
  hybrid     : rank-blend of cf_foldin + cross_nbr

Usage:
    python -m scripts.eval_coldstart_xplat --min-cf 3 --max-cf 20 --min-ac 20
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.collaborative import CollaborativeModel
from scripts.eval_crossplatform import foldin, metrics_at_10

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
console = Console()


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Rank-normalize to [0,1] so heterogeneous scores can be blended."""
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1)


def main(min_cf: int, max_cf: int, min_ac: int) -> None:
    merged = sp.load_npz(INTERACTIONS_DIR / "merged_train.npz").tocsr()
    xmap = json.loads((INTERACTIONS_DIR / "xplat_mappings.json").read_text())
    train_users, n_cf = set(xmap["users"]), xmap["n_cf"]
    cf_problems = xmap["problems"][:n_cf]
    kept_ac = xmap["problems"][n_cf:]
    cf_p_idx = {p: j for j, p in enumerate(cf_problems)}
    ac_p_idx = {p: j for j, p in enumerate(kept_ac)}
    cohort = {int(k): v for k, v in json.loads((INTERACTIONS_DIR / "cohort.json").read_text()).items()}

    # Trained item factors (CF portion) for fold-in.
    m_cross = CollaborativeModel().fit(merged)
    Y = m_cross.item_factors.astype(np.float64)
    reg_I = m_cross.regularization * np.eye(m_cross.factors)
    alpha = m_cross.alpha
    cf_only_train = merged[:, :n_cf].tocsr()
    cf_pop = np.asarray(cf_only_train.sum(axis=0)).ravel()

    # Cohort submatrices for neighborhood transfer (rows = cohort users in matrix).
    cohort_list = [r for r in cohort]
    AC_sub = merged[cohort_list][:, n_cf:].astype(np.float64)   # (C × n_ac)
    CF_sub = cf_only_train[cohort_list].astype(np.float64)      # (C × n_cf)
    ac_norm = np.sqrt(np.asarray(AC_sub.multiply(AC_sub).sum(axis=1)).ravel())
    ac_norm[ac_norm == 0] = 1.0

    # All users' full CF solves (pre-filter) and AtCoder solves, from raw event files.
    cf_by_user: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(INTERACTIONS_DIR / "cf_submissions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            j = cf_p_idx.get(r["problem_id"])
            if j is not None:
                cf_by_user[r["handle"]].append((r["solve_time"], j))
    ac_by_user: dict[str, set[int]] = defaultdict(set)
    with open(INTERACTIONS_DIR / "ac_submissions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            j = ac_p_idx.get(r["problem_id"])
            if j is not None:
                ac_by_user[r["handle"]].add(j)

    # Cold users: few CF solves, real AtCoder history, and NOT in the training matrix.
    cold = [
        h for h, solves in cf_by_user.items()
        if min_cf <= len(solves) <= max_cf and len(ac_by_user.get(h, ())) >= min_ac
        and h not in train_users
    ]
    logger.info(f"{len(cold)} genuinely CF-cold cross-platform users "
                f"(CF solves in [{min_cf},{max_cf}], AtCoder >= {min_ac}, held out of training)")

    agg = defaultdict(list)
    for h in cold:
        solves = sorted(cf_by_user[h])
        n = len(solves)
        n_test = max(1, n // 5)
        train_cols = np.array([c for _, c in solves[: n - n_test]], dtype=np.int64)
        rel = {c for _, c in solves[n - n_test:]}
        if not rel:
            continue

        # AtCoder-neighborhood scores for this cold user.
        ac_vec = np.zeros(len(kept_ac))
        idx = list(ac_by_user[h])
        ac_vec[idx] = 1.0
        cold_norm = np.sqrt((ac_vec * ac_vec).sum()) or 1.0
        sims = np.asarray(AC_sub.dot(ac_vec)).ravel() / (ac_norm * cold_norm)
        nbr_scores = sims @ CF_sub

        x = foldin(Y, train_cols, alpha, reg_I)
        foldin_scores = Y[:n_cf] @ x
        hybrid_scores = rank_norm(foldin_scores) + rank_norm(nbr_scores)

        for name, sc in (("popularity", cf_pop), ("cf_foldin", foldin_scores),
                         ("cross_nbr", nbr_scores), ("hybrid", hybrid_scores)):
            nd, hit, rec = metrics_at_10(sc, train_cols, rel)
            agg[f"{name}:ndcg"].append(nd)
            agg[f"{name}:hit"].append(hit)

    t = Table(title=f"Genuinely CF-cold cross-platform users (n={len(cold)})")
    for c in ("signal", "nDCG@10", "HitRate@10"):
        t.add_column(c, justify="right" if c != "signal" else "left")
    for name in ("popularity", "cf_foldin", "cross_nbr", "hybrid"):
        t.add_row(name, f"{np.mean(agg[f'{name}:ndcg']):.4f}", f"{np.mean(agg[f'{name}:hit']):.4f}")
    console.print(t)

    out = {"n_cold": len(cold), "min_cf": min_cf, "max_cf": max_cf, "min_ac": min_ac,
           "results": {name: {"ndcg10": float(np.mean(agg[f"{name}:ndcg"])),
                              "hit10": float(np.mean(agg[f"{name}:hit"]))}
                       for name in ("popularity", "cf_foldin", "cross_nbr", "hybrid")}}
    (DATA_DIR / "coldstart_xplat_results.json").write_text(json.dumps(out, indent=2))
    logger.success(f"Saved to {DATA_DIR / 'coldstart_xplat_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-platform cold-start on genuinely CF-cold users")
    ap.add_argument("--min-cf", type=int, default=3)
    ap.add_argument("--max-cf", type=int, default=20)
    ap.add_argument("--min-ac", type=int, default=20)
    args = ap.parse_args()
    main(args.min_cf, args.max_cf, args.min_ac)
