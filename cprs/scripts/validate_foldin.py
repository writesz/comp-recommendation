"""
Does serving unseen users by fold-in preserve offline ranking quality?

Wiring CF into the web application means scoring people who were never in the
training cohort, which ALS cannot do directly — see the fold-in derivation in
`models/collaborative.py`. This script measures what that costs, under two
progressively harsher tests.

Test A (sanity). Take cohort users, discard their learned factor rows and
recover them by folding their own training history back in. Because fold-in is
the ALS user step, a converged model should reproduce them almost exactly. This
verifies the implementation, not the deployment.

Test B (the honest test). Hold a block of users out of training altogether, fit
the item factors on the remaining users only, then fold the held-out users in
and rank for them. This is the situation a new registration actually creates:
the user contributed nothing to the latent space they are being scored in. The
gap between Test B and the same users' scores under a model trained *with* them
is the true cost of serving strangers.

Usage:
    python scripts/validate_foldin.py [--holdout 400]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from loguru import logger
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.collaborative import CollaborativeModel
from models.metrics import idcg_table, rank_metrics

INTERACTIONS = Path(__file__).resolve().parent.parent / "data" / "interactions"
K_LIST = [10]


def load_split():
    train = sp.load_npz(INTERACTIONS / "train_matrix.npz").tocsr()
    mappings = json.loads((INTERACTIONS / "mappings.json").read_text())
    test_raw = json.loads((INTERACTIONS / "test.json").read_text())
    users, problems = mappings["users"], mappings["problems"]
    p_idx = {p: j for j, p in enumerate(problems)}
    test = [np.array([p_idx[pid] for pid in test_raw.get(u, []) if pid in p_idx],
                     dtype=np.int64) for u in users]
    return train, users, test


def ndcg_for(scores_of, train, test, rows):
    """Mean and per-user nDCG@10 over `rows`, given a row -> score-vector fn."""
    idcg = idcg_table(K_LIST)
    per_user, kept = [], []
    for u in rows:
        rel = test[u]
        if rel.size == 0:
            continue
        m = rank_metrics(
            scores_of(u),
            seen=train.indices[train.indptr[u]:train.indptr[u + 1]],
            relevant=rel,
            k_list=K_LIST,
            idcg=idcg,
        )
        per_user.append(m["ndcg@10"])
        kept.append(u)
    return np.array(per_user), np.array(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", type=int, default=400,
                    help="users withheld from training for Test B")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, users, test = load_split()
    n_users = train.shape[0]
    rng = np.random.default_rng(args.seed)
    logger.info(f"{n_users} cohort users x {train.shape[1]} problems")

    # ---------------- Test A: fold a training user back in ----------------
    full = CollaborativeModel().fit(train)

    def trained_scores(u):
        return full.user_factors[u] @ full.item_factors.T

    def folded_scores(u):
        cols = train.indices[train.indptr[u]:train.indptr[u + 1]]
        return full.score_folded(cols)

    sample = rng.choice(n_users, size=min(600, n_users), replace=False)
    a_trained, rows_a = ndcg_for(trained_scores, train, test, sample)
    a_folded, _ = ndcg_for(folded_scores, train, test, sample)

    logger.info("--- Test A: fold-in of users the model was trained on ---")
    logger.info(f"  trained factors : nDCG@10 {a_trained.mean():.4f}")
    logger.info(f"  folded in       : nDCG@10 {a_folded.mean():.4f}")
    logger.info(f"  retained        : {a_folded.mean() / a_trained.mean() * 100:.2f}% "
                f"over {len(rows_a)} users")

    # ---------------- Test B: users never seen in training ----------------
    held = rng.choice(n_users, size=min(args.holdout, n_users // 2), replace=False)
    held_mask = np.zeros(n_users, dtype=bool)
    held_mask[held] = True
    keep_rows = np.flatnonzero(~held_mask)

    logger.info(f"--- Test B: {len(held)} users withheld from training entirely ---")
    partial = CollaborativeModel().fit(train[keep_rows])

    def stranger_scores(u):
        cols = train.indices[train.indptr[u]:train.indptr[u + 1]]
        return partial.score_folded(cols)

    b_folded, rows_b = ndcg_for(stranger_scores, train, test, held)
    # the same users, scored by the model that *did* train on them
    b_insider, rows_b2 = ndcg_for(trained_scores, train, test, held)
    assert np.array_equal(rows_b, rows_b2)

    logger.info(f"  trained with them  : nDCG@10 {b_insider.mean():.4f}")
    logger.info(f"  folded in as new   : nDCG@10 {b_folded.mean():.4f}")
    retention = b_folded.mean() / b_insider.mean() * 100 if b_insider.mean() else 0
    logger.info(f"  retained           : {retention:.2f}% over {len(rows_b)} users")

    stat, p = wilcoxon(b_folded, b_insider)
    diff = b_folded - b_insider
    logger.info(f"  Wilcoxon p={p:.3e}, median difference {np.median(diff):+.4f}, "
                f"mean difference {diff.mean():+.4f}")

    out = {
        "n_users": int(n_users),
        "test_a": {
            "n": int(len(rows_a)),
            "ndcg10_trained": float(a_trained.mean()),
            "ndcg10_folded": float(a_folded.mean()),
            "retention_pct": float(a_folded.mean() / a_trained.mean() * 100),
        },
        "test_b": {
            "n_held_out": int(len(rows_b)),
            "n_trained_on": int(len(keep_rows)),
            "ndcg10_insider": float(b_insider.mean()),
            "ndcg10_folded": float(b_folded.mean()),
            "retention_pct": float(retention),
            "wilcoxon_p": float(p),
            "median_diff": float(np.median(diff)),
            "mean_diff": float(diff.mean()),
        },
    }
    dest = INTERACTIONS.parent / "foldin_validation.json"
    dest.write_text(json.dumps(out, indent=2))
    logger.success(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
