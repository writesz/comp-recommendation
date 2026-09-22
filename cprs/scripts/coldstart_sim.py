"""
Cold-start simulation: how models degrade as a user's observed history shrinks.

Day 3 of the final build plan. The main evaluation has no true cold-start users
(the matrix is filtered to >=5 solves each), so it can't show where a hybrid
earns its keep. Here we simulate a newly-joined user by revealing only their k
most-recent training solves and folding that into the trained CF space, for
k in {0,1,2,3,5,10,20}, then predicting their held-out test solves.

CF fold-in (given fixed item factors Y from the full-data ALS fit): for observed
items O with confidence alpha and preference 1,
    x = (alpha * Y_O^T Y_O + reg I)^{-1} (alpha * sum_{i in O} y_i)
    scores = Y x
At k=0 there is no signal (x=0) and CF collapses; popularity and the density-
weighted hybrid (alpha = k/(k+k0)) should carry the user until CF has enough
history. The crossover is the empirical justification for the hybrid.

Usage:
    python -m scripts.coldstart_sim
"""
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
from models.hybrid import normalize_rows

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
K_VALUES = [0, 1, 2, 3, 5, 10, 20]
K0 = 10.0  # hybrid density midpoint (matches HybridModel default)
console = Console()


def ndcg_hit_at_10(scores: np.ndarray, observed: np.ndarray, rel: set) -> tuple[float, float]:
    s = scores.copy()
    s[observed] = -np.inf
    top = np.argpartition(-s, 10)[:10]
    top = top[np.argsort(-s[top])]
    hits = np.array([1.0 if p in rel else 0.0 for p in top])
    dcg = (hits / np.log2(np.arange(2, 12))).sum()
    idcg = (1.0 / np.log2(np.arange(2, 2 + min(len(rel), 10)))).sum()
    return dcg / idcg, (1.0 if hits.any() else 0.0)


def main() -> None:
    train = sp.load_npz(INTERACTIONS_DIR / "train_matrix.npz").tocsr()
    mappings = json.loads((INTERACTIONS_DIR / "mappings.json").read_text())
    users, problems = mappings["users"], mappings["problems"]
    u_idx = {u: i for i, u in enumerate(users)}
    p_idx = {p: j for j, p in enumerate(problems)}
    test_raw = json.loads((INTERACTIONS_DIR / "test.json").read_text())

    # Per-user time-sorted TRAIN item columns (descending: most-recent first).
    test_cols = {u_idx[u]: set(p_idx[p] for p in v) for u, v in test_raw.items()}
    train_by_user_time: dict[int, list[int]] = defaultdict(list)
    tmp: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with open(INTERACTIONS_DIR / "events_filtered.jsonl") as f:
        for line in f:
            r = json.loads(line)
            row, col = u_idx[r["handle"]], p_idx[r["problem_id"]]
            if col in test_cols.get(row, set()):
                continue  # skip held-out test items
            tmp[row].append((r["solve_time"], col))
    for row, evs in tmp.items():
        evs.sort(reverse=True)  # most recent first
        train_by_user_time[row] = [c for _, c in evs]

    # Fit CF once on full train to get the item-factor space.
    cf = CollaborativeModel().fit(train)
    Y = cf.item_factors.astype(np.float64)          # (n_items × f)
    reg, alpha, f = cf.regularization, cf.alpha, cf.factors
    reg_I = reg * np.eye(f)
    pop = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    pop_n = normalize_rows(pop[None, :])[0]

    # Probe users: enough history to truncate up to max(K) and have test items.
    probes = [r for r in range(len(users))
              if len(train_by_user_time[r]) >= max(K_VALUES) and test_cols.get(r)]
    logger.info(f"{len(probes)} probe users (>= {max(K_VALUES)} train solves)")

    rows = {m: {k: [] for k in K_VALUES} for m in ("cf", "popularity", "hybrid_pop")}
    for r in probes:
        rel = test_cols[r]
        hist = train_by_user_time[r]
        for k in K_VALUES:
            obs = np.array(hist[:k], dtype=np.int64)  # k most-recent solves
            if k == 0:
                x = np.zeros(f)
            else:
                Yo = Y[obs]
                A = alpha * (Yo.T @ Yo) + reg_I
                b = alpha * Yo.sum(axis=0)
                x = np.linalg.solve(A, b)
            cf_scores = Y @ x
            a = k / (k + K0)
            hyb = a * normalize_rows(cf_scores[None, :])[0] + (1 - a) * pop_n

            for name, sc in (("cf", cf_scores), ("popularity", pop), ("hybrid_pop", hyb)):
                nd, _ = ndcg_hit_at_10(sc, obs, rel)
                rows[name][k].append(nd)

    # Table: nDCG@10 vs observed history size k
    t = Table(title=f"Cold-start: nDCG@10 vs observed history size (n={len(probes)} probe users)")
    t.add_column("model")
    for k in K_VALUES:
        t.add_column(f"k={k}", justify="right")
    summary = {}
    for name in ("cf", "popularity", "hybrid_pop"):
        means = {k: float(np.mean(rows[name][k])) for k in K_VALUES}
        summary[name] = means
        t.add_row(name, *[f"{means[k]:.4f}" for k in K_VALUES])
    console.print(t)

    (DATA_DIR / "coldstart_results.json").write_text(
        json.dumps({"k_values": K_VALUES, "n_probes": len(probes), "ndcg10": summary}, indent=2)
    )
    logger.success(f"Saved cold-start curve to {DATA_DIR / 'coldstart_results.json'}")


if __name__ == "__main__":
    main()
