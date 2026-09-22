"""
Offline evaluation harness for competitive-programming recommenders.

Days 2–3 of the final build plan. Ranks a held-out (leave-last-N temporal) test
set and reports standard top-K ranking metrics across models and baselines:

    HitRate@K, Precision@K, Recall@K, MRR@K, nDCG@K   (K in 5, 10, 20)

Models compared:
  - random       lower-bound sanity baseline
  - popularity   non-personalized "recommend the most-solved problems"
  - content      content-based scorer (topic-gap + difficulty-fit + popularity)
  - cf           collaborative filtering (implicit ALS matrix factorization)
  - hybrid       density-weighted CF + content blend (cold-start aware)
  - hybrid_pop   density-weighted CF + popularity blend

Results are reported overall and sliced by user-activity bucket (train solves),
which is where the cold-start behaviour of the hybrid shows up. Saved to
data/eval_results.json.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --models popularity cf hybrid
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.collaborative import CollaborativeModel
from models.content import ContentScorer
from models.hybrid import HybridModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
K_LIST = [5, 10, 20]
KMAX = max(K_LIST)
METRICS = ("hit", "precision", "recall", "mrr", "ndcg")
# activity buckets by number of training solves
BUCKETS = [("low[5,20)", 5, 20), ("med[20,100)", 20, 100), ("high[100+)", 100, 10 ** 9)]
console = Console()


def load_split():
    train = sp.load_npz(INTERACTIONS_DIR / "train_matrix.npz").tocsr()
    mappings = json.loads((INTERACTIONS_DIR / "mappings.json").read_text())
    test_raw = json.loads((INTERACTIONS_DIR / "test.json").read_text())
    users, problems = mappings["users"], mappings["problems"]
    p_idx = {p: j for j, p in enumerate(problems)}
    test = [np.array([p_idx[pid] for pid in test_raw.get(u, [])], dtype=np.int64) for u in users]
    return train, users, problems, test


def evaluate_model(score_fn, train: sp.csr_matrix, test: list[np.ndarray]):
    """Return (per_user_metrics dict of arrays, evaluated_user_rows array)."""
    n_users = train.shape[0]
    indptr, indices = train.indptr, train.indices
    idcg = {k: np.cumsum(1.0 / np.log2(np.arange(2, k + 2))) for k in K_LIST}
    cols = [f"{m}@{k}" for k in K_LIST for m in METRICS]
    per_user = {c: [] for c in cols}
    eval_rows = []

    for u in range(n_users):
        rel = test[u]
        if rel.size == 0:
            continue
        eval_rows.append(u)
        scores = score_fn(u).copy()
        scores[indices[indptr[u]:indptr[u + 1]]] = -np.inf  # mask already-solved

        top = np.argpartition(-scores, KMAX)[:KMAX]
        top = top[np.argsort(-scores[top])]
        rel_set = set(rel.tolist())
        hit_flags = np.array([1.0 if p in rel_set else 0.0 for p in top])
        first_hit = int(np.argmax(hit_flags)) if hit_flags.any() else -1

        for k in K_LIST:
            hk = hit_flags[:k]
            n_hit = hk.sum()
            per_user[f"hit@{k}"].append(1.0 if n_hit > 0 else 0.0)
            per_user[f"precision@{k}"].append(n_hit / k)
            per_user[f"recall@{k}"].append(n_hit / rel.size)
            per_user[f"mrr@{k}"].append(1.0 / (first_hit + 1) if 0 <= first_hit < k else 0.0)
            dcg = (hk / np.log2(np.arange(2, k + 2))).sum()
            per_user[f"ndcg@{k}"].append(dcg / idcg[k][min(rel.size, k) - 1])

    return {c: np.array(v) for c, v in per_user.items()}, np.array(eval_rows)


def build_score_fns(model_names, train, problems):
    """Return {name: score_fn(user_row)->vector} for requested models."""
    fns = {}
    n_users, n_problems = train.shape
    density = np.asarray(train.sum(axis=1)).ravel().astype(np.float32)
    pop = np.asarray(train.sum(axis=0)).ravel().astype(np.float32)

    need_cf = any(m in model_names for m in ("cf", "hybrid", "hybrid_pop"))
    need_content = any(m in model_names for m in ("content", "hybrid"))
    need_scorer = need_content or "diffmatch" in model_names

    cf_scores = content_scores = None
    scorer = ContentScorer(problems) if need_scorer else None
    if need_cf:
        cf_scores = CollaborativeModel().fit(train).score_all(np.arange(n_users))
    if need_content:
        content_scores = scorer.score_all(train, np.arange(n_users))

    if "random" in model_names:
        rand = np.random.default_rng(42).random((n_users, n_problems), dtype=np.float32)
        fns["random"] = lambda u, S=rand: S[u]
    if "popularity" in model_names:
        fns["popularity"] = lambda u, P=pop: P
    if "content" in model_names:
        fns["content"] = lambda u, S=content_scores: S[u]
    if "diffmatch" in model_names:
        D = scorer.score_all_match(train, np.arange(n_users))
        fns["diffmatch"] = lambda u, S=D: S[u]
    if "cf" in model_names:
        fns["cf"] = lambda u, S=cf_scores: S[u]
    if "hybrid" in model_names:
        H = HybridModel(cold="content").score_matrix(cf_scores, content_scores, pop, density)
        fns["hybrid"] = lambda u, S=H: S[u]
    if "hybrid_pop" in model_names:
        Hp = HybridModel(cold="popularity").score_matrix(
            cf_scores, content_scores if content_scores is not None else cf_scores, pop, density
        )
        fns["hybrid_pop"] = lambda u, S=Hp: S[u]
    return fns, density


def aggregate(per_user, eval_rows, density):
    """Overall + per-bucket means for each metric."""
    d = density[eval_rows]
    out = {"overall": {c: float(v.mean()) for c, v in per_user.items()}, "buckets": {}}
    for name, lo, hi in BUCKETS:
        mask = (d >= lo) & (d < hi)
        out["buckets"][name] = {
            "n_users": int(mask.sum()),
            **{c: (float(v[mask].mean()) if mask.any() else 0.0) for c, v in per_user.items()},
        }
    return out


def main(model_names: list[str]) -> None:
    train, users, problems, test = load_split()
    logger.info(f"Loaded split: {train.shape[0]} users × {train.shape[1]} problems, "
                f"{sum(t.size for t in test)} held-out solves")

    fns, density = build_score_fns(model_names, train, problems)
    results = {}
    per_user_store = {}  # model -> per-user nDCG@10 (for significance tests / figures)
    eval_rows = None
    for name in model_names:
        if name not in fns:
            logger.warning(f"unknown model '{name}', skipping")
            continue
        logger.info(f"Evaluating '{name}'...")
        pu, rows = evaluate_model(fns[name], train, test)
        results[name] = aggregate(pu, rows, density)
        per_user_store[name] = pu["ndcg@10"]
        eval_rows = rows

    # Persist per-user nDCG@10 + activity density for significance testing and plots.
    if eval_rows is not None:
        np.savez(INTERACTIONS_DIR / "per_user_ndcg.npz",
                 rows=eval_rows, density=density[eval_rows], **per_user_store)

    # Overall table
    cols = ["hit@10", "precision@10", "recall@10", "mrr@10", "ndcg@10", "ndcg@20"]
    t = Table(title="Overall (held-out temporal test)")
    t.add_column("model")
    for c in cols:
        t.add_column(c, justify="right")
    for name in model_names:
        if name in results:
            t.add_row(name, *[f"{results[name]['overall'][c]:.4f}" for c in cols])
    console.print(t)

    # Per-bucket table on the headline metric (nDCG@10)
    tb = Table(title="nDCG@10 by user activity (cold-start view)")
    tb.add_column("model")
    for bname, _, _ in BUCKETS:
        n = results[model_names[0]]["buckets"][bname]["n_users"] if results else 0
        tb.add_column(f"{bname}\n(n={n})", justify="right")
    for name in model_names:
        if name in results:
            tb.add_row(name, *[f"{results[name]['buckets'][b]['ndcg@10']:.4f}" for b, _, _ in BUCKETS])
    console.print(tb)

    out = {"k_list": K_LIST, "buckets": [b[0] for b in BUCKETS],
           "n_users_evaluated": sum(1 for t in test if t.size), "results": results}
    (DATA_DIR / "eval_results.json").write_text(json.dumps(out, indent=2))
    logger.success(f"Saved metrics to {DATA_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate recommenders on held-out test set")
    ap.add_argument("--models", nargs="+",
                    default=["random", "popularity", "content", "cf", "hybrid", "hybrid_pop"],
                    help="which models to evaluate")
    args = ap.parse_args()
    main(args.models)
