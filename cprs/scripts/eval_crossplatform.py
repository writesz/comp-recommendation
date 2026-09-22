"""
Cross-platform value experiment: does AtCoder history improve CF recommendations?

Headline analysis. Two complementary tests, both evaluated on the *unchanged*
Codeforces held-out temporal test set, over the cross-platform cohort:

  Design A (aggregate value): train ALS on CF-only vs the CF+AtCoder-augmented
    matrix; compare CF-side recommendation quality for cohort users with full
    history. Tests whether cross-platform signal helps on average.

  Design B (cold-start remedy — the thesis): using item factors from the merged
    fit, fold each cohort user in from only their k most-recent CF solves, WITH
    vs WITHOUT their AtCoder history, and measure CF recommendation quality vs k.
    If cross-platform history cures cold-start, the WITH-AtCoder curve dominates
    most at small k (thin CF history).

Outputs data/xplat_results.json.

Usage:
    python -m scripts.eval_crossplatform --min-ac 10
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
K_VALUES = [1, 2, 3, 5, 10, 20]
console = Console()


def metrics_at_10(scores: np.ndarray, mask: np.ndarray, rel: set) -> tuple[float, float, float]:
    """Return (nDCG@10, hit@10, recall@10) for a CF score vector."""
    s = scores.copy()
    s[mask] = -np.inf
    top = np.argpartition(-s, 10)[:10]
    top = top[np.argsort(-s[top])]
    hits = np.array([1.0 if p in rel else 0.0 for p in top])
    dcg = (hits / np.log2(np.arange(2, 12))).sum()
    idcg = (1.0 / np.log2(np.arange(2, 2 + min(len(rel), 10)))).sum()
    return dcg / idcg, (1.0 if hits.any() else 0.0), hits.sum() / len(rel)


def foldin(Y: np.ndarray, obs: np.ndarray, alpha: float, reg_I: np.ndarray) -> np.ndarray:
    """ALS fold-in: user factor from observed item columns against fixed item factors."""
    if obs.size == 0:
        return np.zeros(Y.shape[1])
    Yo = Y[obs]
    A = alpha * (Yo.T @ Yo) + reg_I
    b = alpha * Yo.sum(axis=0)
    return np.linalg.solve(A, b)


def load_cf_train_times() -> dict[str, list[int]]:
    """Per-handle CF TRAIN item cols, most-recent first (test items excluded)."""
    mappings = json.loads((INTERACTIONS_DIR / "mappings.json").read_text())
    users, problems = mappings["users"], mappings["problems"]
    u_idx = {u: i for i, u in enumerate(users)}
    p_idx = {p: j for j, p in enumerate(problems)}
    test_raw = json.loads((INTERACTIONS_DIR / "test.json").read_text())
    test_cols = {u_idx[u]: {p_idx[p] for p in v} for u, v in test_raw.items()}
    tmp: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with open(INTERACTIONS_DIR / "events_filtered.jsonl") as f:
        for line in f:
            r = json.loads(line)
            row, col = u_idx[r["handle"]], p_idx[r["problem_id"]]
            if col in test_cols.get(row, set()):
                continue
            tmp[row].append((r["solve_time"], col))
    out = {}
    for row, evs in tmp.items():
        evs.sort(reverse=True)
        out[row] = [c for _, c in evs]
    return out


def main(min_ac: int) -> None:
    merged = sp.load_npz(INTERACTIONS_DIR / "merged_train.npz").tocsr()
    xmap = json.loads((INTERACTIONS_DIR / "xplat_mappings.json").read_text())
    users, n_cf = xmap["users"], xmap["n_cf"]
    u_idx = {u: i for i, u in enumerate(users)}
    p_idx = {p: j for j, p in enumerate(xmap["problems"])}
    cohort = {int(k): v for k, v in json.loads((INTERACTIONS_DIR / "cohort.json").read_text()).items()}

    test_raw = json.loads((INTERACTIONS_DIR / "test.json").read_text())
    cf_test = {u_idx[u]: {p_idx[p] for p in v} for u, v in test_raw.items()}

    cf_only_train = merged[:, :n_cf].tocsr()
    cf_train_times = load_cf_train_times()

    # ---- Design A: full-history CF-only vs cross-platform training ----
    logger.info("Design A: training CF-only and cross-platform models...")
    m_cfonly = CollaborativeModel().fit(cf_only_train)
    m_cross = CollaborativeModel().fit(merged)
    Vcf_only = m_cfonly.item_factors[:n_cf]
    Vcross_cf = m_cross.item_factors[:n_cf]

    cohort_rows = [r for r in cohort if cf_test.get(r)]
    aggA = {"cf_only": defaultdict(list), "cross": defaultdict(list)}
    for r in cohort_rows:
        rel = cf_test[r]
        mask = cf_only_train[r].indices  # observed CF solves
        for name, U, V in (("cf_only", m_cfonly.user_factors, Vcf_only),
                           ("cross", m_cross.user_factors, Vcross_cf)):
            nd, hit, rec = metrics_at_10(U[r] @ V.T, mask, rel)
            aggA[name]["ndcg"].append(nd); aggA[name]["hit"].append(hit); aggA[name]["recall"].append(rec)

    tA = Table(title=f"Design A — full-history CF recs, cohort (n={len(cohort_rows)})")
    for c in ("model", "nDCG@10", "HitRate@10", "Recall@10"):
        tA.add_column(c, justify="right" if c != "model" else "left")
    for name in ("cf_only", "cross"):
        tA.add_row(name, *[f"{np.mean(aggA[name][m]):.4f}" for m in ("ndcg", "hit", "recall")])
    console.print(tA)

    # ---- Design B: cold-start on Codeforces — what helps a brand-new CF user? ----
    # A genuinely new CF user has zero CF history. Their only options are
    # popularity (non-personalized) or their cross-platform history. We compare:
    #   cf_only(k) : ALS fold-in from k revealed CF solves (needs CF history)
    #   popularity : flat non-personalized reference (the k=0 fallback)
    #   cross_nbr  : AtCoder-neighborhood transfer — score CF problems by what
    #                AtCoder-similar users solved on CF (uses NO CF history)
    Y = m_cross.item_factors.astype(np.float64)
    reg_I = m_cross.regularization * np.eye(m_cross.factors)
    alpha = m_cross.alpha
    cf_pop = np.asarray(cf_only_train.sum(axis=0)).ravel()  # CF popularity

    # Cohort AtCoder + CF submatrices for neighborhood transfer.
    cohort_list = cohort_rows
    row_pos = {r: i for i, r in enumerate(cohort_list)}
    AC_sub = merged[cohort_list][:, n_cf:].astype(np.float64)          # (C × n_ac)
    CF_sub = cf_only_train[cohort_list].astype(np.float64)             # (C × n_cf)
    ac_norm = np.sqrt(np.asarray(AC_sub.multiply(AC_sub).sum(axis=1)).ravel())
    ac_norm[ac_norm == 0] = 1.0

    def cross_nbr_scores(local_i: int) -> np.ndarray:
        """CF problem scores from AtCoder-similar cohort neighbors (cosine over AC)."""
        sims = np.asarray(AC_sub.dot(AC_sub[local_i].T).todense()).ravel() / (ac_norm * ac_norm[local_i])
        sims[local_i] = 0.0  # exclude self
        return sims @ CF_sub  # (n_cf,)

    probes = [r for r in cohort_rows
              if cohort[r] >= min_ac and len(cf_train_times.get(r, [])) >= max(K_VALUES)]
    logger.info(f"Design B: {len(probes)} probe users (>= {min_ac} AC solves, "
                f">= {max(K_VALUES)} CF train solves)")

    curveB = {"cf_only": {k: [] for k in K_VALUES}}
    flat = {"popularity": [], "cross_nbr": []}
    no_mask = np.array([], dtype=np.int64)
    for r in probes:
        rel = cf_test[r]
        cf_hist = cf_train_times[r]
        for k in K_VALUES:
            obs_cf = np.array(cf_hist[:k], dtype=np.int64)
            x_cf = foldin(Y, obs_cf, alpha, reg_I)
            nd, _, _ = metrics_at_10(Y[:n_cf] @ x_cf, obs_cf, rel)
            curveB["cf_only"][k].append(nd)
        # k=0 references (no CF history known to the system)
        flat["popularity"].append(metrics_at_10(cf_pop, no_mask, rel)[0])
        flat["cross_nbr"].append(metrics_at_10(cross_nbr_scores(row_pos[r]), no_mask, rel)[0])

    tB = Table(title=f"Design B — CF cold-start nDCG@10 (n={len(probes)})")
    tB.add_column("signal"); tB.add_column("nDCG@10", justify="right")
    tB.add_row("popularity (k=0, no CF hist)", f"{np.mean(flat['popularity']):.4f}")
    tB.add_row("cross_nbr (k=0, AtCoder only)", f"{np.mean(flat['cross_nbr']):.4f}")
    for k in K_VALUES:
        tB.add_row(f"cf_only (k={k} CF solves)", f"{np.mean(curveB['cf_only'][k]):.4f}")
    console.print(tB)

    out = {
        "design_a": {"n_cohort": len(cohort_rows),
                     "results": {n: {m: float(np.mean(aggA[n][m])) for m in ("ndcg", "hit", "recall")}
                                 for n in ("cf_only", "cross")}},
        "design_b": {"k_values": K_VALUES, "n_probes": len(probes), "min_ac": min_ac,
                     "flat": {n: float(np.mean(v)) for n, v in flat.items()},
                     "cf_only_ndcg10": {k: float(np.mean(curveB["cf_only"][k])) for k in K_VALUES}},
    }
    (DATA_DIR / "xplat_results.json").write_text(json.dumps(out, indent=2))
    logger.success(f"Saved cross-platform results to {DATA_DIR / 'xplat_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-platform value experiment")
    ap.add_argument("--min-ac", type=int, default=10, help="min AtCoder solves for Design B probes")
    args = ap.parse_args()
    main(args.min_ac)
