"""
Where, if anywhere, does the density-weighted hybrid beat plain CF?

The headline evaluation reports one number per model over the whole cohort,
which hides the question that actually decides the serving policy: the hybrid
exists to protect users with little history, so it should win *somewhere* even
if it loses on average. This script buckets users by training-history size and
runs the paired comparison inside each bucket.

A caveat about reading the output. nDCG@10 is not comparable *across* buckets:
a user with two held-out problems can reach 1.0, while a user with forty
cannot, because IDCG@10 saturates. Only the within-bucket CF-vs-hybrid
comparison is meaningful, and that comparison is paired over the same users.

Usage:
    python scripts/analyse_hybrid_crossover.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid import HybridModel

DATA = Path(__file__).resolve().parent.parent / "data"
STORE = DATA / "interactions" / "per_user_ndcg.npz"

BUCKETS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, None)]


def main() -> int:
    if not STORE.exists():
        logger.error(f"missing {STORE} — run scripts/evaluate.py first")
        return 1

    d = np.load(STORE)
    dens, cf, hyb, hybp = d["density"], d["cf"], d["hybrid"], d["hybrid_pop"]
    model = HybridModel()

    rows = []
    logger.info("paired CF vs hybrid, bucketed by training-history size")
    print(f"\n{'history':>12} {'n':>5} {'alpha':>6} {'CF':>8} {'hybrid':>8} "
          f"{'hyb_pop':>8} {'CF-hyb':>9} {'p':>10} {'hyb wins':>9}")

    for lo, hi in BUCKETS:
        mask = (dens >= lo) & (dens < (hi if hi is not None else np.inf))
        n = int(mask.sum())
        if n < 10:
            continue
        alpha = float(model.alpha(np.array([dens[mask].mean()]))[0])
        try:
            _, p = wilcoxon(cf[mask], hyb[mask])
        except ValueError:
            p = float("nan")
        delta = float(cf[mask].mean() - hyb[mask].mean())
        wins = float((hyb[mask] > cf[mask]).mean())
        label = f"{lo}-{hi}" if hi is not None else f"{lo}+"
        print(f"{label:>12} {n:5d} {alpha:6.3f} {cf[mask].mean():8.4f} "
              f"{hyb[mask].mean():8.4f} {hybp[mask].mean():8.4f} "
              f"{delta:+9.4f} {p:10.2e} {wins*100:8.1f}%")
        rows.append({
            "bucket": label, "n": n, "mean_alpha": alpha,
            "ndcg10_cf": float(cf[mask].mean()),
            "ndcg10_hybrid": float(hyb[mask].mean()),
            "ndcg10_hybrid_pop": float(hybp[mask].mean()),
            "cf_minus_hybrid": delta, "wilcoxon_p": float(p),
            "hybrid_win_rate": wins,
        })

    worst = max(rows, key=lambda r: r["cf_minus_hybrid"])
    logger.info(
        f"hybrid's largest deficit is in the {worst['bucket']} bucket "
        f"({worst['cf_minus_hybrid']:+.4f} nDCG@10, p={worst['wilcoxon_p']:.1e}) — "
        f"the sparse users it was designed to protect"
    )

    out = {
        "note": (
            "nDCG@10 is not comparable across buckets because IDCG@10 saturates "
            "for users with more than ten held-out items; only the within-bucket "
            "paired CF-vs-hybrid comparison is meaningful."
        ),
        "buckets": rows,
    }
    dest = DATA / "hybrid_crossover.json"
    dest.write_text(json.dumps(out, indent=2))
    logger.success(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
