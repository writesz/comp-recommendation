"""
Statistical significance of model differences on per-user nDCG@10.

Day 5 of the final build plan (a 1st-class requirement: significance + effect
sizes, not just point estimates). Uses the per-user nDCG@10 arrays saved by
evaluate.py and runs paired Wilcoxon signed-rank tests (the metric distributions
are non-normal and zero-inflated, so a paired non-parametric test is appropriate)
between the CF model and every other model, reporting:

  - median per-user difference (CF - other)
  - Wilcoxon signed-rank p-value (two-sided)
  - rank-biserial effect size r = (n_pos - n_neg) / n_nonzero  (paired)
  - bootstrap 95% CI on the mean nDCG@10 for each model

Usage:
    python -m scripts.significance --reference cf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
console = Console()
RESERVED = {"rows", "density"}


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_biserial(diff: np.ndarray) -> float:
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    return float((np.sum(nz > 0) - np.sum(nz < 0)) / nz.size)


def main(reference: str) -> None:
    data = np.load(INTERACTIONS_DIR / "per_user_ndcg.npz")
    models = [k for k in data.files if k not in RESERVED]
    if reference not in models:
        logger.error(f"reference '{reference}' not in {models}")
        sys.exit(1)
    ref = data[reference]
    logger.info(f"{len(ref)} users; models: {models}")

    out = {"reference": reference, "n_users": int(len(ref)), "models": {}}

    tci = Table(title="nDCG@10 with bootstrap 95% CI")
    tci.add_column("model"); tci.add_column("mean", justify="right"); tci.add_column("95% CI", justify="right")
    for m in models:
        lo, hi = bootstrap_ci(data[m])
        tci.add_row(m, f"{data[m].mean():.4f}", f"[{lo:.4f}, {hi:.4f}]")
        out["models"][m] = {"mean_ndcg10": float(data[m].mean()), "ci95": [lo, hi]}
    console.print(tci)

    tsig = Table(title=f"Paired Wilcoxon vs '{reference}' (per-user nDCG@10)")
    for c in ("model", "median Δ", "p-value", "effect r", "sig"):
        tsig.add_column(c, justify="right" if c != "model" else "left")
    for m in models:
        if m == reference:
            continue
        diff = ref - data[m]
        med = float(np.median(diff))
        r = rank_biserial(diff)
        try:
            _, p = wilcoxon(ref, data[m], zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p = 1.0
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        tsig.add_row(m, f"{med:+.4f}", f"{p:.2e}", f"{r:+.3f}", sig)
        out["models"][m].update({"median_diff_vs_ref": med, "wilcoxon_p": float(p), "effect_r": r})
    console.print(tsig)

    (DATA_DIR / "significance_results.json").write_text(json.dumps(out, indent=2))
    logger.success(f"Saved significance results to {DATA_DIR / 'significance_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Significance testing on per-user nDCG@10")
    ap.add_argument("--reference", default="cf", help="model to compare all others against")
    args = ap.parse_args()
    main(args.reference)
