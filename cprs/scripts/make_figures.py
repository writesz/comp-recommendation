"""
Generate all result figures for the final report.

Reads the saved evaluation JSONs and writes publication-style PNGs to figures/.
Every figure corresponds to a claim in the evaluation so results are visualised,
not just tabulated (a 1st-class requirement).

Figures:
  1 model_comparison      nDCG@10 across models with bootstrap 95% CI
  2 metric_vs_k           nDCG@K (K=5,10,20) per model
  3 activity_buckets      nDCG@10 by user-activity bucket
  4 coldstart_curve       nDCG@10 vs revealed history size (CF fold-in collapse)
  5 xplat_skill_scatter   CF-skill vs AtCoder-skill (unified difficulty, r=0.77)
  6 coldstart_skill_mae   cold-start CF-skill estimation error by predictor
  7 online_convergence    live skill-estimate error vs #solves seen
  8 difficulty_by_platform catalogue difficulty per platform on the unified scale

Usage:
    python -m scripts.make_figures
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
FIG_DIR = ROOT / "figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
BLUE, ORANGE, GREEN, RED, GREY = "#2c7fb8", "#e6842e", "#31a354", "#d62728", "#999999"


def _load(name):
    p = DATA_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_model_comparison():
    ev, sig = _load("eval_results.json"), _load("significance_results.json")
    if not ev:
        return
    order = ["random", "content", "diffmatch", "popularity", "hybrid", "hybrid_pop", "cf"]
    models = [m for m in order if m in ev["results"]]
    vals = [ev["results"][m]["overall"]["ndcg@10"] for m in models]
    errs = None
    if sig:
        lo = [ev["results"][m]["overall"]["ndcg@10"] - sig["models"][m]["ci95"][0] for m in models]
        hi = [sig["models"][m]["ci95"][1] - ev["results"][m]["overall"]["ndcg@10"] for m in models]
        errs = [lo, hi]
    colors = [RED if m == "cf" else BLUE for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, vals, yerr=errs, capsize=4, color=colors)
    ax.set_ylabel("nDCG@10"); ax.set_title("Recommender comparison (held-out temporal test)\nerror bars = bootstrap 95% CI")
    plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison.png"); plt.close(fig)


def fig_metric_vs_k():
    ev = _load("eval_results.json")
    if not ev:
        return
    ks = ev["k_list"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in [("cf", RED), ("hybrid_pop", GREEN), ("popularity", BLUE), ("diffmatch", ORANGE)]:
        if m in ev["results"]:
            ax.plot(ks, [ev["results"][m]["overall"][f"ndcg@{k}"] for k in ks], "o-", color=c, label=m)
    ax.set_xlabel("K"); ax.set_ylabel("nDCG@K"); ax.set_xticks(ks)
    ax.set_title("Ranking quality vs cutoff K"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "metric_vs_k.png"); plt.close(fig)


def fig_activity_buckets():
    ev = _load("eval_results.json")
    if not ev:
        return
    buckets = ev["buckets"]
    models = [m for m in ["popularity", "diffmatch", "hybrid_pop", "cf"] if m in ev["results"]]
    x = np.arange(len(buckets)); w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(models):
        vals = [ev["results"][m]["buckets"][b]["ndcg@10"] for b in buckets]
        ax.bar(x + i * w, vals, w, label=m, color=RED if m == "cf" else None)
    ns = [ev["results"][models[0]]["buckets"][b]["n_users"] for b in buckets]
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(buckets, ns)])
    ax.set_ylabel("nDCG@10"); ax.set_title("Recommendation quality by user activity level"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "activity_buckets.png"); plt.close(fig)


def fig_coldstart_curve():
    cs = _load("coldstart_results.json")
    if not cs:
        return
    ks = cs["k_values"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in [("cf", RED), ("hybrid_pop", GREEN), ("popularity", BLUE)]:
        if m in cs["ndcg10"]:
            ax.plot(ks, [cs["ndcg10"][m][str(k)] if str(k) in cs["ndcg10"][m] else cs["ndcg10"][m][k]
                         for k in ks], "o-", color=c, label=m)
    ax.set_xlabel("revealed history size k (solves)"); ax.set_ylabel("nDCG@10")
    ax.set_title("Cold-start: CF collapses at zero history\n(fold-in simulation)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "coldstart_curve.png"); plt.close(fig)


def fig_xplat_skill_scatter():
    diff = {p["cprs_id"]: p.get("difficulty_normalized")
            for p in json.load(open(DATA_DIR / "cprs_unified_tagged.json"))}
    cf, ac = defaultdict(list), defaultdict(list)
    for line in open(INTERACTIONS_DIR / "cf_submissions.jsonl"):
        r = json.loads(line); d = diff.get(r["problem_id"])
        if d is not None: cf[r["handle"]].append(d)
    for line in open(INTERACTIONS_DIR / "ac_submissions.jsonl"):
        r = json.loads(line); d = diff.get(r["problem_id"])
        if d is not None: ac[r["handle"]].append(d)
    users = [h for h in cf if h in ac and len(cf[h]) >= 10 and len(ac[h]) >= 10]
    xs = np.array([np.percentile(ac[h], 75) for h in users])
    ys = np.array([np.percentile(cf[h], 75) for h in users])
    r, _ = pearsonr(xs, ys)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(xs, ys, alpha=0.5, color=BLUE, s=18)
    m, b = np.polyfit(xs, ys, 1)
    xl = np.array([xs.min(), xs.max()]); ax.plot(xl, m * xl + b, color=RED, lw=2)
    ax.set_xlabel("AtCoder skill (75th-pct normalized difficulty)")
    ax.set_ylabel("Codeforces skill (75th-pct normalized difficulty)")
    ax.set_title(f"Cross-platform skill consistency (n={len(users)})\nPearson r = {r:.2f}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "xplat_skill_scatter.png"); plt.close(fig)


def fig_coldstart_skill_mae():
    ad = _load("adaptive_results.json")
    if not ad:
        return
    mae = ad["cold_start"]["mae"]
    order = ["population", "cf_1", "cf_3", "cf_5", "atcoder", "atcoder+cf_1", "atcoder+cf_3", "atcoder+cf_5"]
    order = [k for k in order if k in mae]
    colors = [GREY if k == "population" else (BLUE if k.startswith("cf_") else GREEN) for k in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(order, [mae[k] for k in order], color=colors)
    ax.axhline(mae["population"], color=GREY, ls="--", lw=1)
    ax.set_ylabel("MAE to true CF skill"); plt.xticks(rotation=25)
    ax.set_title("Cold-start skill estimation: AtCoder history beats the population prior\n"
                 "(blue = CF solves only, green = AtCoder seed [+ live correction])")
    fig.tight_layout(); fig.savefig(FIG_DIR / "coldstart_skill_mae.png"); plt.close(fig)


def fig_online_convergence():
    ad = _load("adaptive_results.json")
    if not ad:
        return
    oc = ad["online_convergence"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(oc["seen"], oc["mae"], "o-", color=GREEN)
    ax.axhline(ad["cold_start"]["mae"]["population"], color=GREY, ls="--", lw=1, label="population prior")
    ax.set_xlabel("# solves observed"); ax.set_ylabel("MAE to true skill")
    ax.set_title("Online skill estimate corrects as evidence accrues"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "online_convergence.png"); plt.close(fig)


def fig_difficulty_by_platform():
    """
    Catalogue difficulty distribution per platform on the unified scale.

    This is the figure behind the normalisation claim in the design chapter: if
    the four platform-specific mappings did not produce a comparable scale, the
    distributions would sit in disjoint bands and a cross-platform difficulty
    match would be meaningless. Small multiples rather than four overlaid
    series, because the panels are the comparison — identity comes from the
    panel label and position, so colour carries no information and one hue is
    used throughout.
    """
    import pandas as pd

    csv = DATA_DIR / "cprs_unified_tagged.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv, usecols=["platform", "difficulty_normalized"])

    order = ["codeforces", "atcoder", "codechef", "leetcode"]
    labels = {"codeforces": "Codeforces", "atcoder": "AtCoder",
              "codechef": "CodeChef", "leetcode": "LeetCode"}
    present = [p for p in order if (df["platform"] == p).any()]

    fig, axes = plt.subplots(len(present), 1, figsize=(6.4, 1.35 * len(present)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 41)

    for ax, plat in zip(axes, present):
        d = df.loc[df["platform"] == plat, "difficulty_normalized"].dropna()
        ax.hist(d, bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
        med = float(d.median())
        ax.axvline(med, color="#333333", ls="--", lw=1.2)
        # LeetCode has three ordinal levels, so its "distribution" is 3 spikes;
        # saying so on the panel stops it reading as a failure of the mapping.
        note = " (3 ordinal levels)" if plat == "leetcode" else ""
        ax.text(0.015, 0.82, f"{labels[plat]}{note}", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        ax.text(0.985, 0.82, f"n={len(d):,}  median {med:.2f}",
                transform=ax.transAxes, fontsize=9, color="#666666",
                ha="right", va="top")
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)

    axes[-1].set_xlabel("normalised difficulty  [0, 1]")
    # Deliberately descriptive, not a claim of alignment: the medians differ by
    # up to 0.12 and the reader should see that rather than be told otherwise.
    axes[0].set_title("Four incompatible difficulty scales, mapped onto one range",
                      fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "difficulty_by_platform.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for fn in (fig_model_comparison, fig_metric_vs_k, fig_activity_buckets, fig_coldstart_curve,
               fig_xplat_skill_scatter, fig_coldstart_skill_mae, fig_online_convergence,
               fig_difficulty_by_platform):
        try:
            fn()
            logger.info(f"✓ {fn.__name__}")
        except Exception as e:
            logger.error(f"✗ {fn.__name__}: {e}")
    logger.success(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
