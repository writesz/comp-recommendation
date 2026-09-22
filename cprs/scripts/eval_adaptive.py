"""
Evaluate the online cross-platform skill model (the difficulty/adaptive idea).

Two claims, both on the unified difficulty scale:

  (A) Cross-platform cold-start skill. For a user new to Codeforces, how well can
      we estimate the difficulty level they will actually operate at (their future
      CF skill = 75th-pct difficulty of their later CF solves)? We compare:
        population   : global CF difficulty prior (what you'd guess with nothing)
        cf_k         : from their first k CF solves only
        atcoder      : from their AtCoder history only (zero CF solves)
        atcoder+cf_k : AtCoder seed, then live-corrected by k CF solves
      If cross-platform difficulty transfers, `atcoder` beats `population` and
      rivals several CF solves; `atcoder+cf_k` is best — cross-platform seed plus
      live correction.

  (B) Online convergence. Processing solves sequentially, the live estimate's
      error against the user's true level falls as evidence accrues.

Outputs data/adaptive_results.json.

Usage:
    python -m scripts.eval_adaptive
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.skill import OnlineSkillEstimator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS_DIR = DATA_DIR / "interactions"
Q = 0.75
console = Console()


def load_solves_with_difficulty():
    diff = {p["cprs_id"]: p.get("difficulty_normalized")
            for p in json.load(open(DATA_DIR / "cprs_unified_tagged.json"))}
    cf: dict[str, list[tuple[int, float]]] = defaultdict(list)
    ac: dict[str, list[float]] = defaultdict(list)
    for line in open(INTERACTIONS_DIR / "cf_submissions.jsonl"):
        r = json.loads(line)
        d = diff.get(r["problem_id"])
        if d is not None:
            cf[r["handle"]].append((r["solve_time"], d))
    for line in open(INTERACTIONS_DIR / "ac_submissions.jsonl"):
        r = json.loads(line)
        d = diff.get(r["problem_id"])
        if d is not None:
            ac[r["handle"]].append(d)
    for h in cf:
        cf[h].sort()
    return cf, ac


def main() -> None:
    cf, ac = load_solves_with_difficulty()
    all_cf_diffs = np.array([d for v in cf.values() for _, d in v])
    population = float(np.percentile(all_cf_diffs, Q * 100))
    logger.info(f"population CF skill prior (75th pct) = {population:.3f}")

    # ---- (A) Cross-platform cold-start skill ----
    K = [1, 3, 5]
    max_k = max(K)
    probes = [h for h in cf if h in ac and len(cf[h]) >= max_k + 10 and len(ac[h]) >= 10]
    logger.info(f"(A) {len(probes)} cross-platform probe users")

    err = defaultdict(list)
    for h in probes:
        cf_diffs = [d for _, d in cf[h]]
        target = np.percentile(cf_diffs[max_k:], Q * 100)   # future CF skill (no leakage)
        ac_diffs = ac[h]

        err["population"].append(abs(population - target))
        est_ac = OnlineSkillEstimator(q=Q).seed(ac_diffs)
        err["atcoder"].append(abs(est_ac.theta - target))
        for k in K:
            first_k = cf_diffs[:k]
            err[f"cf_{k}"].append(abs(np.percentile(first_k, Q * 100) - target))
            est = OnlineSkillEstimator(q=Q).seed(ac_diffs)
            for d in first_k:
                est.update(d)
            err[f"atcoder+cf_{k}"].append(abs(est.theta - target))

    tA = Table(title=f"(A) Cold-start CF-skill estimation error — MAE (n={len(probes)})")
    tA.add_column("predictor"); tA.add_column("MAE", justify="right"); tA.add_column("vs population", justify="right")
    base = np.mean(err["population"])
    for name in ["population", "cf_1", "atcoder", "cf_3", "atcoder+cf_1", "cf_5",
                 "atcoder+cf_3", "atcoder+cf_5"]:
        mae = float(np.mean(err[name]))
        tA.add_row(name, f"{mae:.4f}", f"{100*(base-mae)/base:+.0f}%" if name != "population" else "—")
    console.print(tA)

    # ---- (B) Online convergence ----
    seen_grid = [1, 2, 3, 5, 10, 20, 40]
    conv_users = [h for h in cf if len(cf[h]) >= 60]
    conv = {n: [] for n in seen_grid}
    for h in conv_users:
        cf_diffs = [d for _, d in cf[h]]
        target = np.percentile(cf_diffs[40:], Q * 100)
        est = OnlineSkillEstimator(q=Q, init=population)
        for i, d in enumerate(cf_diffs, 1):
            est.update(d)
            if i in conv:
                conv[i].append(abs(est.theta - target))
    tB = Table(title=f"(B) Online skill-estimate error vs #solves seen (n={len(conv_users)})")
    tB.add_column("metric"); [tB.add_column(f"n={n}", justify="right") for n in seen_grid]
    tB.add_row("MAE to true skill", *[f"{np.mean(conv[n]):.4f}" for n in seen_grid])
    console.print(tB)

    out = {
        "population_prior": population,
        "cold_start": {"n_probes": len(probes),
                       "mae": {k: float(np.mean(v)) for k, v in err.items()}},
        "online_convergence": {"n_users": len(conv_users), "seen": seen_grid,
                               "mae": [float(np.mean(conv[n])) for n in seen_grid]},
    }
    (DATA_DIR / "adaptive_results.json").write_text(json.dumps(out, indent=2))
    logger.success(f"Saved adaptive results to {DATA_DIR / 'adaptive_results.json'}")


if __name__ == "__main__":
    main()
