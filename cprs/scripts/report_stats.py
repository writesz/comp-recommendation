"""Descriptive statistics quoted in the final report.

Produces data/report_stats.json. Every number the report states about the
catalogue, the interaction matrix or the choice-set size a learner faces is
regenerated here, so the prose can be checked against the data.

Run:  python scripts/report_stats.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INTER = DATA / "interactions"

# Unified tag vocabulary (Appendix C). Tags outside it are carried through with an
# ``other:`` prefix (contest-series labels, platform-specific oddities) and must NOT be
# counted as topic coverage -- AtCoder's "tags", for instance, are contest names such as
# ``abc``, which carry no topic information at all.
TAXONOMY = set(
    """array backtracking bfs binary_search bitmask brute_force combinatorics
    constructive data_structures design dfs divide_and_conquer dynamic_programming
    fft game_theory geometry graphs greedy hash_table hashing heap implementation
    linked_list math matrix monotonic_stack network_flow number_theory prefix_sum
    probability queue recursion segment_tree shortest_paths simulation sliding_window
    sorting stack string_matching strings trees trie two_pointers""".split()
)
TAG_SEP = "|"


def has_topic_tag(s):
    """True iff the problem carries at least one canonical taxonomy topic."""
    return any(t.strip() in TAXONOMY for t in str(s).split(TAG_SEP))


def tagger_coverage():
    """Topic coverage before vs. after the NLP tagger, per platform."""
    before = pd.read_csv(DATA / "cprs_unified.csv")
    after = pd.read_csv(DATA / "cprs_unified_tagged.csv")
    out = {}
    for plat in sorted(after["platform"].unique()):
        b = before[before["platform"] == plat]["tags_unified_str"].fillna("")
        a = after[after["platform"] == plat]["tags_unified_str"].fillna("")
        out[plat] = {
            "n": int(len(a)),
            "topic_tagged_before": int(b.map(has_topic_tag).sum()),
            "topic_tagged_after": int(a.map(has_topic_tag).sum()),
        }
    return out


def catalogue_stats():
    df = pd.read_csv(DATA / "cprs_unified_tagged.csv")
    out = {"n_problems": int(len(df)), "platforms": {}}
    for plat, g in df.groupby("platform"):
        tagged = g["tags_unified_str"].fillna("").map(has_topic_tag)
        diff = g["difficulty_normalized"].dropna()
        out["platforms"][plat] = {
            "n": int(len(g)),
            "topic_tagged_frac": round(float(tagged.mean()), 4),
            "has_difficulty_frac": round(float(g["difficulty_normalized"].notna().mean()), 4),
            "difficulty_mean": round(float(diff.mean()), 4) if len(diff) else None,
            "difficulty_median": round(float(diff.median()), 4) if len(diff) else None,
            "difficulty_p10": round(float(diff.quantile(0.10)), 4) if len(diff) else None,
            "difficulty_p90": round(float(diff.quantile(0.90)), 4) if len(diff) else None,
        }
    # untagged burden: how many problems carry no canonical topic tag at all
    untagged = int((~df["tags_unified_str"].fillna("").map(has_topic_tag)).sum())
    out["untopiced_problems"] = untagged
    out["untopiced_frac"] = round(float(untagged / len(df)), 4)
    return out, df


def matrix_stats():
    m = sparse.load_npz(INTER / "matrix.npz").tocsr()
    per_user = np.asarray(m.sum(axis=1)).ravel()
    per_item = np.asarray(m.sum(axis=0)).ravel()
    n_u, n_p = m.shape
    return {
        "n_users": int(n_u),
        "n_problems": int(n_p),
        "n_interactions": int(m.nnz),
        "density_pct": round(100.0 * m.nnz / (n_u * n_p), 4),
        "solves_per_user": {
            "mean": round(float(per_user.mean()), 1),
            "median": int(np.median(per_user)),
            "p25": int(np.percentile(per_user, 25)),
            "p75": int(np.percentile(per_user, 75)),
            "min": int(per_user.min()),
            "max": int(per_user.max()),
        },
        "solvers_per_problem": {
            "mean": round(float(per_item.mean()), 1),
            "median": int(np.median(per_item)),
            "p25": int(np.percentile(per_item, 25)),
            "p75": int(np.percentile(per_item, 75)),
            "max": int(per_item.max()),
        },
        # long tail: share of all solves absorbed by the most-solved decile
        "top_decile_solve_share": round(
            float(np.sort(per_item)[::-1][: max(1, n_p // 10)].sum() / per_item.sum()), 4
        ),
    }


def choice_set_stats(df):
    """How many problems sit inside a typical learner's difficulty band?

    This quantifies the discovery problem the project exists to solve: at any
    moment a user must pick one problem from everything at roughly their level.
    """
    m = sparse.load_npz(INTER / "matrix.npz").tocsr()
    mappings = json.loads((INTER / "mappings.json").read_text())
    # problem index -> cprs_id
    if "problems" in mappings:
        pid_list = mappings["problems"]
    else:  # dict form {cprs_id: idx}
        inv = mappings.get("problem_to_idx") or mappings.get("pid_to_idx")
        pid_list = [None] * len(inv)
        for pid, idx in inv.items():
            pid_list[idx] = pid

    diff_by_id = dict(zip(df["cprs_id"], df["difficulty_normalized"]))
    col_diff = np.array([diff_by_id.get(p, np.nan) for p in pid_list], dtype=float)

    # per-user skill = 75th percentile of solved difficulty (the report's definition)
    skills = []
    for u in range(m.shape[0]):
        d = col_diff[m.indices[m.indptr[u] : m.indptr[u + 1]]]
        d = d[~np.isnan(d)]
        if len(d) >= 5:
            skills.append(np.percentile(d, 75))
    skills = np.array(skills)

    # catalogue-wide: problems within +/- 0.05 normalised difficulty of median skill
    all_diff = df["difficulty_normalized"].dropna().to_numpy()
    med_skill = float(np.median(skills))
    band = 0.05
    in_band = int(((all_diff >= med_skill - band) & (all_diff <= med_skill + band)).sum())

    return {
        "n_users_with_skill": int(len(skills)),
        "skill_median": round(med_skill, 4),
        "skill_p25": round(float(np.percentile(skills, 25)), 4),
        "skill_p75": round(float(np.percentile(skills, 75)), 4),
        "band_halfwidth": band,
        "catalogue_problems_in_band_of_median_user": in_band,
        "catalogue_with_difficulty": int(len(all_diff)),
    }


def cohort_stats():
    cohort = json.loads((INTER / "cohort.json").read_text())
    counts = np.array(list(cohort.values()), dtype=float)
    return {
        "n_cross_platform_users": int(len(counts)),
        "atcoder_solves_total": int(counts.sum()),
        "atcoder_solves_median": int(np.median(counts)),
        "users_with_ge10_ac": int((counts >= 10).sum()),
    }


def main():
    cat, df = catalogue_stats()
    stats = {
        "catalogue": cat,
        "tagger_coverage": tagger_coverage(),
        "interactions": matrix_stats(),
        "choice_set": choice_set_stats(df),
        "cross_platform_cohort": cohort_stats(),
        "split": json.loads((INTER / "split_meta.json").read_text()),
    }
    out = DATA / "report_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
