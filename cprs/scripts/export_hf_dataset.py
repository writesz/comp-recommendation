"""Build the public Hugging Face release of the CPRS problem catalogue.

Emits ``data/hf_release/`` containing a Parquet table of all unified problems,
a generated dataset card, and the statistics the card quotes.

What is released is the *catalogue*: factual problem metadata gathered from
public APIs, plus the three derived columns that are this project's own
contribution -- normalised difficulty, the unified tag taxonomy, and the tags
recovered for AtCoder by the NLP transfer tagger.

Two parts of ``data/`` are deliberately NOT released:

* ``data/statements/`` -- verbatim problem statements are the copyright of the
  respective platforms and are not ours to redistribute. The tagger is trained
  on them locally; only its *predictions* are published.
* ``data/interactions/`` -- real Codeforces handles with per-problem solve
  timestamps. The report's ethics section commits to using the cohort only in
  aggregate, with no individual profiled or published.

Run:  python scripts/export_hf_dataset.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.report_stats import TAXONOMY

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "hf_release"

# Overridden with --repo-id once the Hub repo actually exists.
DEFAULT_REPO_ID = "<your-hf-username>/cprs-competitive-programming"

TAG_SEP = "|"

# The native difficulty scale behind ``difficulty_raw``, which is NOT comparable
# across platforms -- hence the column, so nobody averages the three together.
DIFFICULTY_SOURCE = {
    "codeforces": "codeforces_rating",      # 800-3500, problemsetter-assigned
    "atcoder": "atcoder_estimate",          # kenkoooo Elo-style estimate, ~-500-4000
    "leetcode": "leetcode_band",            # Easy/Medium/Hard -> 0.2/0.5/0.85
}

# Column order in the published table.
COLUMNS = [
    "cprs_id",
    "platform",
    "platform_id",
    "name",
    "url",
    "contest_id",
    "difficulty_raw",
    "difficulty_source",
    "difficulty_normalized",
    "tags_original",
    "tags_unified",
    "tags_predicted",
    "has_canonical_topic",
    "solve_count",
    "acceptance_rate",
    "is_premium",
]


def split_tags(s):
    """``a|b`` -> ``['a', 'b']``; blank/NaN -> ``[]``."""
    if pd.isna(s) or not str(s).strip():
        return []
    return [t.strip() for t in str(s).split(TAG_SEP) if t.strip()]


def load_catalogues():
    """The catalogue before and after the NLP tagger, indexed by cprs_id.

    The diff between the two is what tells us which tags are model-predicted
    rather than human-assigned -- provenance the published table must carry.
    """
    before = pd.read_csv(DATA / "cprs_unified.csv").set_index("cprs_id")
    after = pd.read_csv(DATA / "cprs_unified_tagged.csv")
    return before, after


def build_table():
    before, df = load_catalogues()

    df["tags_original"] = df["tags_original_str"].map(split_tags)
    df["tags_unified"] = df["tags_unified_str"].map(split_tags)

    # A row's tags are predicted iff the tagger added topics the pre-tagger
    # catalogue did not have.
    prior = before["tags_unified_str"].map(split_tags).map(set)
    df["tags_predicted"] = [
        bool(set(tags) - prior.get(cid, set()))
        for cid, tags in zip(df["cprs_id"], df["tags_unified"])
    ]

    # ``other:*`` entries are unmapped platform labels (AtCoder contest series
    # such as ``abc``) and carry no topic information -- they must not count as
    # coverage. Same rule as report_stats.py.
    df["has_canonical_topic"] = df["tags_unified"].map(
        lambda ts: any(t in TAXONOMY for t in ts)
    )

    df["difficulty_source"] = df["platform"].map(DIFFICULTY_SOURCE)
    # A platform's scale label is meaningless where no difficulty is recorded.
    df.loc[df["difficulty_raw"].isna(), "difficulty_source"] = None

    df["contest_id"] = df["contest_id"].astype("string")
    df["solve_count"] = df["solve_count"].astype("Int64")
    df["is_premium"] = df["is_premium"].astype(bool)

    return df[COLUMNS].sort_values("cprs_id").reset_index(drop=True)


def stats(df):
    """Every number the dataset card quotes, regenerated from the table."""
    canonical = sorted({t for ts in df["tags_unified"] for t in ts if t in TAXONOMY})
    other = sorted({t for ts in df["tags_unified"] for t in ts if t not in TAXONOMY})

    per_platform = {}
    for plat, g in df.groupby("platform"):
        per_platform[plat] = {
            "n": int(len(g)),
            "with_difficulty": int(g["difficulty_normalized"].notna().sum()),
            "with_canonical_topic": int(g["has_canonical_topic"].sum()),
            "predicted_tags": int(g["tags_predicted"].sum()),
            "difficulty_median": (
                round(float(g["difficulty_normalized"].median()), 4)
                if g["difficulty_normalized"].notna().any()
                else None
            ),
        }

    return {
        "n_problems": int(len(df)),
        "platforms": per_platform,
        "n_canonical_topics": len(canonical),
        "n_other_labels": len(other),
        "canonical_topics": canonical,
        "with_difficulty": int(df["difficulty_normalized"].notna().sum()),
        "with_canonical_topic": int(df["has_canonical_topic"].sum()),
        "predicted_tags_total": int(df["tags_predicted"].sum()),
    }


def tagger_scores():
    """Per-tag F1, and the macro average, from the tagger's cross-validated run."""
    with open(DATA / "tagger_evaluation.json") as f:
        report = json.load(f)
    scores = {
        tag: v["f1-score"]
        for tag, v in report.items()
        if isinstance(v, dict) and "f1-score" in v and not tag.endswith("avg")
    }
    macro = report["macro avg"]["f1-score"]
    return dict(sorted(scores.items(), key=lambda kv: -kv[1])), macro


def f1_table(scores, n=6):
    best = list(scores.items())[:n]
    worst = list(scores.items())[-n:]
    rows = [f"| {t} | {f:.2f} |" for t, f in best]
    rows.append("| … | |")
    rows += [f"| {t} | {f:.2f} |" for t, f in worst]
    return "\n".join(rows)


def card(s, scores, macro, repo_id):
    plat = s["platforms"]
    return f"""---
license: cc-by-4.0
language:
  - en
tags:
  - competitive-programming
  - education
  - recommender-systems
  - codeforces
  - atcoder
  - leetcode
task_categories:
  - tabular-classification
  - tabular-regression
pretty_name: CPRS Cross-Platform Competitive Programming Catalogue
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: cprs_problems.parquet
---

# CPRS — Cross-Platform Competitive Programming Problem Catalogue

{s['n_problems']:,} competitive-programming problems from **Codeforces**, **AtCoder**
and **LeetCode**, normalised onto a single difficulty scale and a single topic
taxonomy so that problems from different platforms can be compared directly.

Built for [CPRS](https://github.com/writesz/comp-recommendation), a cross-platform
problem recommender (CM3070 final project, University of London).

| Platform | Problems | With difficulty | With a topic | Topics from the NLP tagger |
|---|---:|---:|---:|---:|
| Codeforces | {plat['codeforces']['n']:,} | {plat['codeforces']['with_difficulty']:,} | {plat['codeforces']['with_canonical_topic']:,} | {plat['codeforces']['predicted_tags']:,} |
| AtCoder | {plat['atcoder']['n']:,} | {plat['atcoder']['with_difficulty']:,} | {plat['atcoder']['with_canonical_topic']:,} | {plat['atcoder']['predicted_tags']:,} |
| LeetCode | {plat['leetcode']['n']:,} | {plat['leetcode']['with_difficulty']:,} | {plat['leetcode']['with_canonical_topic']:,} | {plat['leetcode']['predicted_tags']:,} |
| **Total** | **{s['n_problems']:,}** | **{s['with_difficulty']:,}** | **{s['with_canonical_topic']:,}** | **{s['predicted_tags_total']:,}** |

## Why this exists

The three largest competitive-programming platforms describe difficulty in three
incompatible ways and tag topics in three different vocabularies — and AtCoder
publishes no topic tags at all. That makes it impossible to ask a question as
basic as "what should this learner attempt next?" across platforms. This dataset
is the normalisation layer: one `[0,1]` difficulty scale, one taxonomy of
{s['n_canonical_topics']} topics, and recovered topics where a platform
published none.

## Columns

| Column | Type | Notes |
|---|---|---|
| `cprs_id` | string | Stable unified key, `{{platform}}:{{platform_id}}` |
| `platform` | string | `codeforces` / `atcoder` / `leetcode` |
| `platform_id` | string | Native identifier |
| `name` | string | Problem title |
| `url` | string | Canonical link to the problem on its platform |
| `contest_id` | string | Native contest identifier, where applicable |
| `difficulty_raw` | float | Native difficulty, **not comparable across platforms** |
| `difficulty_source` | string | Which native scale `difficulty_raw` is on |
| `difficulty_normalized` | float | `[0,1]`, comparable across platforms |
| `tags_original` | list[string] | Platform's own tags, verbatim (empty for AtCoder) |
| `tags_unified` | list[string] | Mapped to the shared taxonomy |
| `tags_predicted` | bool | **True where topics came from the NLP tagger, not a human** |
| `has_canonical_topic` | bool | Has ≥1 real topic (not just an `other:` label) |
| `solve_count` | int64 | Accepted solutions, where the platform reports it |
| `acceptance_rate` | float | Where the platform reports it |
| `is_premium` | bool | LeetCode paywalled problems |

### Difficulty normalisation

| Platform | Native scale | Mapping to `[0,1]` |
|---|---|---|
| Codeforces | Problemsetter rating, 800–3500 | Clipped, then linear |
| AtCoder | [kenkoooo](https://kenkoooo.com/atcoder/) Elo-style estimate, ≈ −500–4000 | Clipped, then linear |
| LeetCode | Easy / Medium / Hard | 0.2 / 0.5 / 0.85 |

The LeetCode mapping is the weak link: three bands cannot capture within-band
spread, so LeetCode difficulty is coarser than the other two. Treat
`difficulty_normalized` as ordinal-comparable, not interval-comparable.

### The taxonomy

{s['n_canonical_topics']} canonical topics, plus {s['n_other_labels']} unmapped
platform labels carried through with an `other:` prefix (contest-series names
such as `other:abc`, platform-specific oddities). **`other:` labels are not
topics** — filter with `has_canonical_topic` rather than checking whether
`tags_unified` is non-empty.

<details>
<summary>The {s['n_canonical_topics']} canonical topics</summary>

{', '.join('`' + t + '`' for t in s['canonical_topics'])}

</details>

## ⚠️ Predicted tags: read this before using `tags_unified`

AtCoder publishes no topic tags. {plat['atcoder']['predicted_tags']:,} AtCoder
problems here carry tags from a TF-IDF + one-vs-rest logistic-regression
classifier trained on tagged Codeforces statements and transferred across
platforms. **These are model output, not ground truth**, and their quality
varies enormously by tag:

| Tag | Cross-validated F1 |
|---|---:|
{f1_table(scores)}

Macro-F1 across all {len(scores)} predicted tags is **{macro:.2f}**. Tags with a
strong lexical signature (`strings`, `math`) transfer well; tags that describe a
*solution technique* invisible in the problem text (`dynamic_programming`,
`two_pointers`) are near-useless — a statement rarely says it wants a DP.

Only tags above a confidence threshold were written, which is why
{plat['atcoder']['predicted_tags']:,} of {plat['atcoder']['n']:,} AtCoder
problems are tagged rather than all of them. **Filter on `tags_predicted` if you
need human-assigned tags only.**

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")

# Human-tagged problems in a difficulty band, across all three platforms
band = ds.filter(
    lambda r: not r["tags_predicted"]
    and r["difficulty_normalized"] is not None
    and 0.3 <= r["difficulty_normalized"] <= 0.5
)
```

## Provenance and terms

All metadata was collected from **public endpoints only**, with a fixed polite
request delay, exponential-backoff retries and resumable caching. No
authenticated or private data was accessed.

| Source | What it feeds |
|---|---|
| [Codeforces API](https://codeforces.com/apiHelp) | Codeforces catalogue, problemsetter ratings, tags |
| [kenkoooo AtCoder Problems](https://kenkoooo.com/atcoder/) | AtCoder catalogue and its difficulty estimates |
| LeetCode public problem listing | LeetCode catalogue and Easy/Medium/Hard bands |
| [atcoder.jp](https://atcoder.jp/) user contest history | **Nothing in this dataset** — contest ratings for the parent project's skill model |

Three platforms, four sources: AtCoder takes two, and is the one platform whose
metadata here is not first-party. It publishes neither topic tags nor an
official difficulty, so difficulty comes from the community-run kenkoooo
estimate and topics from the NLP tagger described above — which is why AtCoder
is also the least completely covered platform in the table at the top.

**Problem statements are not included in this dataset.** They remain the
copyright of the respective platforms; the tagger was trained on them locally
and only its predictions are published here.

**No user data is included.** The recommender in the parent project is trained
on solve histories from real accounts; those are not released, in keeping with
the project's commitment to use that cohort only in aggregate.

The `cc-by-4.0` licence applies to **this compilation and its derived columns**
— the normalised difficulty, the unified taxonomy and the predicted tags. Facts
about the underlying problems (titles, identifiers, URLs, solve counts) are the
platforms' own, and the AtCoder difficulty estimates are the kenkoooo project's.
This dataset is not affiliated with or endorsed by Codeforces, AtCoder, LeetCode
or the kenkoooo AtCoder Problems project.

## Known limitations

- **Snapshot, not a feed.** Collected mid-2025; neither problems added since nor
  drifting solve counts are reflected.
- **Coverage is uneven.** {plat['atcoder']['with_canonical_topic']:,} of
  {plat['atcoder']['n']:,} AtCoder problems have any topic at all, and only
  {plat['atcoder']['with_difficulty']:,} have a difficulty estimate.
- **Tag mapping is lossy.** Collapsing three vocabularies into
  {s['n_canonical_topics']} topics merges distinctions some platforms make.
- **LeetCode difficulty is three-valued**, as noted above.
- **`solve_count` semantics differ** between platforms and are not directly
  comparable; use it for within-platform popularity only.

## Citation

```bibtex
@misc{{cprs2025,
  title  = {{CPRS: A Cross-Platform Competitive Programming Problem Catalogue}},
  author = {{Diyas, Zhannur}},
  year   = {{2025}},
  note   = {{CM3070 Final Project, University of London}},
  howpublished = {{Hugging Face Datasets}}
}}
```
"""


def build_release(repo_id=DEFAULT_REPO_ID):
    """Write the release directory. Returns the table and its statistics."""
    df = build_table()
    s = stats(df)
    scores, macro = tagger_scores()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    df.to_parquet(OUT / "cprs_problems.parquet", index=False)
    (OUT / "README.md").write_text(card(s, scores, macro, repo_id))
    with open(OUT / "release_stats.json", "w") as f:
        json.dump(s, f, indent=2)
    return df, s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo the card's usage example should name.",
    )
    args = ap.parse_args()

    df, s = build_release(args.repo_id)

    size = (OUT / "cprs_problems.parquet").stat().st_size
    print(f"wrote {OUT}")
    print(f"  cprs_problems.parquet  {len(df):,} rows, {size/1e6:.1f} MB")
    print(f"  README.md              dataset card")
    print(f"  release_stats.json     {s['n_canonical_topics']} topics, "
          f"{s['predicted_tags_total']:,} rows with predicted tags")


if __name__ == "__main__":
    main()
