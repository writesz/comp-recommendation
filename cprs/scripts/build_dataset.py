"""
Build the unified CPRS dataset.

Fetches problems from Codeforces, AtCoder, CodeChef and LeetCode, normalizes
them to a common schema, and saves as a single CSV + JSON dataset.

This script produces the cross-platform dataset that is a core contribution
of the CPRS project — no existing dataset combines competitive programming
problems across these four platforms with normalized difficulty and tags.

Note that CodeChef's catalogue endpoint returns difficulty and solve counts
but no tags: those live on the per-problem detail endpoint, one request each.
Tag and statement enrichment is therefore a separate pass
(scripts/enrich_codechef.py), exactly as AtCoder's tags come from the NLP
tagger rather than from its catalogue.

Usage:
    python -m scripts.build_dataset
"""
import json
import sys
from typing import Optional
from pathlib import Path

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Add parent dir to path so we can import fetchers/models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetchers import codeforces, atcoder, codechef, leetcode
from models.unified_schema import (
    Platform,
    UnifiedProblem,
    normalize_cf_difficulty,
    normalize_ac_difficulty,
    normalize_cc_difficulty,
    normalize_lc_difficulty,
    unify_tags,
    unify_tags_strict,
)

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def convert_cf_problems(raw_problems: list) -> list:
    """Convert Codeforces problems to unified schema."""
    unified = []
    for p in tqdm(raw_problems, desc="Converting CF problems"):
        contest_id = p.get("contestId")
        index = p.get("index", "")
        platform_id = f"{contest_id}{index}" if contest_id else p.get("name", "unknown")

        unified.append(UnifiedProblem(
            platform=Platform.CODEFORCES,
            platform_id=platform_id,
            cprs_id=f"cf:{platform_id}",
            url=f"https://codeforces.com/problemset/problem/{contest_id}/{index}",
            name=p.get("name", ""),
            contest_id=str(contest_id) if contest_id else None,
            difficulty_raw=p.get("rating"),
            difficulty_normalized=normalize_cf_difficulty(p.get("rating")),
            tags_original=p.get("tags", []),
            tags_unified=unify_tags(p.get("tags", [])),
            solve_count=p.get("solvedCount"),
        ))
    return unified


def convert_ac_problems(raw_problems: list, models: dict) -> list:
    """Convert AtCoder problems to unified schema."""
    unified = []
    for p in tqdm(raw_problems, desc="Converting AC problems"):
        pid = p["id"]
        contest = p.get("contest_id", "")
        model = models.get(pid, {})
        diff = model.get("difficulty")

        # AtCoder problems don't have tags from the API,
        # but we can infer contest category from the problem index
        # (a=easiest, b, c, ... in typical ABC contests)
        tags_original = []
        if contest.startswith("abc"):
            tags_original.append("abc")
        elif contest.startswith("arc"):
            tags_original.append("arc")
        elif contest.startswith("agc"):
            tags_original.append("agc")

        unified.append(UnifiedProblem(
            platform=Platform.ATCODER,
            platform_id=pid,
            cprs_id=f"ac:{pid}",
            url=f"https://atcoder.jp/contests/{contest}/tasks/{pid}",
            name=p.get("title", ""),
            contest_id=contest,
            difficulty_raw=diff,
            difficulty_normalized=normalize_ac_difficulty(diff),
            tags_original=tags_original,
            tags_unified=unify_tags(tags_original),
            solve_count=model.get("solver_count"),
        ))
    return unified


def convert_cc_problems(raw_problems: list, details: Optional[dict] = None) -> list:
    """
    Convert CodeChef problems to unified schema.

    `details` optionally maps problem code -> detail payload from the
    enrichment pass; when absent the problem still carries difficulty and
    solve statistics, just no tags. Acceptance rate is derived from the
    distinct-solver and total-submission counts the catalogue reports.
    """
    details = details or {}
    unified = []
    for p in tqdm(raw_problems, desc="Converting CC problems"):
        code = p.get("code", "")
        if not code:
            continue
        contest = p.get("contest_code") or "PRACTICE"

        detail = details.get(code, {})
        # computed_tags is CodeChef's own taxonomy; user_tags is crowd-sourced
        # and mixes topics with setter usernames, so both go through the strict
        # mapper that treats the unified taxonomy as a whitelist.
        tags_original = list(detail.get("computed_tags") or []) + \
            list(detail.get("user_tags") or [])

        total = _as_int(p.get("total_submissions"))
        solved = _as_int(p.get("distinct_successful_submissions"))
        acceptance = (solved / total) if total and solved is not None else None

        unified.append(UnifiedProblem(
            platform=Platform.CODECHEF,
            platform_id=code,
            cprs_id=f"cc:{code}",
            url=f"https://www.codechef.com/problems/{code}",
            name=p.get("name", ""),
            contest_id=str(contest),
            difficulty_raw=_as_int(p.get("difficulty_rating")),
            difficulty_normalized=normalize_cc_difficulty(p.get("difficulty_rating")),
            tags_original=tags_original,
            tags_unified=unify_tags_strict(tags_original),
            solve_count=solved,
            acceptance_rate=acceptance,
        ))
    return unified


def _as_int(value) -> Optional[int]:
    """CodeChef returns numeric fields as strings; sentinels become None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def convert_lc_problems(raw_problems: list) -> list:
    """Convert LeetCode problems to unified schema."""
    unified = []
    for p in tqdm(raw_problems, desc="Converting LC problems"):
        qid = p.get("frontendQuestionId", "")
        tags = [t["name"] for t in p.get("topicTags", [])]

        unified.append(UnifiedProblem(
            platform=Platform.LEETCODE,
            platform_id=str(qid),
            cprs_id=f"lc:{qid}",
            url=f"https://leetcode.com/problems/{p.get('titleSlug', '')}/",
            name=p.get("title", ""),
            difficulty_raw={"Easy": 1, "Medium": 2, "Hard": 3}.get(
                p.get("difficulty"), None
            ),
            difficulty_normalized=normalize_lc_difficulty(p.get("difficulty", "")),
            tags_original=tags,
            tags_unified=unify_tags(tags),
            acceptance_rate=p.get("acRate"),
            is_premium=p.get("isPaidOnly", False),
        ))
    return unified


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a rich summary table of the dataset."""
    console.print("\n[bold]CPRS Unified Dataset Summary[/bold]\n")

    # Per-platform counts
    table = Table(title="Problems by Platform")
    table.add_column("Platform", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("With Difficulty", justify="right")
    table.add_column("With Tags", justify="right")

    for platform in ["codeforces", "atcoder", "codechef", "leetcode"]:
        subset = df[df["platform"] == platform]
        with_diff = subset["difficulty_normalized"].notna().sum()
        with_tags = (subset["tags_unified"].apply(len) > 0).sum()
        table.add_row(
            platform,
            str(len(subset)),
            f"{with_diff} ({100*with_diff/len(subset):.0f}%)" if len(subset) > 0 else "0",
            f"{with_tags} ({100*with_tags/len(subset):.0f}%)" if len(subset) > 0 else "0",
        )

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(df)}[/bold]",
        f"[bold]{df['difficulty_normalized'].notna().sum()}[/bold]",
        f"[bold]{(df['tags_unified'].apply(len) > 0).sum()}[/bold]",
    )
    console.print(table)

    # Difficulty distribution
    console.print("\n[bold]Difficulty Distribution (normalized 0-1):[/bold]")
    valid = df["difficulty_normalized"].dropna()
    if len(valid) > 0:
        console.print(f"  Mean: {valid.mean():.3f}")
        console.print(f"  Median: {valid.median():.3f}")
        console.print(f"  Std: {valid.std():.3f}")
        console.print(f"  Range: [{valid.min():.3f}, {valid.max():.3f}]")

    # Top unified tags
    from collections import Counter
    all_tags = []
    for tags in df["tags_unified"]:
        all_tags.extend(tags)
    tag_counts = Counter(all_tags)
    console.print(f"\n[bold]Top 15 Unified Tags (across all platforms):[/bold]")
    for tag, count in tag_counts.most_common(15):
        console.print(f"  {tag}: {count}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Fetch raw data ---
    console.print("[bold cyan]Step 1: Fetching data from all platforms...[/bold cyan]\n")

    cf_raw = codeforces.fetch_all_problems()
    ac_raw = atcoder.fetch_all_problems()
    ac_models = atcoder.fetch_difficulty_models()
    cc_raw = codechef.fetch_all_problems()
    lc_raw = leetcode.fetch_all_problems()

    # Save raw data
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    for name, data in [("cf_problems", cf_raw), ("ac_problems", ac_raw),
                       ("cc_problems", cc_raw), ("lc_problems", lc_raw)]:
        with open(raw_dir / f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)
    with open(raw_dir / "ac_models.json", "w") as f:
        json.dump(ac_models, f, indent=2)
    logger.info(f"Raw data saved to {raw_dir}")

    # --- Convert to unified schema ---
    console.print("\n[bold cyan]Step 2: Normalizing to unified schema...[/bold cyan]\n")

    # CodeChef tags live behind a per-problem endpoint, so reuse the
    # enrichment pass's output when it has been run.
    cc_details = {}
    cc_details_path = raw_dir / "cc_details.json"
    if cc_details_path.exists():
        cc_details = json.loads(cc_details_path.read_text())
        logger.info(f"Loaded CodeChef detail for {len(cc_details)} problems")
    else:
        logger.warning("No cc_details.json — CodeChef problems will carry no tags. "
                       "Run scripts/enrich_codechef.py to populate them.")

    cf_unified = convert_cf_problems(cf_raw)
    ac_unified = convert_ac_problems(ac_raw, ac_models)
    cc_unified = convert_cc_problems(cc_raw, cc_details)
    lc_unified = convert_lc_problems(lc_raw)

    all_problems = cf_unified + ac_unified + cc_unified + lc_unified
    logger.info(f"Total unified problems: {len(all_problems)}")

    # --- Save unified dataset ---
    console.print("\n[bold cyan]Step 3: Saving unified dataset...[/bold cyan]\n")

    # JSON (full fidelity)
    problems_dicts = [p.model_dump() for p in all_problems]
    with open(DATA_DIR / "cprs_unified.json", "w") as f:
        json.dump(problems_dicts, f, indent=2)

    # CSV (for quick analysis in pandas/notebooks)
    df = pd.DataFrame(problems_dicts)
    # Convert list columns to pipe-separated strings for CSV compatibility
    df["tags_original_str"] = df["tags_original"].apply(lambda x: "|".join(x))
    df["tags_unified_str"] = df["tags_unified"].apply(lambda x: "|".join(x))
    csv_cols = [
        "cprs_id", "platform", "platform_id", "name", "url", "contest_id",
        "difficulty_raw", "difficulty_normalized",
        "tags_original_str", "tags_unified_str",
        "solve_count", "acceptance_rate", "is_premium",
    ]
    df[csv_cols].to_csv(DATA_DIR / "cprs_unified.csv", index=False)

    logger.info(f"Saved {len(df)} problems to {DATA_DIR / 'cprs_unified.csv'}")
    logger.info(f"Saved {len(df)} problems to {DATA_DIR / 'cprs_unified.json'}")

    # --- Summary ---
    print_dataset_summary(df)


if __name__ == "__main__":
    main()
