"""
Merge the CodeChef catalogue into the existing unified dataset.

This exists instead of simply re-running scripts/build_dataset.py, and the
reason is methodological rather than convenience. build_dataset.py re-fetches
all platforms from live APIs, so re-running it would silently move every
Codeforces, AtCoder and LeetCode count — and those counts are quoted
throughout the report and underpin the evaluation, which was run against this
exact snapshot. Adding a platform must not perturb the three already measured.

So this script treats the existing rows as immutable: for each dataset file it
drops any CodeChef rows already present (making the merge idempotent and
re-runnable as enrichment progresses), converts the CodeChef catalogue and
appends it, passing the other three platforms through untouched.

It writes two pairs, not one. cprs_unified.{json,csv} is the pre-tagger
snapshot and cprs_unified_tagged.{json,csv} the post-tagger one;
report_stats.py measures what the NLP tagger contributed by differencing them.
CodeChef therefore has to enter both carrying only its own coarse tags, or the
tagger would appear to have supplied labels the catalogue already had.

Usage:
    python3 -m scripts.add_codechef_to_dataset
    python3 -m scripts.add_codechef_to_dataset --dry-run
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_dataset import convert_cc_problems

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# The pre-tagger snapshot and the tagged one are kept as a matched pair:
# report_stats.py measures the tagger's contribution by differencing them, so
# CodeChef has to enter BOTH carrying only its own coarse `computed_tags`.
# scripts/auto_tag_codechef.py then edits the tagged file alone, and the
# difference between the two is exactly what the model added.
UNTAGGED_JSON = DATA_DIR / "cprs_unified.json"
UNTAGGED_CSV = DATA_DIR / "cprs_unified.csv"
UNIFIED_JSON = DATA_DIR / "cprs_unified_tagged.json"
UNIFIED_CSV = DATA_DIR / "cprs_unified_tagged.csv"
CC_CATALOGUE = RAW_DIR / "cc_problems.json"
CC_DETAILS = RAW_DIR / "cc_details.json"

CSV_COLS = [
    "cprs_id", "platform", "platform_id", "name", "url", "contest_id",
    "difficulty_raw", "difficulty_normalized",
    "tags_original_str", "tags_unified_str",
    "solve_count", "acceptance_rate", "is_premium",
]


def summarise(rows: list, title: str) -> None:
    by_platform = Counter(r["platform"] for r in rows)
    t = Table(title=title)
    for col in ("platform", "problems", "with difficulty", "with tags"):
        t.add_column(col, justify="right" if col != "platform" else "left")
    for platform in sorted(by_platform):
        subset = [r for r in rows if r["platform"] == platform]
        n = len(subset)
        diff = sum(1 for r in subset if r.get("difficulty_normalized") is not None)
        tags = sum(1 for r in subset if r.get("tags_unified"))
        t.add_row(platform, f"{n:,}", f"{diff:,} ({diff / n:.0%})",
                  f"{tags:,} ({tags / n:.0%})")
    total = len(rows)
    d = sum(1 for r in rows if r.get("difficulty_normalized") is not None)
    g = sum(1 for r in rows if r.get("tags_unified"))
    t.add_row("[bold]total[/bold]", f"[bold]{total:,}[/bold]",
              f"[bold]{d:,} ({d / total:.0%})[/bold]",
              f"[bold]{g:,} ({g / total:.0%})[/bold]")
    console.print(t)


def write_pair(rows: list, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(rows, indent=2))
    df = pd.DataFrame(rows)
    df["tags_original_str"] = df["tags_original"].apply(lambda x: "|".join(x or []))
    df["tags_unified_str"] = df["tags_unified"].apply(lambda x: "|".join(x or []))
    df[CSV_COLS].to_csv(csv_path, index=False)


def merge_into(path: Path, cc_rows: list) -> list:
    """Replace the CodeChef rows of one dataset file, leaving others untouched."""
    existing = json.loads(path.read_text())
    kept = [r for r in existing if r["platform"] != "codechef"]
    if len(kept) != len(existing):
        logger.info(f"{path.name}: dropped {len(existing) - len(kept):,} existing "
                    "CodeChef rows (merge is idempotent)")
    return kept + cc_rows


def main(dry_run: bool) -> None:
    for required in (UNIFIED_JSON, UNTAGGED_JSON):
        if not required.exists():
            raise SystemExit(f"{required} missing")
    if not CC_CATALOGUE.exists():
        raise SystemExit(f"{CC_CATALOGUE} missing — fetch the catalogue first")

    summarise(json.loads(UNIFIED_JSON.read_text()), "Before")

    details = {}
    if CC_DETAILS.exists():
        details = json.loads(CC_DETAILS.read_text())
        logger.info(f"Loaded CodeChef tag detail for {len(details):,} problems")
    else:
        logger.warning("No cc_details.json — CodeChef rows will carry no tags yet. "
                       "Re-run this script after scripts/enrich_codechef.py.")

    raw = json.loads(CC_CATALOGUE.read_text())
    cc_rows = [p.model_dump() for p in convert_cc_problems(raw, details)]
    logger.info(f"Converted {len(cc_rows):,} CodeChef problems")

    merged = merge_into(UNIFIED_JSON, cc_rows)
    merged_untagged = merge_into(UNTAGGED_JSON, cc_rows)
    summarise(merged, "After")

    if dry_run:
        console.print("\n[yellow]dry run — nothing written[/yellow]")
        return

    write_pair(merged, UNIFIED_JSON, UNIFIED_CSV)
    write_pair(merged_untagged, UNTAGGED_JSON, UNTAGGED_CSV)
    logger.success(f"Wrote {len(merged):,} problems to the tagged pair and "
                   f"{len(merged_untagged):,} to the pre-tagger pair")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge CodeChef into the unified dataset")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the merge without writing")
    args = ap.parse_args()
    main(args.dry_run)
