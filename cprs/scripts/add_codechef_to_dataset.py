"""
Merge the CodeChef catalogue into the existing unified dataset.

This exists instead of simply re-running scripts/build_dataset.py, and the
reason is methodological rather than convenience. build_dataset.py re-fetches
all platforms from live APIs, so re-running it would silently move every
Codeforces, AtCoder and LeetCode count — and those counts are quoted
throughout the report and underpin the evaluation, which was run against this
exact snapshot. Adding a platform must not perturb the three already measured.

So this script treats the existing dataset as immutable: it loads
cprs_unified_tagged.json, drops any CodeChef rows already present (making the
merge idempotent and re-runnable as enrichment progresses), converts the
CodeChef catalogue, appends it, and writes the result back out in both JSON
and CSV form. Rows for the other three platforms are passed through untouched.

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


def main(dry_run: bool) -> None:
    if not UNIFIED_JSON.exists():
        raise SystemExit(f"{UNIFIED_JSON} missing")
    if not CC_CATALOGUE.exists():
        raise SystemExit(f"{CC_CATALOGUE} missing — fetch the catalogue first")

    existing = json.loads(UNIFIED_JSON.read_text())
    logger.info(f"Loaded {len(existing):,} problems from the unified dataset")
    summarise(existing, "Before")

    kept = [r for r in existing if r["platform"] != "codechef"]
    if len(kept) != len(existing):
        logger.info(f"Dropped {len(existing) - len(kept):,} existing CodeChef rows "
                    "(merge is idempotent)")

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

    merged = kept + cc_rows
    summarise(merged, "After")

    if dry_run:
        console.print("\n[yellow]dry run — nothing written[/yellow]")
        return

    UNIFIED_JSON.write_text(json.dumps(merged, indent=2))

    df = pd.DataFrame(merged)
    df["tags_original_str"] = df["tags_original"].apply(lambda x: "|".join(x or []))
    df["tags_unified_str"] = df["tags_unified"].apply(lambda x: "|".join(x or []))
    df[CSV_COLS].to_csv(UNIFIED_CSV, index=False)

    logger.success(f"Wrote {len(merged):,} problems to {UNIFIED_JSON.name} "
                   f"and {UNIFIED_CSV.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge CodeChef into the unified dataset")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the merge without writing")
    args = ap.parse_args()
    main(args.dry_run)
