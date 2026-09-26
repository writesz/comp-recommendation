"""
Enrich the CodeChef catalogue with tags and problem statements.

CodeChef's catalogue endpoint returns difficulty and solve counts but no tags
and no statement text. Both live on the per-problem detail endpoint, at one
request per problem — so this is the expensive pass, and it is deliberately
separate from scripts/build_dataset.py.

The pass is worth its cost twice over, because that single request yields:

  * `computed_tags` / `user_tags` — topic labels, which go through the strict
    taxonomy mapper (CodeChef's crowd-sourced tags include setter usernames);
  * `problemComponents.statement` — the full problem text as structured JSON.

That second point is what makes CodeChef valuable to the NLP auto-tagger.
AtCoder statements had to be scraped from HTML pages protected against
automated access, which held corpus coverage to 16% and capped catalogue-level
tag coverage at 8.5%. CodeChef serves statements through the same JSON API as
everything else, so the tagger can reach effectively the whole catalogue —
addressing the corpus-access bottleneck rather than the modelling one.

Writes JSON Lines incrementally and skips codes already fetched, so a long
harvest is resumable and interruptible.

Usage:
    python3 -m scripts.enrich_codechef                 # whole catalogue
    python3 -m scripts.enrich_codechef --rated-only    # only rated problems
    python3 -m scripts.enrich_codechef --limit 500     # bounded trial
"""
import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetchers import codechef
from fetchers.codechef import RateLimited
from models.unified_schema import CC_DIFFICULTY_SENTINELS

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CATALOGUE = RAW_DIR / "cc_problems.json"
EVENTS = RAW_DIR / "cc_details.jsonl"
COMPILED = RAW_DIR / "cc_details.json"


def _rated(problem: dict) -> bool:
    try:
        return int(problem.get("difficulty_rating", -1)) not in CC_DIFFICULTY_SENTINELS
    except (TypeError, ValueError):
        return False


def load_done() -> set:
    if not EVENTS.exists():
        return set()
    done = set()
    with open(EVENTS) as f:
        for line in f:
            try:
                done.add(json.loads(line)["code"])
            except (ValueError, KeyError):
                continue
    return done


def compile_details() -> int:
    """Fold the JSONL event log into the dict build_dataset.py consumes."""
    details = {}
    with open(EVENTS) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            details[rec["code"]] = {
                "computed_tags": rec.get("computed_tags", []),
                "user_tags": rec.get("user_tags", []),
            }
    COMPILED.write_text(json.dumps(details, indent=2))
    return len(details)


def main(limit, rated_only: bool, delay: float) -> None:
    if not CATALOGUE.exists():
        raise SystemExit(f"{CATALOGUE} missing — run scripts/build_dataset.py first")

    problems = json.loads(CATALOGUE.read_text())
    if rated_only:
        problems = [p for p in problems if _rated(p)]
        logger.info(f"Restricted to {len(problems)} rated problems")

    done = load_done()
    todo = [p for p in problems if p.get("code") and p["code"] not in done]
    if limit:
        todo = todo[:limit]
    logger.info(f"{len(done)} already fetched; {len(todo)} to go")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    statements_dir = DATA_DIR / "statements"
    statements_dir.mkdir(parents=True, exist_ok=True)
    stmt_path = statements_dir / "cc_statements.jsonl"

    n_ok = n_tagged = n_stmt = 0
    with open(EVENTS, "a") as ev, open(stmt_path, "a") as st:
        for p in tqdm(todo, desc="CodeChef detail"):
            code = p["code"]
            try:
                detail = codechef.fetch_problem_detail(code, delay=delay)
                computed = detail.get("computed_tags") or []
                user = detail.get("user_tags") or []
                statement = codechef.extract_statement(detail)
            except RateLimited:
                logger.error(f"rate limited on {code}; stopping so the run stays resumable")
                break
            except Exception as e:
                # Neither a missing problem nor an unparseable statement should
                # abort a multi-hour pass. The code is recorded as attempted so
                # a resumed run does not retry it forever.
                logger.warning(f"{code}: {type(e).__name__}: {str(e)[:120]}")
                ev.write(json.dumps({"code": code, "error": str(e)[:200]}) + "\n")
                ev.flush()
                continue

            ev.write(json.dumps({
                "code": code,
                "computed_tags": computed,
                "user_tags": user,
                "has_statement": bool(statement),
            }) + "\n")
            ev.flush()

            if statement:
                st.write(json.dumps({"code": code, "statement": statement}) + "\n")
                st.flush()
                n_stmt += 1
            n_ok += 1
            n_tagged += bool(computed or user)

    n_total = compile_details()
    console.print(f"\n[bold]fetched[/bold] {n_ok} this run "
                  f"({n_tagged} with tags, {n_stmt} with statements)")
    console.print(f"[bold]cc_details.json[/bold]: {n_total} problems")
    console.print(f"[bold]cc_statements.jsonl[/bold]: {stmt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch CodeChef tags + statements")
    ap.add_argument("--limit", type=int, default=None, help="max problems this run")
    ap.add_argument("--rated-only", action="store_true",
                    help="only problems with a usable difficulty_rating")
    ap.add_argument("--delay", type=float, default=codechef.REQUEST_DELAY)
    args = ap.parse_args()
    main(args.limit, args.rated_only, args.delay)
