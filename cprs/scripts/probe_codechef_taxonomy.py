"""
Fifth-stage probe: how much normalisation work would CodeChef actually need?

CPRS normalises every platform onto two axes: a [0,1] difficulty and a unified
tag taxonomy (models/unified_schema.py). This script measures how far CodeChef
sits from both:

  DIFFICULTY — the real spread of difficulty_rating across the catalogue, and
               how many problems carry a usable value (vs the -1 / 9999
               sentinels the list endpoint emits).
  TAGS       — the computed_tags / user_tags vocabulary, and how much of it
               already has a home in UNIFIED_TAG_MAP.

Usage:
    python3 -m scripts.probe_codechef_taxonomy --problems 40
"""
import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unified_schema import UNIFIED_TAG_MAP

console = Console()
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def get(url: str, delay: float = 0.6):
    backoff = delay
    for _ in range(4):
        time.sleep(backoff)
        r = requests.get(url, headers=UA, timeout=25)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            backoff *= 2
            continue
        r.raise_for_status()
    raise RuntimeError("rate limited")


def difficulty_spread(pages: int) -> list[str]:
    console.rule("[bold]CodeChef difficulty spread")
    codes, vals, sentinel = [], [], Counter()
    step = max(1, 216 // max(pages, 1))
    for i in range(pages):
        page = i * step
        d = get("https://www.codechef.com/api/list/problems?sort_by=difficulty_rating"
                f"&sort_order=asc&search=&limit=100&page={page}")
        for r in d.get("data", []):
            codes.append(r["code"])
            try:
                v = int(r.get("difficulty_rating", -1))
            except (TypeError, ValueError):
                v = -1
            if v <= 0 or v >= 9000:
                sentinel[v] += 1
            else:
                vals.append(v)
    n = len(codes)
    console.print(f"sampled {n} problems across {pages} pages spanning the catalogue")
    console.print(f"usable difficulty: {len(vals)}/{n} = {len(vals) / max(n,1):.1%}")
    console.print(f"sentinel values: {dict(sentinel)}")
    if vals:
        vals.sort()
        q = lambda p: vals[int(p * (len(vals) - 1))]
        console.print(f"min={vals[0]} p10={q(.1)} median={q(.5)} p90={q(.9)} max={vals[-1]}")
        console.print("[dim]CF for comparison spans 800–3500; the CPRS normaliser "
                      "clamps to that window.[/dim]")
    return codes


def tag_vocabulary(codes: list[str], n_problems: int, seed: int) -> None:
    console.rule("[bold]CodeChef tag vocabulary vs UNIFIED_TAG_MAP")
    random.seed(seed)
    picked = random.sample(codes, min(n_problems, len(codes)))
    computed, user = Counter(), Counter()
    fetched = 0
    for code in picked:
        try:
            d = get(f"https://www.codechef.com/api/contests/PRACTICE/problems/{code}")
        except Exception:
            continue
        fetched += 1
        for t in d.get("computed_tags") or []:
            computed[t] += 1
        for t in d.get("user_tags") or []:
            user[t] += 1

    console.print(f"fetched detail for {fetched}/{len(picked)} problems")
    console.print(f"distinct computed_tags: {len(computed)} | distinct user_tags: {len(user)}")

    known = {k.lower() for k in UNIFIED_TAG_MAP}
    t = Table(title="computed_tags (CodeChef's own taxonomy)")
    t.add_column("tag"); t.add_column("n"); t.add_column("already in UNIFIED_TAG_MAP?")
    for tag, c in computed.most_common(30):
        t.add_row(tag, str(c), "yes" if tag.lower() in known else "[red]NO[/red]")
    console.print(t)

    mapped = sum(c for tag, c in computed.items() if tag.lower() in known)
    total = sum(computed.values())
    console.print(f"computed_tag occurrences already mappable: "
                  f"{mapped}/{total} = {mapped / max(total,1):.1%}")
    console.print(f"[bold]=> {len([t for t in computed if t.lower() not in known])} "
                  f"new taxonomy entries needed for the sampled vocabulary[/bold]")

    console.print(f"\nuser_tags sample (crowd-sourced, free text): "
                  f"{[t for t, _ in user.most_common(20)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--problems", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    codes = difficulty_spread(args.pages)
    tag_vocabulary(codes, args.problems, args.seed)
