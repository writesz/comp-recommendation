"""
Sixth-stage probe: the REAL cost of a CodeChef cross-platform cohort.

probe_codechef_history_cost.py extrapolated from a handful of named accounts
and produced a >100h estimate — but it sampled elite competitors (tourist:
362 pages) alongside a dormant account (2 pages). The cost of a cohort fetch
is set by the *distribution* of page counts over handles that actually match,
not by its tail, so this script measures that distribution directly.

For a random sample of the CF handle list it:
  1. probes CodeChef for a matching account,
  2. for each match, reads page 0 of the submissions widget to get max_page,
  3. reports the page-count distribution and the implied cohort fetch hours.

Usage:
    python3 -m scripts.probe_codechef_cohort_cost --sample 80
"""
import argparse
import random
import statistics
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

console = Console()
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HANDLES = DATA_DIR / "interactions" / "cf_fetched_handles.txt"
COHORT_N = 2459  # users in the CPRS interaction matrix


def _get(url: str, delay: float, max_retries: int = 4):
    backoff = delay
    for _ in range(max_retries):
        time.sleep(backoff)
        try:
            r = requests.get(url, headers=UA, timeout=20, allow_redirects=False)
        except Exception:
            return None
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            backoff *= 2
            continue
        return r
    return None


def matched_handles(sample: int, seed: int, delay: float) -> list[str]:
    handles = [h.strip() for h in HANDLES.read_text().splitlines() if h.strip()]
    random.seed(seed)
    picked = random.sample(handles, min(sample, len(handles)))
    console.rule(f"[bold]Probing {len(picked)} CF handles on CodeChef")

    found = []
    for i, h in enumerate(picked, 1):
        r = _get(f"https://www.codechef.com/users/{h}", delay)
        if r is not None and r.status_code == 200 and "all_rating" in r.text:
            found.append(h)
        if i % 20 == 0:
            console.print(f"  {i}/{len(picked)} probed, {len(found)} matched")
    console.print(f"[bold]matched {len(found)}/{len(picked)} = "
                  f"{len(found) / len(picked):.1%}[/bold]  {found}")
    return found, len(picked)


def page_counts(handles: list[str], delay: float) -> list[tuple[str, int]]:
    console.rule("[bold]Submission-history page counts for matched handles")
    out = []
    for h in handles:
        r = _get(f"https://www.codechef.com/recent/user?user_handle={h}&page=0", delay)
        if r is None:
            console.print(f"  {h}: [red]blocked")
            continue
        try:
            mp = r.json().get("max_page", 0)
        except Exception:
            console.print(f"  {h}: [red]unparsed")
            continue
        out.append((h, mp + 1))
        console.print(f"  {h}: {mp + 1} pages")
    return out


def report(pages: list[int], match_rate: float, delay: float) -> None:
    console.rule("[bold]Implied cohort fetch cost")
    if not pages:
        console.print("[red]no data")
        return
    pages_sorted = sorted(pages)
    total = sum(pages_sorted)
    mean = statistics.mean(pages_sorted)
    median = statistics.median(pages_sorted)

    t = Table(title="pages per matched user")
    t.add_column("stat"); t.add_column("value")
    for k, v in (("n", len(pages_sorted)), ("min", pages_sorted[0]),
                 ("median", median), ("mean", f"{mean:.1f}"),
                 ("max", pages_sorted[-1]), ("total", total)):
        t.add_row(k, str(v))
    console.print(t)

    projected_users = round(match_rate * COHORT_N)
    est_pages = mean * projected_users
    console.print(f"\nprojected matched cohort: {projected_users} users "
                  f"({match_rate:.1%} of {COHORT_N})")
    console.print(f"projected total requests: {est_pages:,.0f} "
                  f"(mean {mean:.1f} pages/user)")
    for d in (delay, 1.5, 1.0):
        console.print(f"  at {d:.1f}s/request -> {est_pages * d / 3600:.1f} hours")
    console.print("\n[dim]The mean is what sets the bill, but the tail sets the "
                  "wall-clock: a handful of heavy users dominate. Capping pages per "
                  "user trades history completeness for a bounded fetch.[/dim]")
    for cap in (20, 50, 100):
        capped = sum(min(p, cap) for p in pages_sorted) / len(pages_sorted)
        kept = sum(min(p, cap) for p in pages_sorted) / total
        console.print(f"  cap {cap:>3} pages/user -> mean {capped:5.1f} pages, "
                      f"{kept:.0%} of all submission pages retained, "
                      f"{capped * projected_users * delay / 3600:.1f} h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delay", type=float, default=3.0)
    args = ap.parse_args()
    found, n = matched_handles(args.sample, args.seed, args.delay)
    counts = page_counts(found, args.delay)
    report([c for _, c in counts], len(found) / n, args.delay)
