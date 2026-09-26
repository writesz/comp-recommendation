"""
Third-stage probe: can a fourth platform join the *interaction matrix*?

The CPRS cross-platform cohort is built by handle-matching: each of the 2,459
Codeforces handles is probed on the second platform, and those that exist with
accepted solves form the bridge. AtCoder yielded 382 such users.

This script measures the same quantity for CodeChef and DMOJ on a random
sample of the CF handle list, and separately measures catalogue economics
(difficulty coverage, request cost). Those two numbers decide whether a
platform can only extend the content catalogue or can also extend the
collaborative/skill-transfer analysis.

Usage:
    python3 -m scripts.probe_platform_overlap --sample 60
"""
import argparse
import random
import time
from collections import Counter
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

console = Console()
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HANDLES = DATA_DIR / "interactions" / "cf_fetched_handles.txt"


def load_handles() -> list[str]:
    return [h.strip() for h in HANDLES.read_text().splitlines() if h.strip()]


def probe_codechef_handle(h: str, delay: float, max_retries: int = 4) -> str:
    """exists / missing / blocked.

    CodeChef rate-limits profile pages aggressively: a naive 0.4s loop is
    throttled to HTTP 429 after roughly a dozen requests, so a usable probe
    needs a slow base delay plus exponential backoff. The cost of that
    throttling is itself a finding — it is what makes cohort construction on
    CodeChef an order of magnitude slower than on AtCoder.
    """
    backoff = delay
    for _ in range(max_retries):
        try:
            time.sleep(backoff)
            r = requests.get(f"https://www.codechef.com/users/{h}", headers=UA,
                             timeout=20, allow_redirects=False)
            if r.status_code == 200:
                # a real profile embeds the rating array; a stub page does not
                return "exists" if "all_rating" in r.text else "exists_norating"
            if r.status_code in (301, 302, 404):
                return "missing"
            if r.status_code == 429:
                backoff *= 2
                continue
            return f"http_{r.status_code}"
        except Exception as e:
            return f"err_{type(e).__name__}"
    return "http_429"


def probe_dmoj_handle(h: str) -> str:
    try:
        time.sleep(0.3)
        r = requests.get(f"https://dmoj.ca/api/v2/user/{h}", headers=UA, timeout=20)
        if r.status_code == 200:
            return "exists"
        if r.status_code == 404:
            return "missing"
        return f"http_{r.status_code}"
    except Exception as e:
        return f"err_{type(e).__name__}"


def overlap(sample: int, seed: int, cc_delay: float = 3.0) -> None:
    handles = load_handles()
    random.seed(seed)
    picked = random.sample(handles, min(sample, len(handles)))
    console.rule(f"[bold]Handle overlap — {len(picked)} of {len(handles)} CF handles")

    cc, dm = Counter(), Counter()
    for i, h in enumerate(picked, 1):
        cc[probe_codechef_handle(h, cc_delay)] += 1
        dm[probe_dmoj_handle(h)] += 1
        if i % 10 == 0:
            console.print(f"  {i}/{len(picked)}  codechef={dict(cc)}  dmoj={dict(dm)}")

    t = Table(title="Handle-match rate against the CF cohort")
    t.add_column("platform"); t.add_column("outcome"); t.add_column("n"); t.add_column("rate")
    n = len(picked)
    for name, ctr in (("codechef", cc), ("dmoj", dm)):
        for k, v in ctr.most_common():
            t.add_row(name, k, str(v), f"{v / n:.1%}")
    console.print(t)

    hit_cc = cc["exists"] + cc["exists_norating"]
    console.print(f"\n[bold]CodeChef[/bold]: {hit_cc}/{n} = {hit_cc / n:.1%} "
                  f"-> projected cohort over 2,459 handles ≈ {round(hit_cc / n * 2459)}")
    console.print(f"[bold]DMOJ[/bold]:     {dm['exists']}/{n} = {dm['exists'] / n:.1%} "
                  f"-> projected cohort ≈ {round(dm['exists'] / n * 2459)}")
    console.print("(AtCoder, for comparison, yielded 382 cross-platform users "
                  "of which 216 had >=10 solves.)")


def catalogue_economics(pages: int) -> None:
    console.rule("[bold]CodeChef catalogue economics")
    unrated = rated = 0
    ratings = []
    for page in range(pages):
        time.sleep(0.5)
        url = ("https://www.codechef.com/api/list/problems?sort_by=difficulty_rating"
               f"&sort_order=desc&search=&limit=100&page={page}")
        rows = requests.get(url, headers=UA, timeout=25).json().get("data", [])
        for r in rows:
            try:
                v = int(r.get("difficulty_rating", -1))
            except (TypeError, ValueError):
                v = -1
            if v > 0:
                rated += 1
                ratings.append(v)
            else:
                unrated += 1
    total = rated + unrated
    console.print(f"sampled {total} problems over {pages} pages "
                  f"(catalogue is 21,568 => {21568 // 100 + 1} pages, "
                  f"~{(21568 // 100 + 1) * 0.5 / 60:.0f} min to enumerate)")
    if total:
        console.print(f"difficulty_rating present: {rated}/{total} = {rated / total:.1%}")
    if ratings:
        ratings.sort()
        console.print(f"rating range on sample: min={ratings[0]} "
                      f"median={ratings[len(ratings) // 2]} max={ratings[-1]}")
    console.print("\nTags require ONE request per problem "
                  "(/api/contests/PRACTICE/problems/<CODE>):")
    console.print(f"  21,568 problems x 0.5s ≈ {21568 * 0.5 / 3600:.1f} hours")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--cc-delay", type=float, default=3.0)
    args = ap.parse_args()
    catalogue_economics(args.pages)
    overlap(args.sample, args.seed, args.cc_delay)
