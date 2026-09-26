"""
Fourth-stage probe: what does one CodeChef user's solve history actually cost?

CodeChef removed the "problems solved" block from the profile page, so a user's
accepted-solve set can only be reconstructed by walking the paginated
/recent/user submissions widget. This measures the shape of that walk — pages
per user, distinct solved problems per page, and whether the AC verdict and
problem code are both recoverable — which is what turns "CodeChef is
reachable" into a concrete hours-of-fetching number.

Contrast: AtCoder's kenkoooo endpoint returns 500 submissions per request in
clean JSON, so a whole user costs a handful of calls.

Usage:
    python3 -m scripts.probe_codechef_history_cost --handles a,b,c
"""
import argparse
import re
import time

import requests
from rich.console import Console

console = Console()
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def fetch_page(handle: str, page: int, delay: float) -> dict:
    backoff = delay
    for _ in range(4):
        time.sleep(backoff)
        r = requests.get(
            f"https://www.codechef.com/recent/user?user_handle={handle}&page={page}",
            headers=UA, timeout=25)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            backoff *= 2
            continue
        r.raise_for_status()
    raise RuntimeError("rate limited out")


def analyse(handle: str, delay: float) -> None:
    console.rule(f"[bold]{handle}")
    try:
        d = fetch_page(handle, 0, delay)
    except Exception as e:
        console.print(f"[red]{type(e).__name__}: {e}")
        return
    content = d.get("content", "")
    max_page = d.get("max_page", 0)

    # each row: problem link, verdict icon, language, submission id
    codes = re.findall(r'/problems/([A-Z0-9_]{2,})', content)
    verdicts = re.findall(r'title="([^"]{2,40})"', content)
    rows = content.count("<tr")
    console.print(f"max_page={max_page}  content={len(content)}B  table rows≈{rows}")
    console.print(f"problem codes/page: {len(codes)} (distinct {len(set(codes))}) "
                  f"sample={sorted(set(codes))[:8]}")
    console.print(f"verdict-ish titles: {sorted(set(verdicts))[:8]}")

    ac_like = [v for v in verdicts if "accept" in v.lower() or v.strip() == "AC"]
    console.print(f"accepted-verdict markers on page 0: {len(ac_like)}")

    per_page = max(len(set(codes)), 1)
    total_pages = max_page + 1
    console.print(f"[bold]=> full history ≈ {total_pages} requests; "
                  f"at 3s/request (the rate limit) ≈ {total_pages * 3 / 60:.1f} min "
                  f"for this ONE user[/bold]")
    console.print(f"   projected 541-user cohort ≈ "
                  f"{541 * total_pages * 3 / 3600:.0f} h at this page count")
    console.print(f"   (AtCoder equivalent: ~{max(1, total_pages * per_page // 500)} "
                  f"requests via kenkoooo's 500-per-call JSON)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--handles", default="gennady.korotkevich,uwi,anta")
    ap.add_argument("--delay", type=float, default=3.0)
    args = ap.parse_args()
    for h in args.handles.split(","):
        analyse(h.strip(), args.delay)
