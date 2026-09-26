"""
Probe candidate fourth platforms for CPRS extension feasibility.

For each candidate judge we test the three things the CPRS pipeline actually
needs, in order of how binding they are:

1. CATALOGUE  — can we enumerate all problems with metadata (id, name, tags)?
2. DIFFICULTY — does the platform publish a per-problem difficulty/rating?
3. HISTORY    — can we read a *named user's* accepted-solve history publicly?

(3) is the binding constraint: without public per-user solve histories a
platform can only join the content catalogue, never the interaction matrix,
so it cannot participate in collaborative filtering or the cross-platform
skill-transfer analysis.

Usage:
    python -m scripts.probe_new_platforms
    python -m scripts.probe_new_platforms --handle <name>
"""
import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

console = Console()

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
TIMEOUT = 20


@dataclass
class Probe:
    platform: str
    layer: str          # catalogue | difficulty | history
    url: str
    method: str = "GET"
    kind: str = "json"  # json | html
    payload: Optional[dict] = None
    note: str = ""
    # filled in by run()
    status: str = ""
    detail: str = ""


def run(p: Probe) -> Probe:
    try:
        time.sleep(0.4)
        if p.method == "POST":
            r = requests.post(p.url, json=p.payload, headers=UA, timeout=TIMEOUT)
        else:
            r = requests.get(p.url, headers=UA, timeout=TIMEOUT)
        code = r.status_code
        body = r.text
        if code != 200:
            p.status = f"HTTP {code}"
            p.detail = body[:120].replace("\n", " ")
            return p
        if p.kind == "json":
            try:
                data = r.json()
            except Exception:
                p.status = "NOT JSON"
                p.detail = body[:120].replace("\n", " ")
                return p
            p.status = "OK"
            p.detail = _describe(data)
        else:
            p.status = "OK"
            p.detail = f"{len(body)} bytes html"
    except Exception as e:
        p.status = "ERROR"
        p.detail = f"{type(e).__name__}: {e}"[:140]
    return p


def _describe(data) -> str:
    if isinstance(data, list):
        head = data[0] if data else None
        keys = list(head.keys())[:8] if isinstance(head, dict) else type(head).__name__
        return f"list[{len(data)}] keys={keys}"
    if isinstance(data, dict):
        ks = list(data.keys())[:10]
        # dig one level for the common {data: {...}} envelope
        extra = ""
        for k in ("data", "objects", "result", "results"):
            v = data.get(k)
            if isinstance(v, list) and v:
                inner = list(v[0].keys())[:8] if isinstance(v[0], dict) else type(v[0]).__name__
                extra = f" | {k}=list[{len(v)}] keys={inner}"
                break
            if isinstance(v, dict) and v:
                extra = f" | {k} keys={list(v.keys())[:8]}"
                break
        return f"dict keys={ks}{extra}"
    return type(data).__name__


def build(handle: str, cc_handle: str) -> list[Probe]:
    return [
        # --- CodeChef -----------------------------------------------------
        Probe("codechef", "catalogue", "https://api.codechef.com/problems/school",
              note="old official API (OAuth-gated)"),
        Probe("codechef", "catalogue",
              "https://www.codechef.com/api/list/problems?sort_by=difficulty_rating"
              "&sort_order=asc&search=&limit=20&page=0",
              note="site-internal JSON behind the problems browser"),
        Probe("codechef", "difficulty",
              "https://www.codechef.com/api/contests/PRACTICE/problems/FLOW001",
              note="per-problem detail: difficulty_rating + tags"),
        Probe("codechef", "history", f"https://www.codechef.com/users/{cc_handle}",
              kind="html", note="profile page: rating graph + solved list in HTML"),
        Probe("codechef", "history",
              f"https://www.codechef.com/recent/user?user_handle={cc_handle}&page=0",
              note="recent-submissions widget (paginated, site-internal)"),

        # --- DMOJ ---------------------------------------------------------
        Probe("dmoj", "catalogue", "https://dmoj.ca/api/v2/problems?page=1",
              note="documented public REST API v2"),
        Probe("dmoj", "history", f"https://dmoj.ca/api/v2/submissions?user={handle}",
              note="submissions list may require a token"),
        Probe("dmoj", "history", "https://dmoj.ca/api/v2/user/info/kefaa",
              note="user profile incl. solved problems"),

        # --- Codewars -----------------------------------------------------
        Probe("codewars", "catalogue",
              "https://www.codewars.com/api/v1/code-challenges/multiply",
              note="per-kata only; no list-all endpoint"),
        Probe("codewars", "history",
              "https://www.codewars.com/api/v1/users/g964/code-challenges/completed?page=0",
              note="public completed-kata list"),

        # --- Aizu Online Judge (CodeNet's other half) ----------------------
        Probe("aizu", "catalogue", "https://judgeapi.u-aizu.ac.jp/problems",
              note="documented public API"),
        Probe("aizu", "history",
              f"https://judgeapi.u-aizu.ac.jp/submission_records/users/{handle}?page=0&size=20",
              note="public per-user submissions"),

        # --- Luogu --------------------------------------------------------
        Probe("luogu", "catalogue",
              "https://www.luogu.com.cn/problem/list?page=1&_contentOnly=1",
              note="site-internal JSON; difficulty 0-7 + tags"),

        # --- Kattis -------------------------------------------------------
        Probe("kattis", "catalogue", "https://open.kattis.com/problems?page=0",
              kind="html", note="no API; HTML table carries 1.0-10.0 difficulty"),

        # --- CSES ---------------------------------------------------------
        Probe("cses", "catalogue", "https://cses.fi/problemset/", kind="html",
              note="no API, no difficulty, no tags"),

        # --- HackerRank ----------------------------------------------------
        Probe("hackerrank", "catalogue",
              "https://www.hackerrank.com/rest/contests/master/tracks/algorithms/"
              "challenges?offset=0&limit=20",
              note="site-internal REST"),
        Probe("hackerrank", "history",
              f"https://www.hackerrank.com/rest/hackers/{handle}/recent_challenges?limit=20",
              note="public profile activity"),
    ]


def main(handle: str, cc_handle: str) -> None:
    probes = [run(p) for p in build(handle, cc_handle)]

    t = Table(title="Candidate platform probe", show_lines=False)
    t.add_column("platform"); t.add_column("layer"); t.add_column("status")
    t.add_column("detail", overflow="fold"); t.add_column("note", overflow="fold")
    for p in probes:
        colour = "green" if p.status == "OK" else "red"
        t.add_row(p.platform, p.layer, f"[{colour}]{p.status}[/{colour}]", p.detail, p.note)
    console.print(t)

    print()
    print(json.dumps([{
        "platform": p.platform, "layer": p.layer, "url": p.url,
        "status": p.status, "detail": p.detail,
    } for p in probes], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--handle", default="tourist")
    ap.add_argument("--cc-handle", default="gennady.korotkevich")
    args = ap.parse_args()
    main(args.handle, args.cc_handle)
