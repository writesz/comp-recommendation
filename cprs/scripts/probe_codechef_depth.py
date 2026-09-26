"""
Second-stage probe of the two platforms that passed the first screen.

probe_new_platforms.py established *reachability*. This script asks the harder
question: does the reachable data actually contain the fields CPRS needs to
put a platform into the unified catalogue and the interaction matrix?

Checks:
  CodeChef  — catalogue size, difficulty coverage, tags, per-user solve history
  DMOJ      — catalogue size, points/difficulty, types (tags), public submissions

Usage:
    python3 -m scripts.probe_codechef_depth
"""
import json
import re
import time

import requests
from rich.console import Console

console = Console()
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
TIMEOUT = 25


def get(url, kind="json"):
    time.sleep(0.5)
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json() if kind == "json" else r.text


def rule(title):
    console.rule(f"[bold]{title}")


def codechef_catalogue():
    rule("CodeChef — catalogue enumeration")
    url = ("https://www.codechef.com/api/list/problems?sort_by=difficulty_rating"
           "&sort_order=asc&search=&limit=100&page=0")
    d = get(url)
    console.print(f"envelope: status={d.get('status')} count={d.get('count')}")
    rows = d.get("data", [])
    console.print(f"page rows: {len(rows)}")
    if rows:
        console.print("sample row:")
        console.print(json.dumps(rows[0], indent=1)[:900])
        rated = [r for r in rows if r.get("difficulty_rating")]
        console.print(f"rows with difficulty_rating on this page: {len(rated)}/{len(rows)}")
    # deep page — does pagination keep working past the first few pages?
    deep = get("https://www.codechef.com/api/list/problems?sort_by=difficulty_rating"
               "&sort_order=asc&search=&limit=100&page=30")
    console.print(f"page=30 rows: {len(deep.get('data', []))} (pagination depth check)")
    return d.get("count")


def codechef_problem_detail():
    rule("CodeChef — per-problem detail (tags?)")
    for code in ("FLOW001", "SUMTRIAN", "CHEFSQUA"):
        try:
            d = get(f"https://www.codechef.com/api/contests/PRACTICE/problems/{code}")
        except Exception as e:
            console.print(f"{code}: {type(e).__name__} {e}")
            continue
        tagish = {k: v for k, v in d.items()
                  if "tag" in k.lower() or "difficult" in k.lower()
                  or k in ("category_name", "problem_author", "editorial_url",
                           "successful_submissions", "user_was_solved")}
        console.print(f"{code}: keys={len(d)} | {json.dumps(tagish)[:400]}")


def codechef_user_history(handle="gennady.korotkevich"):
    rule(f"CodeChef — user solve history for {handle}")
    html = get(f"https://www.codechef.com/users/{handle}", kind="html")
    console.print(f"profile html: {len(html)} bytes")
    # the old profile exposed fully-solved problem codes as /problems/CODE links
    codes = set(re.findall(r'/problems/([A-Z0-9_]{3,})"', html))
    console.print(f"distinct /problems/<CODE> links on profile: {len(codes)}")
    console.print(f"sample: {sorted(codes)[:15]}")
    # rating history is embedded as a JS array
    m = re.search(r"var all_rating\s*=\s*(\[.*?\]);", html, re.S)
    if m:
        try:
            hist = json.loads(m.group(1))
            console.print(f"all_rating entries: {len(hist)} | sample={json.dumps(hist[0])[:220]}")
        except Exception as e:
            console.print(f"all_rating present but unparsed: {e}")
    else:
        console.print("no all_rating array found")

    rule(f"CodeChef — recent-submissions widget for {handle}")
    d = get(f"https://www.codechef.com/recent/user?user_handle={handle}&page=0")
    content = d.get("content", "")
    console.print(f"max_page={d.get('max_page')} content={len(content)} bytes")
    subs = re.findall(r'/status/([A-Z0-9_]+)[^>]*>([^<]*)<', content)
    console.print(f"submission rows parsed: {len(subs)} | sample={subs[:5]}")
    probs = re.findall(r'/problems/([A-Z0-9_]{3,})', content)
    console.print(f"problem codes in page 0: {len(probs)} distinct={len(set(probs))}")
    console.print(f"=> max_page {d.get('max_page')} x ~{len(set(probs))} problems/page "
                  f"is the whole public history for this user")


def dmoj():
    rule("DMOJ — catalogue + public submissions")
    d = get("https://dmoj.ca/api/v2/problems?page=1")["data"]
    console.print(f"total_objects={d['total_objects']} total_pages={d['total_pages']}")
    obj = d["objects"][0]
    console.print("sample problem:")
    console.print(json.dumps(obj, indent=1)[:700])

    s = get("https://dmoj.ca/api/v2/submissions?user=tourist")["data"]
    console.print(f"\nsubmissions for 'tourist': total_objects={s['total_objects']}")
    if s["objects"]:
        console.print(json.dumps(s["objects"][0], indent=1)[:500])
    else:
        console.print("EMPTY -> per-user submissions likely need an API token")

    for u in ("kefaaa", "Xyene"):
        try:
            info = get(f"https://dmoj.ca/api/v2/user/{u}")
            console.print(f"user/{u}: {json.dumps(info)[:300]}")
        except Exception as e:
            console.print(f"user/{u}: {type(e).__name__} {e}")


if __name__ == "__main__":
    for fn in (codechef_catalogue, codechef_problem_detail,
               codechef_user_history, dmoj):
        try:
            fn()
        except Exception as e:
            console.print(f"[red]{fn.__name__} failed: {type(e).__name__}: {e}")
