"""
Scrape problem statements from Codeforces and AtCoder.

CF statements are needed as training data (they have tags).
AtCoder statements are needed for inference (predict tags via NLP).

We use aiohttp for concurrent requests with rate limiting to be
respectful to the platforms.

Usage:
    python -m scripts.scrape_statements --platform cf --limit 2000
    python -m scripts.scrape_statements --platform ac --limit 3000
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_html(html: str) -> str:
    """Extract clean text from problem statement HTML."""
    soup = BeautifulSoup(html, "lxml")
    # Remove script/style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def scrape_cf_problem(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    contest_id: int,
    index: str,
) -> dict:
    """Scrape a single Codeforces problem statement."""
    url = f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return {"contest_id": contest_id, "index": index, "statement": None, "error": f"HTTP {resp.status}"}
                html = await resp.text()
        except Exception as e:
            return {"contest_id": contest_id, "index": index, "statement": None, "error": str(e)}

    soup = BeautifulSoup(html, "lxml")
    # CF problem statement is in div.problem-statement
    stmt_div = soup.find("div", class_="problem-statement")
    if not stmt_div:
        return {"contest_id": contest_id, "index": index, "statement": None, "error": "no statement div"}

    # Remove input/output specification headers but keep the text
    statement_text = clean_html(str(stmt_div))
    return {"contest_id": contest_id, "index": index, "statement": statement_text, "error": None}


async def scrape_ac_problem(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    contest_id: str,
    problem_id: str,
) -> dict:
    """Scrape a single AtCoder problem statement."""
    url = f"https://atcoder.jp/contests/{contest_id}/tasks/{problem_id}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return {"problem_id": problem_id, "statement": None, "error": f"HTTP {resp.status}"}
                html = await resp.text()
        except Exception as e:
            return {"problem_id": problem_id, "statement": None, "error": str(e)}

    soup = BeautifulSoup(html, "lxml")
    # AtCoder statements are in the task-statement div
    # English statements are in span.lang-en if bilingual, otherwise in #task-statement directly
    en_span = soup.find("span", class_="lang-en")
    if en_span:
        statement_text = clean_html(str(en_span))
    else:
        task_div = soup.find("div", id="task-statement")
        if task_div:
            statement_text = clean_html(str(task_div))
        else:
            return {"problem_id": problem_id, "statement": None, "error": "no statement div"}

    return {"problem_id": problem_id, "statement": statement_text, "error": None}


async def scrape_cf_batch(problems: list, concurrency: int = 3, delay: float = 0.4) -> list:
    """
    Scrape CF problem statements with rate limiting.
    concurrency=3, delay=0.4s => ~7.5 req/sec max, well within limits.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for p in problems:
            contest_id = p.get("contestId")
            index = p.get("index", "")
            if not contest_id:
                continue
            tasks.append(scrape_cf_problem(session, semaphore, contest_id, index))

        results = await tqdm_asyncio.gather(*tasks, desc="Scraping CF statements")

    # Add delay between batches isn't needed since semaphore handles concurrency
    return results


async def scrape_ac_batch(problems: list, concurrency: int = 3) -> list:
    """Scrape AtCoder problem statements with rate limiting."""
    semaphore = asyncio.Semaphore(concurrency)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for p in problems:
            problem_id = p.get("id", "")
            contest_id = p.get("contest_id", "")
            if not contest_id or not problem_id:
                continue
            tasks.append(scrape_ac_problem(session, semaphore, contest_id, problem_id))

        results = await tqdm_asyncio.gather(*tasks, desc="Scraping AC statements")

    return results


def load_cf_problems(limit: int) -> list:
    """Load CF problems from raw data, sorted by solve count (most-solved first)."""
    raw_path = DATA_DIR / "raw" / "cf_problems.json"
    with open(raw_path) as f:
        problems = json.load(f)

    # Only keep problems with tags and a contest ID (for URL construction)
    problems = [p for p in problems if p.get("tags") and p.get("contestId")]
    # Sort by solvedCount descending — most-solved problems have clearest statements
    problems.sort(key=lambda p: p.get("solvedCount", 0), reverse=True)
    return problems[:limit]


def load_ac_problems(limit: int) -> list:
    """Load AtCoder problems from raw data."""
    raw_path = DATA_DIR / "raw" / "ac_problems.json"
    with open(raw_path) as f:
        problems = json.load(f)

    # Filter to those with contest_id (needed for URL)
    problems = [p for p in problems if p.get("contest_id")]
    return problems[:limit]


def main():
    parser = argparse.ArgumentParser(description="Scrape problem statements")
    parser.add_argument("--platform", required=True, choices=["cf", "ac", "both"])
    parser.add_argument("--limit", type=int, default=2000, help="Max problems to scrape")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent requests")
    args = parser.parse_args()

    statements_dir = DATA_DIR / "statements"
    statements_dir.mkdir(parents=True, exist_ok=True)

    if args.platform in ("cf", "both"):
        problems = load_cf_problems(args.limit)
        logger.info(f"Scraping {len(problems)} CF problem statements...")
        results = asyncio.run(scrape_cf_batch(problems, concurrency=args.concurrency))

        success = [r for r in results if r.get("statement")]
        failed = [r for r in results if not r.get("statement")]
        logger.info(f"CF: {len(success)} scraped, {len(failed)} failed")

        with open(statements_dir / "cf_statements.json", "w") as f:
            json.dump(results, f, indent=2)

    if args.platform in ("ac", "both"):
        problems = load_ac_problems(args.limit)
        logger.info(f"Scraping {len(problems)} AtCoder problem statements...")
        results = asyncio.run(scrape_ac_batch(problems, concurrency=args.concurrency))

        success = [r for r in results if r.get("statement")]
        failed = [r for r in results if not r.get("statement")]
        logger.info(f"AtCoder: {len(success)} scraped, {len(failed)} failed")

        with open(statements_dir / "ac_statements.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
