"""
Explore Codeforces API - see what data is available
"""
import requests
import json


def fetch_problems():
    url = "https://codeforces.com/api/problemset.problems"
    resp = requests.get(url)
    data = resp.json()

    problems = data["result"]["problems"]
    stats = data["result"]["problemStatistics"]

    print(f"Total problems: {len(problems)}")
    print()
    print("=== Sample problem ===")
    print(json.dumps(problems[0], indent=2))
    print()
    print("=== Sample stats ===")
    print(json.dumps(stats[0], indent=2))
    print()

    # What fields exist?
    all_keys = set()
    for p in problems:
        all_keys.update(p.keys())
    print("=== All available fields ===")
    for k in sorted(all_keys):
        print(f"  {k}")

    # Tag distribution
    tag_counts = {}
    for p in problems:
        for tag in p.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print()
    print("=== Top 20 tags ===")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag}: {count}")

    # Difficulty distribution
    ratings = [p["rating"] for p in problems if "rating" in p]
    print()
    print(f"=== Difficulty range: {min(ratings)} - {max(ratings)} ===")
    from collections import Counter
    dist = Counter(ratings)
    for r in sorted(dist):
        print(f"  {r}: {dist[r]} problems")


if __name__ == "__main__":
    fetch_problems()
