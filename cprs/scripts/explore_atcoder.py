"""
Explore AtCoder Problems API (kenkoooo) - see what data is available
"""
import requests
import json


BASE = "https://kenkoooo.com/atcoder/resources"


def fetch_problems():
    problems = requests.get(f"{BASE}/problems.json").json()
    models = requests.get(f"{BASE}/problem-models.json").json()

    print(f"Total problems: {len(problems)}")
    print()
    print("=== Sample problem ===")
    print(json.dumps(problems[0], indent=2))
    print()

    all_keys = set()
    for p in problems:
        all_keys.update(p.keys())
    print("=== All available fields ===")
    for k in sorted(all_keys):
        print(f"  {k}")

    print()
    print("=== Sample model (difficulty estimate) ===")
    sample_id = problems[0]["id"]
    if sample_id in models:
        print(json.dumps(models[sample_id], indent=2))
    else:
        # find one that has a model
        for p in problems:
            if p["id"] in models:
                print(f"Problem: {p['id']}")
                print(json.dumps(models[p["id"]], indent=2))
                break

    # Model fields
    model_keys = set()
    for v in models.values():
        model_keys.update(v.keys())
    print()
    print("=== Model fields ===")
    for k in sorted(model_keys):
        print(f"  {k}")

    # How many problems have difficulty estimates?
    with_difficulty = sum(1 for p in problems if p["id"] in models and "difficulty" in models[p["id"]])
    print()
    print(f"Problems with difficulty estimate: {with_difficulty}/{len(problems)}")


if __name__ == "__main__":
    fetch_problems()
