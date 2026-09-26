"""
Apply the NLP transfer tagger to CodeChef problems.

The tagger trained on Codeforces statements in scripts/auto_tag_atcoder.py is
reused verbatim here — same TF-IDF features, same one-vs-rest logistic
regression, same target taxonomy — and pointed at a second target platform.
That reuse is the point: if a model trained on one judge's statements
transfers to two others, the transfer is a property of the method rather than
a quirk of the AtCoder corpus.

Two things differ from the AtCoder run, and both matter for how the result
should be read.

1. CORPUS ACCESS. AtCoder statements had to be scraped from pages protected
   against automated access, and only 16% were reachable, which capped
   catalogue-level tag coverage at 8.5% however well the model performed.
   CodeChef serves statements as structured JSON from the same endpoint that
   carries its tags, so coverage is bounded by fetch time rather than by
   access. This tests the claim made in the AtCoder limitations — that the
   bottleneck was corpus access and not modelling.

2. GAP-FILLING, NOT OVERWRITING. AtCoder publishes no topic tags at all, so
   there the tagger writes into an empty field. CodeChef publishes coarse
   `computed_tags` for some problems, so predictions are applied only where
   the catalogue yielded no unified tags. A problem that CodeChef itself
   labelled keeps its own label; the tagger fills the silence.

Usage:
    python3 -m scripts.auto_tag_codechef
    python3 -m scripts.auto_tag_codechef --dry-run
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.auto_tag_atcoder import load_cf_training_data, train_and_evaluate

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATEMENTS = DATA_DIR / "statements" / "cc_statements.jsonl"
UNIFIED_JSON = DATA_DIR / "cprs_unified_tagged.json"
UNIFIED_CSV = DATA_DIR / "cprs_unified_tagged.csv"
PREDICTIONS = DATA_DIR / "cc_predicted_tags.json"

CSV_COLS = [
    "cprs_id", "platform", "platform_id", "name", "url", "contest_id",
    "difficulty_raw", "difficulty_normalized",
    "tags_original_str", "tags_unified_str",
    "solve_count", "acceptance_rate", "is_premium",
]


def load_cc_statements() -> tuple:
    """Load CodeChef statements written by scripts/enrich_codechef.py."""
    if not STATEMENTS.exists():
        raise SystemExit(f"{STATEMENTS} missing — run scripts/enrich_codechef.py first")

    codes, texts, seen = [], [], set()
    with open(STATEMENTS) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            code, text = rec.get("code"), rec.get("statement")
            # The harvest is append-only and resumable, so it can contain a
            # code twice if a run was interrupted mid-write.
            if not code or not text or code in seen:
                continue
            seen.add(code)
            codes.append(code)
            texts.append(text)

    logger.info(f"Loaded {len(texts)} CodeChef statements")
    return codes, texts


def predict(vectorizer, classifier, mlb, codes, texts) -> dict:
    console.print("\n[bold cyan]Predicting tags for CodeChef problems...[/bold cyan]\n")
    X = vectorizer.transform(texts)
    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)

    predictions = {}
    for i, code in enumerate(codes):
        tags = list(mlb.inverse_transform(y_pred[i:i + 1])[0])
        predictions[code] = {
            "predicted_tags": tags,
            "tag_scores": {t: float(y_proba[i][j]) for j, t in enumerate(mlb.classes_)},
        }

    tagged = sum(1 for p in predictions.values() if p["predicted_tags"])
    avg = np.mean([len(p["predicted_tags"]) for p in predictions.values()])
    console.print(f"Tagged {tagged}/{len(predictions)} CodeChef statements "
                  f"({100 * tagged / max(len(predictions), 1):.1f}%), "
                  f"averaging {avg:.1f} tags each")

    dist = Counter(t for p in predictions.values() for t in p["predicted_tags"])
    console.print("\n[bold]Predicted tag distribution:[/bold]")
    for tag, n in dist.most_common(15):
        console.print(f"  {tag}: {n}")
    return predictions


def apply_to_dataset(predictions: dict, dry_run: bool) -> None:
    dataset = json.loads(UNIFIED_JSON.read_text())

    filled = kept = missing = 0
    for problem in dataset:
        if problem["platform"] != "codechef":
            continue
        if problem.get("tags_unified"):
            kept += 1            # CodeChef labelled it; leave it alone
            continue
        pred = predictions.get(problem["platform_id"])
        if not pred or not pred["predicted_tags"]:
            missing += 1
            continue
        problem["tags_unified"] = pred["predicted_tags"]
        problem["tags_original"] = [f"predicted:{t}" for t in pred["predicted_tags"]]
        filled += 1

    console.print(f"\n[bold]{filled}[/bold] problems tagged by the model, "
                  f"[bold]{kept}[/bold] kept CodeChef's own tags, "
                  f"[bold]{missing}[/bold] still untagged")

    if dry_run:
        console.print("[yellow]dry run — nothing written[/yellow]")
        return

    UNIFIED_JSON.write_text(json.dumps(dataset, indent=2))
    df = pd.DataFrame(dataset)
    df["tags_original_str"] = df["tags_original"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else "")
    df["tags_unified_str"] = df["tags_unified"].apply(
        lambda x: "|".join(x) if isinstance(x, list) else "")
    df[CSV_COLS].to_csv(UNIFIED_CSV, index=False)

    t = Table(title="Tag coverage after the CodeChef pass")
    t.add_column("platform"); t.add_column("total", justify="right")
    t.add_column("with tags", justify="right")
    for platform in ["codeforces", "atcoder", "codechef", "leetcode"]:
        subset = df[df["platform"] == platform]
        n = len(subset)
        with_tags = (subset["tags_unified"].apply(
            lambda x: len(x) if isinstance(x, list) else 0) > 0).sum()
        t.add_row(platform, f"{n:,}", f"{with_tags:,} ({100 * with_tags / n:.0f}%)")
    console.print(t)


def main(dry_run: bool) -> None:
    texts, tag_lists = load_cf_training_data()
    if not texts:
        raise SystemExit("no Codeforces training data — run scripts/scrape_statements.py")

    vectorizer, classifier, mlb, valid_tags, report = train_and_evaluate(texts, tag_lists)

    codes, cc_texts = load_cc_statements()
    predictions = predict(vectorizer, classifier, mlb, codes, cc_texts)

    PREDICTIONS.write_text(json.dumps(predictions, indent=2))
    logger.info(f"Wrote predictions for {len(predictions)} problems to {PREDICTIONS.name}")

    apply_to_dataset(predictions, dry_run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transfer-tag CodeChef problems")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(args.dry_run)
