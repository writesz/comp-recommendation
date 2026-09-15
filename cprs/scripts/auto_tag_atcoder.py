"""
NLP-based auto-tagger for AtCoder problems.

Trains a multi-label classifier on Codeforces problem statements (which have
human-curated tags) and uses it to predict tags for AtCoder problems (which
have no tags).

Pipeline:
1. Load CF statements + tags as training data
2. TF-IDF vectorization of problem text
3. Train OneVsRest multi-label classifier
4. Evaluate with cross-validation (precision, recall, F1)
5. Predict tags for AtCoder problems
6. Save enriched dataset

This is a key technical contribution of CPRS: using NLP to transfer
topic knowledge across competitive programming platforms.

Usage:
    python -m scripts.auto_tag_atcoder
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unified_schema import unify_tags

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Minimum number of training samples for a tag to be included
MIN_TAG_COUNT = 50
# Tags to predict (unified taxonomy)
TARGET_TAGS = [
    "math", "greedy", "dynamic_programming", "implementation",
    "data_structures", "brute_force", "sorting", "strings",
    "binary_search", "two_pointers", "dfs", "bfs", "graphs",
    "trees", "number_theory", "constructive", "combinatorics",
    "geometry", "hashing", "game_theory", "bitmask",
    "divide_and_conquer", "union_find", "shortest_paths",
]


def load_cf_training_data():
    """
    Load CF problems with statements and tags.
    Returns (texts, tag_lists) where tag_lists uses unified taxonomy.
    """
    # Load statements
    with open(DATA_DIR / "statements" / "cf_statements.json") as f:
        statements = json.load(f)

    # Load raw problems (for tags)
    with open(DATA_DIR / "raw" / "cf_problems.json") as f:
        problems = json.load(f)

    # Build lookup: (contestId, index) -> tags
    tag_lookup = {}
    for p in problems:
        key = (p.get("contestId"), p.get("index"))
        tag_lookup[key] = unify_tags(p.get("tags", []))

    texts = []
    tag_lists = []
    for s in statements:
        if not s.get("statement"):
            continue
        key = (s["contest_id"], s["index"])
        tags = tag_lookup.get(key, [])
        # Filter to target tags only
        tags = [t for t in tags if t in TARGET_TAGS]
        if not tags:
            continue
        texts.append(s["statement"])
        tag_lists.append(tags)

    logger.info(f"Loaded {len(texts)} CF problems with statements and tags")
    return texts, tag_lists


def load_ac_inference_data():
    """Load AtCoder problems with statements for tag prediction."""
    statements_path = DATA_DIR / "statements" / "ac_statements.json"
    if not statements_path.exists():
        logger.error(f"AtCoder statements not found at {statements_path}")
        logger.error("Run: python -m scripts.scrape_statements --platform ac")
        return [], []

    with open(statements_path) as f:
        statements = json.load(f)

    problem_ids = []
    texts = []
    for s in statements:
        if not s.get("statement"):
            continue
        problem_ids.append(s["problem_id"])
        texts.append(s["statement"])

    logger.info(f"Loaded {len(texts)} AtCoder problems with statements")
    return problem_ids, texts


def train_and_evaluate(texts, tag_lists):
    """
    Train the multi-label classifier and evaluate with cross-validation.
    Returns (vectorizer, classifier, mlb) for inference.
    """
    console.print("\n[bold cyan]Training NLP auto-tagger...[/bold cyan]\n")

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=TARGET_TAGS)
    y = mlb.fit_transform(tag_lists)

    # Filter tags with too few samples
    tag_counts = y.sum(axis=0)
    valid_mask = tag_counts >= MIN_TAG_COUNT
    valid_tags = [t for t, v in zip(TARGET_TAGS, valid_mask) if v]
    logger.info(f"Tags with >= {MIN_TAG_COUNT} samples: {len(valid_tags)}/{len(TARGET_TAGS)}")

    # Re-fit with only valid tags
    mlb = MultiLabelBinarizer(classes=valid_tags)
    filtered_tags = [[t for t in tags if t in valid_tags] for tags in tag_lists]
    y = mlb.fit_transform(filtered_tags)

    # TF-IDF vectorization
    logger.info("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF matrix: {X.shape[0]} docs x {X.shape[1]} features")

    # Train classifier
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        ),
        n_jobs=-1,
    )

    # Cross-validation evaluation
    logger.info("Running 5-fold cross-validation...")
    y_pred_cv = cross_val_predict(classifier, X, y, cv=5)

    # Per-tag metrics
    console.print("\n[bold]Cross-Validation Results (5-fold):[/bold]\n")
    table = Table(title="Per-Tag Classification Metrics")
    table.add_column("Tag", style="cyan")
    table.add_column("Support", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    report = classification_report(y, y_pred_cv, target_names=valid_tags, output_dict=True, zero_division=0)
    for tag in valid_tags:
        m = report[tag]
        table.add_row(
            tag,
            str(int(m["support"])),
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1-score']:.3f}",
        )

    # Overall metrics
    macro_f1 = f1_score(y, y_pred_cv, average="macro", zero_division=0)
    micro_f1 = f1_score(y, y_pred_cv, average="micro", zero_division=0)
    table.add_row("", "", "", "", "")
    table.add_row("[bold]Macro avg[/bold]", "", "", "", f"[bold]{macro_f1:.3f}[/bold]")
    table.add_row("[bold]Micro avg[/bold]", "", "", "", f"[bold]{micro_f1:.3f}[/bold]")
    console.print(table)

    # Train final model on all data
    logger.info("Training final model on all data...")
    classifier.fit(X, y)

    return vectorizer, classifier, mlb, valid_tags, report


def predict_ac_tags(vectorizer, classifier, mlb, problem_ids, texts):
    """Predict tags for AtCoder problems using the trained classifier."""
    console.print("\n[bold cyan]Predicting tags for AtCoder problems...[/bold cyan]\n")

    X_ac = vectorizer.transform(texts)
    y_pred = classifier.predict(X_ac)
    y_proba = classifier.predict_proba(X_ac)

    predictions = {}
    for i, pid in enumerate(problem_ids):
        pred_tags = mlb.inverse_transform(y_pred[i:i+1])[0]
        # Also store confidence scores for all tags
        tag_scores = {tag: float(y_proba[i][j]) for j, tag in enumerate(mlb.classes_)}
        predictions[pid] = {
            "predicted_tags": list(pred_tags),
            "tag_scores": tag_scores,
        }

    # Summary
    tagged = sum(1 for p in predictions.values() if p["predicted_tags"])
    avg_tags = np.mean([len(p["predicted_tags"]) for p in predictions.values()])
    console.print(f"Tagged {tagged}/{len(predictions)} AtCoder problems ({100*tagged/len(predictions):.1f}%)")
    console.print(f"Average tags per problem: {avg_tags:.1f}")

    # Tag distribution
    from collections import Counter
    all_pred_tags = []
    for p in predictions.values():
        all_pred_tags.extend(p["predicted_tags"])
    tag_dist = Counter(all_pred_tags)
    console.print("\n[bold]Predicted tag distribution for AtCoder:[/bold]")
    for tag, count in tag_dist.most_common(15):
        console.print(f"  {tag}: {count}")

    return predictions


def update_unified_dataset(predictions):
    """Update the unified dataset with predicted AtCoder tags."""
    console.print("\n[bold cyan]Updating unified dataset...[/bold cyan]\n")

    with open(DATA_DIR / "cprs_unified.json") as f:
        dataset = json.load(f)

    updated = 0
    for problem in dataset:
        if problem["platform"] != "atcoder":
            continue
        pid = problem["platform_id"]
        if pid in predictions:
            pred = predictions[pid]
            problem["tags_unified"] = pred["predicted_tags"]
            problem["tags_original"] = [f"predicted:{t}" for t in pred["predicted_tags"]]
            updated += 1

    logger.info(f"Updated {updated} AtCoder problems with predicted tags")

    # Save updated dataset
    with open(DATA_DIR / "cprs_unified_tagged.json", "w") as f:
        json.dump(dataset, f, indent=2)

    # Also save as CSV
    df = pd.DataFrame(dataset)
    df["tags_original_str"] = df["tags_original"].apply(lambda x: "|".join(x) if isinstance(x, list) else "")
    df["tags_unified_str"] = df["tags_unified"].apply(lambda x: "|".join(x) if isinstance(x, list) else "")
    csv_cols = [
        "cprs_id", "platform", "platform_id", "name", "url", "contest_id",
        "difficulty_raw", "difficulty_normalized",
        "tags_original_str", "tags_unified_str",
        "solve_count", "acceptance_rate", "is_premium",
    ]
    df[csv_cols].to_csv(DATA_DIR / "cprs_unified_tagged.csv", index=False)

    # Print updated coverage
    table = Table(title="Updated Tag Coverage")
    table.add_column("Platform", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("With Tags", justify="right")

    for platform in ["codeforces", "atcoder", "leetcode"]:
        subset = df[df["platform"] == platform]
        with_tags = (subset["tags_unified"].apply(lambda x: len(x) if isinstance(x, list) else 0) > 0).sum()
        table.add_row(platform, str(len(subset)), f"{with_tags} ({100*with_tags/len(subset):.0f}%)")
    console.print(table)


def main():
    # Step 1: Load training data
    texts, tag_lists = load_cf_training_data()
    if not texts:
        logger.error("No training data. Run scrape_statements first.")
        return

    # Step 2: Train and evaluate
    vectorizer, classifier, mlb, valid_tags, report = train_and_evaluate(texts, tag_lists)

    # Save evaluation report
    with open(DATA_DIR / "tagger_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    # Step 3: Predict AtCoder tags
    problem_ids, ac_texts = load_ac_inference_data()
    if not ac_texts:
        logger.warning("No AtCoder statements available yet. Skipping prediction.")
        return

    predictions = predict_ac_tags(vectorizer, classifier, mlb, problem_ids, ac_texts)

    # Save predictions
    with open(DATA_DIR / "ac_predicted_tags.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # Step 4: Update unified dataset
    update_unified_dataset(predictions)

    # Log to decisions
    console.print("\n[bold green]Done! NLP auto-tagger complete.[/bold green]")
    console.print(f"Results saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
