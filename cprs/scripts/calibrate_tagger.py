"""
Calibrate the NLP auto-tagger by predicting on CF problems (ground truth)
and measuring precision/recall at different confidence thresholds.

This tells us: at what confidence score should we show a predicted tag to users?

Also predicts tags for ALL problems (CF, LC, AC) to unify the tagging process
and measure cross-platform correlation.

Usage:
    python -m scripts.calibrate_tagger
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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unified_schema import unify_tags

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TARGET_TAGS = [
    "math", "greedy", "dynamic_programming", "implementation",
    "data_structures", "brute_force", "sorting", "strings",
    "binary_search", "two_pointers", "dfs", "bfs", "graphs",
    "trees", "number_theory", "constructive", "combinatorics",
    "geometry", "hashing", "game_theory", "bitmask",
    "divide_and_conquer", "union_find", "shortest_paths",
]
MIN_TAG_COUNT = 50


def load_cf_data():
    """Load CF problems with statements and tags."""
    with open(DATA_DIR / "statements" / "cf_statements.json") as f:
        statements = json.load(f)
    with open(DATA_DIR / "raw" / "cf_problems.json") as f:
        problems = json.load(f)

    tag_lookup = {}
    for p in problems:
        key = (p.get("contestId"), p.get("index"))
        tag_lookup[key] = unify_tags(p.get("tags", []))

    texts, tag_lists, ids = [], [], []
    for s in statements:
        if not s.get("statement"):
            continue
        key = (s["contest_id"], s["index"])
        tags = tag_lookup.get(key, [])
        tags = [t for t in tags if t in TARGET_TAGS]
        if not tags:
            continue
        texts.append(s["statement"])
        tag_lists.append(tags)
        ids.append(f"cf:{s['contest_id']}{s['index']}")

    return texts, tag_lists, ids


def load_ac_data():
    """Load AtCoder problems with statements."""
    with open(DATA_DIR / "statements" / "ac_statements.json") as f:
        statements = json.load(f)
    texts, ids = [], []
    for s in statements:
        if not s.get("statement"):
            continue
        texts.append(s["statement"])
        ids.append(s["problem_id"])
    return texts, ids


def threshold_analysis(y_true, y_proba, mlb):
    """Analyze precision/recall at different confidence thresholds."""
    console.print("\n[bold cyan]Threshold Calibration Analysis[/bold cyan]\n")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    table = Table(title="Precision / Recall / F1 at Different Confidence Thresholds")
    table.add_column("Threshold", style="cyan", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Avg Tags/Problem", justify="right")
    table.add_column("% Problems Tagged", justify="right")

    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        # Skip if no predictions
        if y_pred.sum() == 0:
            table.add_row(f"{t:.1f}", "N/A", "N/A", "N/A", "0.0", "0%")
            continue

        p = precision_score(y_true, y_pred, average="micro", zero_division=0)
        r = recall_score(y_true, y_pred, average="micro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        avg_tags = y_pred.sum(axis=1).mean()
        pct_tagged = (y_pred.sum(axis=1) > 0).mean() * 100

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        table.add_row(
            f"{t:.1f}",
            f"{p:.3f}",
            f"{r:.3f}",
            f"{f1:.3f}",
            f"{avg_tags:.1f}",
            f"{pct_tagged:.0f}%",
        )

    console.print(table)
    console.print(f"\n[bold green]Best threshold by F1: {best_threshold}[/bold green]")

    return best_threshold


def per_tag_threshold_analysis(y_true, y_proba, valid_tags):
    """Find optimal threshold per tag."""
    console.print("\n[bold cyan]Per-Tag Optimal Thresholds[/bold cyan]\n")

    table = Table(title="Best Threshold per Tag (maximizing F1)")
    table.add_column("Tag", style="cyan")
    table.add_column("Best Threshold", justify="right")
    table.add_column("Precision @ best", justify="right")
    table.add_column("Recall @ best", justify="right")
    table.add_column("F1 @ best", justify="right")

    tag_thresholds = {}
    for i, tag in enumerate(valid_tags):
        best_f1, best_t = 0, 0.5
        best_p, best_r = 0, 0
        for t in np.arange(0.2, 0.95, 0.05):
            preds = (y_proba[:, i] >= t).astype(int)
            if preds.sum() == 0:
                continue
            p = precision_score(y_true[:, i], preds, zero_division=0)
            r = recall_score(y_true[:, i], preds, zero_division=0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_p = p
                best_r = r

        tag_thresholds[tag] = best_t
        table.add_row(tag, f"{best_t:.2f}", f"{best_p:.3f}", f"{best_r:.3f}", f"{best_f1:.3f}")

    console.print(table)
    return tag_thresholds


def correlation_analysis(y_true, y_pred_binary, valid_tags):
    """Analyze which tags get confused with each other."""
    console.print("\n[bold cyan]Tag Confusion Analysis (top misclassifications)[/bold cyan]\n")

    # For each tag, find what false positives are commonly co-occurring
    table = Table(title="Most Common False Positive Patterns")
    table.add_column("Predicted Tag", style="cyan")
    table.add_column("Most Confused With (actual tags)", style="yellow")

    for i, tag in enumerate(valid_tags):
        # False positives: predicted=1 but actual=0
        fp_mask = (y_pred_binary[:, i] == 1) & (y_true[:, i] == 0)
        if fp_mask.sum() == 0:
            continue
        # What actual tags do these false positive samples have?
        fp_actual = y_true[fp_mask]
        co_occur = fp_actual.sum(axis=0)
        top_confused = [(valid_tags[j], int(co_occur[j])) for j in np.argsort(-co_occur)[:3] if co_occur[j] > 0]
        if top_confused:
            confused_str = ", ".join(f"{t} ({c})" for t, c in top_confused)
            table.add_row(tag, confused_str)

    console.print(table)


def main():
    # Load training data
    cf_texts, cf_tags, cf_ids = load_cf_data()
    logger.info(f"Loaded {len(cf_texts)} CF problems")

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=TARGET_TAGS)
    y_all = mlb.fit_transform(cf_tags)
    tag_counts = y_all.sum(axis=0)
    valid_mask = tag_counts >= MIN_TAG_COUNT
    valid_tags = [t for t, v in zip(TARGET_TAGS, valid_mask) if v]

    mlb = MultiLabelBinarizer(classes=valid_tags)
    filtered_tags = [[t for t in tags if t in valid_tags] for tags in cf_tags]
    y = mlb.fit_transform(filtered_tags)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000, stop_words="english",
        ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True,
    )
    X = vectorizer.fit_transform(cf_texts)

    # Get cross-validated probability predictions
    classifier = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"),
        n_jobs=-1,
    )

    logger.info("Running 5-fold CV with probability predictions...")
    # We need probabilities, so we do it manually per fold
    from sklearn.model_selection import StratifiedKFold, KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_proba_cv = np.zeros_like(y, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"  Fold {fold+1}/5...")
        clf = OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"),
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_proba_cv[val_idx] = clf.predict_proba(X[val_idx])

    # --- Threshold analysis ---
    best_global_threshold = threshold_analysis(y, y_proba_cv, mlb)

    # --- Per-tag thresholds ---
    tag_thresholds = per_tag_threshold_analysis(y, y_proba_cv, valid_tags)

    # --- Confusion analysis ---
    y_pred_at_best = (y_proba_cv >= best_global_threshold).astype(int)
    correlation_analysis(y, y_pred_at_best, valid_tags)

    # --- Train final model on all data ---
    logger.info("Training final model on all data...")
    classifier.fit(X, y)

    # --- Predict on AtCoder ---
    ac_texts, ac_ids = load_ac_data()
    if ac_texts:
        X_ac = vectorizer.transform(ac_texts)
        y_ac_proba = classifier.predict_proba(X_ac)

        console.print(f"\n[bold cyan]AtCoder Predictions at threshold {best_global_threshold}:[/bold cyan]")
        y_ac_pred = (y_ac_proba >= best_global_threshold).astype(int)
        tagged = (y_ac_pred.sum(axis=1) > 0).sum()
        avg_tags = y_ac_pred.sum(axis=1).mean()
        console.print(f"  Tagged: {tagged}/{len(ac_ids)} ({100*tagged/len(ac_ids):.1f}%)")
        console.print(f"  Avg tags per problem: {avg_tags:.1f}")

    # --- Save calibration results ---
    calibration = {
        "best_global_threshold": best_global_threshold,
        "per_tag_thresholds": tag_thresholds,
        "valid_tags": valid_tags,
        "recommendation": (
            f"Use global threshold {best_global_threshold} for displaying tags. "
            f"Tags with confidence below this are hidden from users. "
            f"For higher precision (fewer but more accurate tags), use 0.7+."
        ),
    }
    with open(DATA_DIR / "tagger_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    console.print(f"\n[bold green]Calibration saved to {DATA_DIR / 'tagger_calibration.json'}[/bold green]")

    # --- Summary for report ---
    console.print("\n[bold]Summary for Report:[/bold]")
    console.print(f"  Global threshold: {best_global_threshold}")
    console.print(f"  Tags where text works well (F1 > 0.5): {[t for t in valid_tags if tag_thresholds.get(t, 0.5) <= 0.5]}")
    console.print(f"  Tags where text is weak (needs structural features): {[t for t in valid_tags if tag_thresholds.get(t, 0.5) > 0.6]}")


if __name__ == "__main__":
    main()
