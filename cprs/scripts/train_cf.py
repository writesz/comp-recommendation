"""
Train the collaborative-filtering model once and persist it for serving.

`scripts/evaluate.py` fits ALS on every run, which is fine for a batch
experiment but not for a web request. This script fits the model a single time
and writes the item factors, the hyperparameters and the item space to
`data/cf_model.npz`, from which the application loads it at startup and folds
individual users in on demand.

The cross-platform matrix is the default training input: it covers 10,239
problems (Codeforces plus AtCoder) against the Codeforces-only matrix's 7,568,
so the served model can say something about more of the catalogue. Neither
matrix contains LeetCode problems — the cohort was sampled from Codeforces
submissions — so LeetCode recommendations remain content-only by construction.

Usage:
    python scripts/train_cf.py                 # cross-platform matrix
    python scripts/train_cf.py --matrix cf     # Codeforces-only matrix
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import scipy.sparse as sp
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.collaborative import CollaborativeModel

DATA = Path(__file__).resolve().parent.parent / "data"
INTERACTIONS = DATA / "interactions"

SOURCES = {
    "xplat": (INTERACTIONS / "merged_train.npz", INTERACTIONS / "xplat_mappings.json"),
    "cf": (INTERACTIONS / "train_matrix.npz", INTERACTIONS / "mappings.json"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", choices=sorted(SOURCES), default="xplat",
                    help="which interaction matrix to train on (default: xplat)")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--regularization", type=float, default=0.05)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--out", default=str(DATA / "cf_model.npz"))
    args = ap.parse_args()

    matrix_path, mappings_path = SOURCES[args.matrix]
    for p in (matrix_path, mappings_path):
        if not p.exists():
            logger.error(f"missing {p} — run the interaction build scripts first")
            return 1

    train = sp.load_npz(matrix_path).tocsr()
    problem_ids = json.load(open(mappings_path))["problems"]

    if train.shape[1] != len(problem_ids):
        logger.error(
            f"matrix has {train.shape[1]} columns but mappings list "
            f"{len(problem_ids)} problems — these artefacts disagree"
        )
        return 1

    platforms = Counter(pid.split(":")[0] for pid in problem_ids)
    logger.info(
        f"training on {args.matrix}: {train.shape[0]} users x {train.shape[1]} problems, "
        f"{train.nnz} interactions, item space {dict(platforms)}"
    )

    model = CollaborativeModel(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha,
    ).fit(train)

    model.save(args.out, problem_ids=problem_ids)

    # A trained user and their fold-in should agree. Report it here so a bad
    # artefact is obvious at build time rather than at serving time.
    import numpy as np
    rows = np.random.default_rng(0).choice(train.shape[0], size=min(50, train.shape[0]),
                                           replace=False)
    sims = []
    for u in rows:
        cols = train.indices[train.indptr[u]:train.indptr[u + 1]]
        if cols.size == 0:
            continue
        trained = model.user_factors[u]
        folded = model.fold_in(cols)
        denom = np.linalg.norm(trained) * np.linalg.norm(folded)
        if denom > 0:
            sims.append(float(trained @ folded / denom))
    if sims:
        logger.info(
            f"fold-in agreement with trained factors over {len(sims)} users: "
            f"mean cosine {np.mean(sims):.4f} (min {np.min(sims):.4f})"
        )

    logger.success(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
