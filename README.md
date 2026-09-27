# CPRS — Cross-Platform Competitive Programming Recommender

CPRS recommends the next competitive-programming problem a user should attempt,
drawing on their solve history across **Codeforces, AtCoder and LeetCode** at once.
It unifies 24,335 problems from the three platforms onto a single difficulty scale
and a single topic taxonomy, learns from the solve patterns of 2,459 real users,
and calibrates difficulty from contest ratings rather than from practice habits.

Final-year project for CM3070, University of London
(Template 1.1 — Data-Driven Personalised Educational Content Recommendation).

---

## What it does

- **Unifies three catalogues.** Codeforces ratings, AtCoder difficulty estimates and
  LeetCode Easy/Medium/Hard are normalised onto one `[0,1]` scale; platform tag
  vocabularies fold into 98 shared topics.
- **Recovers missing metadata.** AtCoder publishes no topic tags, so a multi-label
  classifier trained on tagged Codeforces statements transfers them across platforms.
- **Recommends collaboratively.** Implicit-feedback ALS over a user×problem matrix,
  served to users who were never in the training cohort via closed-form fold-in.
- **Calibrates difficulty from contests.** A contest rating is an Elo estimate of what
  you can solve unaided under time pressure — a better target than the difficulty of
  problems you have already solved, which counts editorial-assisted and upsolved work.
- **Reports progress.** Per-topic mastery, strengths and weak areas, and a contest
  timeline with derived improvement points.

## Headline results

Offline evaluation over 2,459 users, full-catalogue ranking (no sampled negatives),
temporal leave-last-N split.

| Model | nDCG@10 | HitRate@10 |
|---|---:|---:|
| **Collaborative filtering** | **0.0959** | **0.2952** |
| Hybrid + popularity | 0.0901 | 0.2985 |
| Hybrid (CF + content) | 0.0810 | 0.2908 |
| Popularity | 0.0374 | 0.1440 |
| Difficulty-match | 0.0061 | 0.0407 |
| Content-based | 0.0012 | 0.0094 |
| Random | 0.0012 | 0.0114 |

Three findings the project did not expect, and which shaped the final design:

1. **The hybrid never beats plain CF**, and is *worst* where it was meant to help —
   for users with fewer than ten solves it roughly halves nDCG@10. A small blend
   weight hands most of the score to a content signal that performs at chance, so the
   cold-start path dilutes a working signal in inverse proportion to available history.
   The deployed system therefore falls back to content only when CF is *silent*,
   rather than blending proportionally.
2. **Content-based scoring performs at chance** on next-solve prediction. This is a
   metric–task mismatch rather than a broken model: it recommends what a learner
   *should* practise, while the metric rewards predicting what they *did* solve next.
3. **Cross-platform collaborative transfer did not help** (0.0393 single-platform vs
   0.0380 merged), although cross-platform *skill* transfer did.

The clearest win is cold start: CF rises from 0.001 with no history to **0.135 after a
single observed solve**, where popularity stays flat at 0.010.

---

## Quick start

Requires **Python 3.9+**.

```bash
pip install fastapi uvicorn requests aiohttp pydantic python-dotenv sqlalchemy \
            beautifulsoup4 lxml rich loguru tqdm numpy scipy pandas \
            scikit-learn implicit pytest

cd cprs
python -m uvicorn app:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>. Register, add at least one platform handle
(`tourist` on Codeforces or AtCoder, `lee215` on LeetCode are good test accounts),
and recommendations appear on `/app`.

The app reads **public profile data only** and never asks for platform credentials.

### Collaborative filtering

The app serves content-based recommendations out of the box. To enable CF, build the
model once (about 15 seconds):

```bash
cd cprs
python scripts/train_cf.py
```

This writes `data/cf_model.npz`. The app picks it up at startup; without it, it logs a
warning and continues content-only.

### Tests

```bash
cd cprs
python -m pytest tests -q        # 91 tests
```

---

## How it is put together

```
fetchers/            Codeforces REST, AtCoder (kenkoooo), LeetCode GraphQL
  └─ scripts/build_dataset.py            24,335 problems, unified schema
       ├─ scripts/auto_tag_atcoder.py    TF-IDF + one-vs-rest logistic regression
       └─ scripts/fetch_interactions.py
            └─ scripts/build_interactions.py    k-core prune -> 2,459 x 7,568
                 ├─ scripts/split_interactions.py    temporal leave-last-N
                 ├─ scripts/build_crossplatform.py   + AtCoder -> 2,459 x 10,239
                 ├─ scripts/evaluate.py              7 models x 5 metrics x 3 cut-offs
                 │    ├─ scripts/significance.py     Wilcoxon + bootstrap CIs
                 │    └─ scripts/coldstart_sim.py    nDCG vs. observed solves
                 └─ scripts/train_cf.py              persisted ALS factors
                      └─ app.py                      fold-in per request
```

### Models (`cprs/models/`)

| Module | What it implements |
|---|---|
| `unified_schema.py` | difficulty normalisation and the 98-tag taxonomy |
| `recommender.py` | content engine: topic gap, difficulty fit, popularity, diversity |
| `collaborative.py` | implicit ALS (Hu–Koren–Volinsky), fold-in, persistence |
| `hybrid.py` | density-weighted blend, `α(u) = n/(n+k₀)` |
| `contests.py` | contest normalisation, improvement insights, difficulty calibration |
| `skill.py` | online Robbins–Monro quantile skill estimator |
| `metrics.py` | HitRate, Precision, Recall, MRR, nDCG |
| `database.py` | SQLite: users, platform handles, sessions, subscribers |

### Serving unseen users

ALS is transductive — it learns a user-factor table with one row per training user, so
a new registration has no row to look up. The application folds each request's user
into the fixed item factors using the closed-form ALS user step:

```
x_u = (YᵀY + α·Y_SᵀY_S + λI)⁻¹ (1+α) Σ_{i∈S} y_i
```

`YᵀY` is precomputed, leaving one 64×64 solve per user: 0.28 ms, or 0.63 ms including
scoring all 10,239 items. Validated by withholding 400 users from training entirely and
folding them in — 0.1035 nDCG@10 against 0.1025 for the same users under a model that
did train on them (Wilcoxon p = 0.93), i.e. statistically indistinguishable.

Reproduce with `python scripts/validate_foldin.py`.

---

## The dataset

| | Codeforces | AtCoder | LeetCode |
|---|---:|---:|---:|
| Problems | 11,263 | 9,095 | 3,977 |
| With difficulty | 97% | 52% | 100% |
| With topic tags | 98% | 32% → 35% | 97% |

AtCoder tag coverage moves from 32% to 35% rather than to full coverage, because only
795 of roughly 5,000 statements could be scraped to feed the classifier — the ceiling is
statement availability, not classifier quality.

Interaction matrix: 2,459 users × 7,568 problems, 532,059 solves, after an iterative
k-core prune (users with ≥5 solves, problems with ≥5 solvers). The cross-platform
matrix adds AtCoder columns solved by at least three cohort users, giving
2,459 × 10,239.

Committed artefacts under `cprs/data/` allow every result to be reproduced without
re-fetching from the platform APIs.

---

## Ethics and data handling

Only public profile data is read, through documented public APIs, with rate limiting.
The application stores identity, not behaviour: solve histories are re-fetched from the
origin platforms on demand rather than mirrored, so no copy of anyone's submission
record is held. Passwords are salted and hashed, and the credential database is never
committed.
