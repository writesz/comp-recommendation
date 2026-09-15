# CPRS — Final Project Build Plan (~1 Week)

> Goal: take the prototype to a **1st-class** final submission (Template 1.1, Data Science)
> in ~7 days of everyday work, **offline evaluation only**.
> Written 2026-09-14. Execution starts **after exam prep**.

## Decisions locked

- **Ambition:** aim for 1st from the start.
- **Evaluation data:** offline-only, on real interaction data (no user study).
- **First technical thrust:** **Hybrid = collaborative filtering + existing content-based** (chosen for highest grade-per-effort; it moves 3rd→2:1 and forces a real cold-start story).
- **Novelty framing (the 1st-class "novel approach"):** cross-platform hybrid recommendation + rigorous comparative evaluation on a *curated cross-platform dataset*. The curated dataset already satisfies a 1st-class criterion. A sequence/RNN model is a **stretch goal only**, not on the critical path.

## How this maps to the 1st-class rubric

| 1st-class criterion | Delivered by |
|---|---|
| Novel approach / adaptation of SOTA | Cross-platform hybrid (CF + content) with density-weighted blending + cold-start switching |
| Creation/curation of high-quality dataset | ✅ already done (24,335 problems, unified difficulty + tags) |
| Comprehensive eval w/ baselines + statistical significance | Phase B: 5 baselines, paired significance tests, cross-validation |
| Detailed analysis of performance and insights | Phase B/C: ablations, difficulty calibration, cross-platform value analysis |

## Hard constraints & top risks

1. **No interaction dataset on disk yet.** `data/raw/` has problem catalogues only. Collaborative filtering needs a user×problem matrix. **This is the #1 risk.** → Day 1 resolves it (see below), with contingency.
2. **One week must cover build + final report + demo video.** This is aggressive. If time slips, cut in this order: RNN stretch goal → embedding tagger → extra baselines. Never cut: hybrid model, core metrics, significance testing, visualisations.
3. Exam prep runs first and will eat calendar time — the 7 "build days" are working days, not consecutive.

## Day-by-day

### Day 1 — Acquire interaction data + build matrix  *(highest risk)*
- Obtain the CF Open Dataset (17.6M submissions / ~15K users). **Primary:** download the published dataset. **Contingency A:** if unavailable/too large, fetch submissions for a sampled set of ~2–5K CF handles via the CF API (`user.status`) — slower but self-sufficient. **Contingency B:** sub-sample users to keep the matrix tractable.
- Build sparse user×problem implicit-feedback matrix (solved = 1).
- Time-based split: hold out each user's most-recent-N solves as test.
- **Deliverable:** `interaction_matrix.npz` + train/test split script.

### Day 2 — Collaborative filtering model
- Matrix factorisation on implicit feedback (ALS via `implicit`, or SVD). Train, generate top-N.
- Sanity-check recommendations for known users.
- **Deliverable:** `models/collaborative.py` producing ranked lists.

### Day 3 — Hybrid model + cold-start
- Blend CF score with the existing content-based score; blend weight α = f(user interaction density). Sparse/new users → content-only (**cold-start**); dense users → CF-weighted.
- **Deliverable:** `models/hybrid.py`; documented cold-start behaviour (answers the design gap for real).

### Day 4 — Evaluation harness
- Metrics: HitRate@K, MRR, nDCG@K, Precision/Recall@K + domain metrics (difficulty calibration, topic coverage).
- Baselines: random, popularity, content-only, CF-only, hybrid.
- k-fold cross-validation over users.
- **Deliverable:** `scripts/evaluate.py` → metrics table across all models.

### Day 5 — Statistical significance + visualisation
- Paired significance tests (Wilcoxon / paired t-test) on per-user metric distributions between models; report effect sizes.
- Figures: metric-vs-K curves, PR curves, model-comparison bars, difficulty-calibration plot, coverage plot (matplotlib/seaborn).
- **Deliverable:** `data/eval_results.json` + `figures/*.png`.

### Day 6 — Cross-platform value analysis + iteration buffer
- Quantify the core thesis: single-platform vs merged cross-platform profiles, with significance. This is the novel evaluation angle.
- Buffer for whatever broke on Days 1–5. Optional stretch: per-tag tagger thresholds.
- **Deliverable:** cross-platform value result + figure.

### Day 7 — Final report integration + demo + cleanup
- Fold results into final report: Results/Discussion chapters with figures; add cold-start to Design; add gap→contribution table to Lit Review; ToC; citation-consistency pass.
- Record 3–5 min demo video (MP4).
- Code cleanup + README.
- **Deliverable:** final report draft, demo video, clean repo.

## Definition of done (1st-class bar)
- [ ] Hybrid recommender working end-to-end
- [ ] ≥5 models compared on ≥4 metrics with cross-validation
- [ ] Statistical significance reported with effect sizes
- [ ] All results visualised, not just tabulated
- [ ] Cold-start handled in design + code
- [ ] Cross-platform value quantified
- [ ] Final report + MP4 demo + clean code
