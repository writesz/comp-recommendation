# CPRS — Final Project Build Plan (v2, rubric-mapped)

> Goal: take the working prototype to a **1st-class** final submission
> (Template 1.1, Data Science) in **~1 week** of working days, **offline evaluation only**.
> Revised 2026-09-19 against `final_report_rubric_v2.pdf`. Target submission ~2026-09-27.
> Supersedes the v1 plan; day structure tightened to fit build + report + demo in one week.

## Decisions locked

- **Ambition:** aim for 1st from the start.
- **Deadline:** ~1 week of working days (revised 2026-09-19).
- **Evaluation data:** offline-only on real interaction data (no user study).
- **Interaction data source:** **CF Open Dataset (download) as primary**, API-fetch of ~2–5K
  sampled CF handles as contingency, sub-sample to keep the matrix tractable. *Why we need it:*
  collaborative filtering is a function of a user×problem matrix, and every offline ranking
  metric requires held-out real solves — it is the linchpin for Implementation, the whole
  Evaluation cluster, and Originality (~40+ report points).
- **First technical thrust:** **Hybrid = collaborative filtering + existing content-based**
  (moves 3rd→2:1, forces a real cold-start story).
- **Novelty framing (the 1st-class "novel approach"):** cross-platform hybrid recommendation
  + rigorous comparative evaluation on a *curated cross-platform dataset*.
- **Sequence/RNN model:** **cut-first stretch goal only.** Build only if Days 1–4 finish clean.

## What already exists (do not rebuild)

- ✅ Curated unified dataset — 24,335 problems (CF/AtCoder/LeetCode), normalised difficulty +
  unified tag taxonomy. (Banks a 1st-class dataset criterion + much of Originality.)
- ✅ NLP auto-tagger for AtCoder — TF-IDF + OneVsRest LogReg, 5-fold CV, threshold calibration.
- ✅ Content-based recommender + multi-platform user profiles (CF/AtCoder/LeetCode APIs).
- ✅ FastAPI + SQLite web app (auth, profiles, recent-performance, recommendation cards).
- ✅ Preliminary report (`.md`/`.tex`), decisions log.

## The gap to a 1st (all unbuilt — this is the week's work)

- ❌ No user×problem interaction dataset on disk.
- ❌ No collaborative filtering, hybrid, or cold-start implementation.
- ❌ No evaluation harness, baselines, significance tests, or figures.
- ❌ No design/architecture diagrams (Diagrams = 10 pts, near-zero right now).
- ❌ Report Results/Discussion chapters; ACM citation pass; ToC.

## Rubric coverage map (final_report_rubric_v2.pdf)

| Rubric criterion | Max | Status now | Delivered by |
|---|---:|---|---|
| Clearly written | 10 | partial | Report pass (Day 6–7) |
| Diagrams appropriate & clear | 10 | ❌ ~zero | Design + eval figures (Day 4–5) |
| Knowledge of area / lit | 10 | ✅ strong | Existing lit review |
| Critically evaluate prior work | 6 | weak (TA-flagged) | Gap→contribution table (Day 5) |
| Proper ACM citation | 4 | ❌ | Citation pass (Day 6) |
| Design clear & high quality | 12 | weak | Design chapter + diagrams + cold-start (Day 3,5) |
| Concept justified by domain/users | 8 | ✅ mostly | Existing + tighten (Day 5) |
| Implementation high quality | 22 | partial (content-only) | Hybrid + eval end-to-end (Day 1–4) |
| Implementation technically challenging | 8 | partial | CF + hybrid + cross-platform (Day 2–3) |
| Eval strategy appropriate | 6 | ❌ | Harness design (Day 2) |
| Eval coverage | 5 | ❌ | 5 models × ≥4 metrics + CV (Day 2–4) |
| Eval results presented well | 5 | ❌ | Figures + tables (Day 4) |
| Eval → critical analysis vs objectives | 4 | ❌ | Discussion chapter (Day 6) |
| Originality | 10 | ✅ mostly | Cross-platform hybrid + comparative eval (Day 3–4) |
| Video: final product + technical | 10 | ❌ | Demo MP4 (Day 7) |

## Day-by-day (working days)

### Day 1 — Interaction data + evaluation split  *(highest risk, linchpin)*
- Acquire CF Open Dataset (primary) / API-fetch ~2–5K sampled handles (contingency B: sub-sample).
- Build sparse user×problem implicit-feedback matrix (solved = 1).
- Time-based split: hold out each user's most-recent-N solves as test.
- **Deliverables:** `scripts/build_interactions.py`, `scripts/split_interactions.py`,
  `data/interactions/matrix.npz` + train/test split.
- **Feeds:** Implementation(22), all Evaluation(20), Originality(10).

### Day 2 — Collaborative filtering + evaluation harness
- Matrix factorisation on implicit feedback (ALS via `implicit`, or SVD); generate top-N.
- Evaluation harness: HitRate@K, MRR, nDCG@K, Precision/Recall@K.
- Baselines wired in: random, popularity, content-only, CF-only.
- **Deliverables:** `models/collaborative.py`, `scripts/evaluate.py`, first metrics table.
- **Feeds:** Implementation, Eval strategy(6) + coverage(5), Technical challenge(8).

### Day 3 — Hybrid model + cold-start
- Blend CF + content score; blend weight α = f(user interaction density).
  Sparse/new users → content-only (**cold-start**); dense users → CF-weighted.
- Add hybrid to the harness; k-fold cross-validation over users.
- **Deliverables:** `models/hybrid.py`, documented cold-start behaviour.
- **Feeds:** Implementation(22), Technical challenge(8), Design(12, the TA-requested cold-start).

### Day 4 — Significance + figures + cross-platform value
- Paired significance tests (Wilcoxon / paired t) on per-user metrics between models + effect sizes.
- Figures: metric-vs-K curves, PR curves, model-comparison bars, difficulty-calibration plot,
  topic-coverage plot, **single-platform vs merged cross-platform** profiles (the novel angle).
- **Deliverables:** `data/eval_results.json`, `figures/*.png`.
- **Feeds:** Eval results(5), critical analysis(4), Diagrams(10), Originality(10).

### Day 5 — Design diagrams + report core chapters
- Diagrams: system architecture, data pipeline, model/hybrid diagram, ER diagram.
- Write/refresh: Design chapter (diagrams + cold-start), Implementation chapter,
  gap→contribution table in Lit Review, Table of Contents.
- **Deliverables:** `figures/design/*`, report Design + Implementation sections.
- **Feeds:** Design(12), Diagrams(10), Lit critical eval(6).

### Day 6 — Results/Discussion + citations
- Results chapter (tables + figures woven in); Discussion critically analysing results
  vs the project's stated objectives (balanced good/bad, evidence-driven).
- ACM citation pass — every source cited, correct ACM style (4 pts all-or-nothing).
- **Deliverables:** report Results + Discussion, `references.bib` in ACM style.
- **Feeds:** Results(5), critical analysis(4), Citations(4), Written(10).

### Day 7 — Demo video + cleanup + buffer
- Record 3–5 min MP4 demo (web app walkthrough + evaluation highlights).
- README, code cleanup, final proofread, export report PDF.
- Buffer for whatever slipped on Days 1–6.
- **Deliverables:** `demo.mp4`, `README.md`, final report PDF, clean repo.
- **Feeds:** Video(/10), Written presentation(10).

## Cut order if time slips
RNN stretch (already cut) → extra baselines → cross-platform *significance* (keep descriptive)
→ figure polish. **Never cut:** hybrid model, core ranking metrics, ≥1 significance test,
design diagrams, demo video.

## Definition of done (1st-class bar)
- [ ] Interaction matrix built + time-based train/test split
- [ ] Hybrid recommender working end-to-end
- [ ] ≥5 models compared on ≥4 metrics with cross-validation
- [ ] Statistical significance reported with effect sizes
- [ ] All results visualised, not just tabulated
- [ ] Cold-start handled in design + code
- [ ] Cross-platform value quantified
- [ ] Design + architecture diagrams in report
- [ ] ACM citation pass complete
- [ ] Final report PDF + MP4 demo + clean repo with README
