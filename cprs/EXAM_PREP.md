# CPRS — Final Project Exam Prep Pack

> **The exam is entirely about YOUR project.** No general CS. Answer **THREE of FIVE**
> questions, **20 marks each, 60 total**. Format is stable across the example and 2022
> papers — the same five question types recur. This pack has a model answer for all five.

## Exam strategy (read first)

- **Pick your best 3 on the day.** This pack preps all 5 so nothing surprises you. Likely strongest for CPRS: Q2 (moments), Q5 (alternatives), Q1 (references), Q4 (self-teaching).
- **Every answer must be grounded in CPRS specifics** — numbers, file names, decisions. Graders reward concrete evidence over generic reflection. Memorise the key figures below.
- **Structure each 20-mark answer**: direct claim → specific evidence → reasoning/critique. Where a question says "choose THREE/TWO", label them explicitly (a/b/c) so the grader can tick each.
- **Show self-awareness.** These questions reward honest critique (what went wrong, what you'd improve) as much as successes.

### Key figures to memorise
- **24,335 problems** unified: Codeforces 11,263 · AtCoder 9,095 · LeetCode 3,977.
- Difficulty coverage: CF 97%, AtCoder 52%, LC 100%. Tag coverage: CF 98%, AtCoder **32% → the core data problem**, LC 97%.
- NLP tagger: **micro-F1 0.530 / macro-F1 0.435**, 5-fold CV on 1,999 CF statements; global threshold **0.5**. Two tiers: content tags (strings 0.74, math 0.69) vs method tags (DP 0.20, two-pointers 0.18).
- AtCoder scraping: only **795 / ~5,000** statements (16%), rest HTTP 403.
- Recommender formula: `0.4·topic_gap + 0.4·difficulty_fit + 0.1·popularity + 0.1·diversity`; difficulty target = 75th percentile of solved + 10% stretch.
- Tested live on `tourist` (rating 3439, 3019 solved) and `jiangly`.

---

## Q1 — References (choose 3: high / medium / low quality; assess reliability)

*Strategy: pick three references at genuinely different reliability tiers and judge each on the same axes — venue & peer review, author authority, citations, methodological transparency, reproducibility, recency, and domain fit.*

**(a) High quality — Koren, Bell & Volinsky, "Matrix Factorization Techniques for Recommender Systems," *IEEE Computer*, 2009 [5].**
Highly reliable. It is **peer-reviewed** in an established IEEE venue, written by the authors who **won the Netflix Prize**, and is one of the most-cited papers in the field (>10,000 citations). Its claims are **empirically validated** on a large public benchmark with reproducible methodology, and its results have been independently replicated many times, which is the strongest form of reliability. The main caveat I apply is **domain fit**: it was developed for entertainment (movie ratings), where the goal is preference-matching, not the skill-progression goal of educational recommendation — so I treat its *techniques* as authoritative but its *framing* as needing adaptation for CPRS. Its age (2009, pre-deep-learning) is not a weakness here, because Dacrema et al. [24] later showed matrix factorisation remains a strong baseline.

**(b) Medium quality — kenkoooo, "AtCoder Problems: Difficulty Estimation," 2024 [17].**
Moderately reliable. It is the **de-facto community standard** for AtCoder data and difficulty, its methodology is **transparent** (logistic regression on solve rates) and empirically grounded, and I could inspect its behaviour directly. But it is **not peer-reviewed**, maintained by a **single developer**, and its difficulty values are **model-estimated, not official** — so they carry modelling assumptions I cannot fully audit. I therefore rely on it for data plumbing and *relative* difficulty, but flag in the report that cross-platform difficulty built on it is an approximation, not ground truth.

**(c) Lower reliability — Ahmed, "A4: A Codeforces Practice Assistant," GitHub, 2023 [3].**
Least reliable as an academic source. It is **self-published on GitHub**, **not peer-reviewed**, has **no formal evaluation** of its recommendation quality, and could change or disappear at any time (no archival guarantee). However, it is still **valuable as evidence of prior art and practical demand** — it shows a single-platform recommender is feasible and used. I cite it to motivate the gap CPRS fills (cross-platform), not to support any empirical claim. I would strengthen a reliability assessment of it by looking for an accompanying paper or independent benchmarks, of which there are none.

*Closing line:* reliability is not binary — I match how heavily I lean on each source to its evidential strength: canonical peer-reviewed work for method claims, community tools for data, and self-published tools only as motivation.

---

## Q2 — Three significant moments (discoveries or setbacks; how you overcame/exploited them)

**(a) Setback → contribution: AtCoder has no topic tags.**
Early in the data build I discovered AtCoder exposes difficulty estimates but **zero topic tags** — 9,095 problems (37% of the dataset) with no topic metadata, which would have made topic-based recommendation impossible for a third of the corpus. Rather than accept the gap or hand-map tags, I turned it into an **NLP contribution**: a multi-label classifier (TF-IDF + One-vs-Rest logistic regression) trained on 1,999 tagged Codeforces statements and used to **transfer tags cross-platform** to AtCoder. This converted a data limitation into one of the project's most technically interesting components (and a required template technique). It also produced a genuine research insight — the **content-tag vs method-tag tiering** (strings F1 0.74 vs dynamic-programming 0.20) — which I now use to bound expectations honestly rather than overclaim.

**(b) Setback: AtCoder blocked automated scraping (HTTP 403).**
To get statements for the tagger I scraped AtCoder, but only **795 of ~5,000** pages returned content — the rest were HTTP 403 blocks. I overcame it **partially**: throttling requests and targeting newer ABC contest pages that were accessible, which was enough to prove the tag-transfer concept. Critically, I **documented it honestly as a coverage limitation** rather than hiding it, and identified a concrete contingency for the final build — the **IBM CodeNet dataset** (4,053 AtCoder statements) as an offline alternative to scraping. This taught me to design around brittle data access with fallbacks rather than assuming an API/scrape will hold.

**(c) Discovery: LeetCode user data was accessible after all.**
My initial assumption was that LeetCode offered no public user data, so I **disabled LeetCode input in the UI** ("API limited — coming soon"), which would have reduced the project to two platforms. Instead of accepting it, I probed LeetCode's **GraphQL endpoint** and found two queries that work **without authentication** — `userPublicProfile` and `recentAcSubmissionList` — verified live against user `lee215` (642 solved returned correctly). This overturned a project-limiting assumption and **unlocked the full three-platform cross-platform value proposition**, which is the core of CPRS. The lesson: verify "impossible" constraints empirically before designing around them.

---

## Q3 — Evaluate your project process and strategy

### (a) TWO aspects that were good
1. **A unified dataset was built first, as the foundation.** I front-loaded the cross-platform dataset (24,335 normalised problems) before building models. This was the right sequencing: it de-risked the project's central assumption (that heterogeneous platforms *can* be unified) early, and the curated dataset is itself a first-class contribution. Everything downstream — tagger, profiles, recommender — depended on it, so stabilising it first avoided rework.
2. **Decisions were logged continuously and evaluation was kept honest.** I maintained a running `DECISIONS_LOG.md` capturing every design choice, alternative considered, and result. This gave traceability (I can justify *why* each decision was made, not reconstruct it after the fact) and kept evaluation candid — e.g. reporting the tagger's weak method-tag F1 and the two-tier insight rather than cherry-picking the strong tags.

### (b) TWO aspects that required improvement
1. **Recommender evaluation was only qualitative.** The engine was validated by inspecting outputs for known users (tourist, jiangly) — sensible but not rigorous. There were **no formal offline metrics** (Hit Rate@K, MRR, nDCG), no baselines, and no significance testing. This is the single biggest weakness and limits any claim about recommendation *quality*.
2. **Data-access strategy underestimated brittleness.** The AtCoder scrape achieved only 8.7% statement coverage because I underestimated 403 blocking, and LeetCode's public API returns only ~200 recent accepted submissions (no failed attempts, so no solve rate). I should have identified offline dataset fallbacks (CodeNet) at design time rather than after hitting the wall.

### (c) TWO improvements + how I'd ensure they happen
1. **Add formal offline evaluation.** Time-based hold-out on the Codeforces Open Dataset (hide each user's most recent N solves), measuring Hit Rate@K, MRR, nDCG against baselines (random, popularity, content-only, CF-only), with paired significance tests and cross-validation. *How ensured:* it is scheduled as Days 4–5 of my final build plan, with a dedicated `evaluate.py` harness and pre-committed metrics so results can't be retrofitted.
2. **Move from content-only to a hybrid recommender with cold-start handling.** *How ensured:* a concrete sequenced plan — acquire interaction data (Day 1), matrix factorisation (Day 2), density-weighted blend where sparse/new users fall back to content-based and dense users get CF weight (Day 3). This is documented in `FINAL_PROJECT_PLAN.md` with contingencies (API-fetch and sub-sampling) if the full dataset is unavailable.

---

## Q4 — Self-teaching (prior knowledge, new areas, how you acquired skills)

**(a) Pre-existing knowledge and how it shaped my template choice.**
I came in with strong **domain familiarity as a competitive programmer** — fluent with Codeforces, AtCoder and LeetCode as a user, and comfortable with algorithms and data structures — plus **Python** and foundational ML from coursework. This directly shaped choosing **Template 1.1 (Data Science)** over a web-development framing: I deliberately leaned into an area where my domain knowledge was an asset (e.g. knowing that "difficulty" means different things on each platform, and that AtCoder lacks tags), so I could spend learning effort on the data-science techniques rather than the domain.

**(b) New subject areas I had to engage with.**
- **Recommender systems theory** — collaborative vs content-based vs hybrid, matrix factorisation, the cold-start problem, factorisation machines (Koren [5], Burke [7], Rendle [8]).
- **Educational data mining / knowledge tracing** — Bayesian Knowledge Tracing, Deep Knowledge Tracing, and Elo-/IRT-based learner models (Corbett & Anderson [9], Piech [13], Pelánek [11]).
- **Multi-label NLP text classification** — TF-IDF, One-vs-Rest logistic regression, class imbalance handling, micro/macro-F1, and confidence-threshold calibration.
- **Recommender evaluation methodology** — Herlocker's framework [23], ranking metrics (Hit Rate, MRR, nDCG).
- **Practical API/data engineering** — Codeforces REST, the kenkoooo AtCoder API, LeetCode GraphQL, rate limiting, scraping, and building a FastAPI + SQLite service.

**(c) How I acquired these skills.**
Primarily by **reading primary literature** (the seminal papers behind each technique, not just tutorials), which is why I can critique the work rather than only apply it — e.g. reading Dacrema et al. [24] taught me to distrust complex-model hype and keep a strong MF baseline. Alongside that: **scikit-learn and API documentation** for implementation, and **empirical probing** for the parts with no documentation (I discovered the LeetCode public GraphQL queries by directly testing the endpoint). Above all I learned by **iterative prototyping** — building the end-to-end pipeline and letting failures (403 blocks, low method-tag F1) direct what I needed to learn next.

---

## Q5 — Alternative approaches (compare your solution with TWO routes you could have taken)

**My chosen route:** a **cross-platform, content-based, knowledge-gap recommender** built on a curated unified dataset, with NLP tag transfer to fill AtCoder's missing metadata — evolving toward a content+collaborative **hybrid** for the final system.

**Alternative 1 — Single-platform collaborative filtering (Codeforces-only), like A4 [3].**
I could have used only the Codeforces Open Dataset (17.6M submissions) with matrix factorisation or a Random Forest to predict solvable problems.
- *Advantages of that route:* dense interaction data (CF only), no cross-platform difficulty-normalisation or tagging problems, and a proven precedent.
- *Disadvantages:* it forfeits the **actual research gap** (no cross-platform tool exists); it inherits CF's cold-start problem; and pure CF **reinforces existing strengths** ("problems like ones you solved") rather than targeting weaknesses, which is wrong for an educational goal.
- *Why I chose mine:* the cross-platform gap is genuinely novel and the educational objective demands **weakness-targeting**, which my inverse-solve-rate `topic_gap` term does directly.

**Alternative 2 — Deep knowledge tracing with RNNs (Piech et al. [13]).**
I could have modelled each user's submission history as a temporal sequence with an RNN to predict mastery and recommend accordingly.
- *Advantages:* captures learning dynamics over time and is state-of-the-art on some EDM benchmarks.
- *Disadvantages:* it needs **long per-user sequences** (LeetCode's ~200-submission cap alone makes this fragile), it is **poorly interpretable** (users can't see *why* a problem is recommended), it **overfits on small data**, and Dacrema et al. [24] plus Wilson et al. [25] show such complex models often only match well-tuned simpler ones.
- *Why I chose mine:* **interpretability** is a first-class requirement here ("targets your weak topic: dynamic programming"), my data is sparse, and I can lean on platform Elo ratings as priors instead of learning skill from scratch.

**Research I did to make this choice.** A 26-source literature review spanning recommender foundations, EDM/knowledge-tracing, and programming-education recommenders; a survey of existing datasets (confirming **no cross-platform dataset exists**); direct investigation of all three platform APIs; and a prototype evaluation validating that the content-based cross-platform approach produces sensible recommendations for real users before committing to it.

---

## Quick-recall cheat sheet (last-minute)

- **The story arc:** heterogeneous platforms → unified dataset → AtCoder tag gap → NLP transfer → cross-platform profiles → gap-targeted recommender → (next) hybrid + formal eval.
- **Three moments:** AtCoder no-tags (→NLP), 403 scraping wall (→CodeNet fallback), LeetCode GraphQL discovery (→3 platforms).
- **Two honest weaknesses:** qualitative-only eval; brittle data access.
- **Two alternatives rejected:** single-platform CF (no novelty, reinforces strengths); DKT/RNN (uninterpretable, data-hungry).
- **Three refs:** Koren [5] high · kenkoooo [17] medium · A4 [3] low.
