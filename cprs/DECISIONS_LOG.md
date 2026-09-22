# CPRS — Decisions & Discussion Log

## 2026-05-24

### Q: Are there existing datasets for a competitive programming recommendation system?

**Key findings:**
- **CF Open Dataset (End of 2024)** — 17.6M submissions from ~15K Codeforces users. Richest user-problem interaction data. Codeforces-only.
- **CF User Practice and Rating History (Kaggle)** — user practice patterns + rating progression.
- **CF Problems Category Dataset (Kaggle)** — structured problemset with category, difficulty, participation metrics.
- **LeetCode Problem Dataset (Kaggle)** — ~1,825 problems with metadata.
- **LeetCodeDataset (arXiv)** — 2,869 problems with rich metadata.
- **IBM Project CodeNet** — 13.9M submissions, 4,053 problems (mainly AtCoder), 50+ languages.
- **AtCoder on Zenodo** — 4.5M submissions, 1,481 problems, 82K users.
- **No existing dataset combines user histories across platforms.**

Existing similar projects: LeetPath (graph-based), SmartCode (LeetCode scraper + recommender), CF ML Recommender (Random Forest, predicts rating + recommends by weakest topics).

---

### Q: Does the CF Open Dataset include problems from other platforms?

**A:** No, it's Codeforces-only. Multi-platform coverage requires combining separate datasets:
- IBM Project CodeNet → AtCoder, Aizu
- CodeContests (DeepMind) → AtCoder, CF, CodeChef, HackerEarth
- LeetCode datasets are all standalone

---

### Q: How will recommendations actually work? Is it complex enough?

**Discussion:**

The system requires 6 engineering layers:
1. Data collection (API clients for CF, LC, AtCoder)
2. Data processing (ETL, normalization, unified schema)
3. Database (problems, users, submissions, models)
4. Recommendation engine (ML models, scoring, ranking)
5. Backend API (REST endpoints)
6. Frontend (user dashboard, problem browser)

Recommendation approaches considered:
- **Content-based filtering** — match problem features to user skill gaps
- **Collaborative filtering** — "users with similar profiles solved these next"
- **Knowledge tracing** — model mastery per topic over time (BKT, DKT)
- **Hybrid** — combine multiple approaches (required for 2:1+ grade)
- **Sequence-aware** — optimal ordering for learning

Cross-platform complexity adds novelty:
- Difficulty normalization across platforms
- Unified tag taxonomy
- User skill modelling from heterogeneous submission data

**Conclusion:** More than complex enough. Risk is over-scoping, not under-scoping.

---

### Q: What are most common student recommendation system projects?

**A:** Most common (well-trodden ground):
1. Movie recommendations (MovieLens) — most overdone
2. Book recommendations (Goodbooks-10K)
3. Music recommendations (Spotify API)
4. E-commerce / product recommendations (Amazon reviews)
5. Course / learning content recommendations

**Why CPRS stands out:**
- No ready-made dataset (you curate one — 1st-class criteria)
- Natural skill progression model (unlike subjective movie/book preference)
- Cross-platform unification is a real unsolved problem
- Measurable outcomes (did user solve the problem? did rating improve?)
- No existing end-to-end solution to follow

---

### Open question: Which project template to submit under?

Options discussed:
- **Data Science (1.1)** — emphasizes ML, evaluation metrics, data pipeline
- **Advanced Web Design (7.2)** — emphasizes API design, web app, user testing

Decision: **TBD** — affects what examiners prioritize.

---

### Q: What exactly do the official documents require? Is there a rubric?

**A:** Extracted verbatim requirements to `REQUIREMENTS.md`. Key takeaways:

**Course level (CM3070):**
- 6 learning outcomes examiners assess against (propose, research, design, develop, test/evaluate, report)
- Final report + code = 60% of grade (150 hours expected)
- Written exam = 20%
- Preliminary report at week 10 = 10%
- Must produce: literature review, design, working software, evaluation, report, demo video

**Template level (1.1 Data Science — Recommendation System):**
- Final product: "a data-driven recommendation engine, likely implemented as a Python API or a web service"
- Prototype: "data ingestion, preprocessing, model training, and recommendation generation" — "A complete user interface is not essential at this stage"
- Must evaluate using: precision/recall, RMSE, AUC, cross-validation, statistical significance
- Must produce: trained models, recommendation scores, user profiles, performance visualisations, processed dataframes

**Grading rubric (verbatim from template):**
- **3rd (pass):** basic CF algorithm (e.g. matrix factorisation), public dataset, single metric, clear docs
- **2:1 (good):** hybrid model, preprocessing + feature engineering, multiple metrics + cross-val, strengths/weaknesses analysis
- **1st (outstanding):** novel approach or SOTA adaptation, curated high-quality dataset, baseline comparisons + statistical significance, detailed analysis, potentially publishable

**Key insight:** The template explicitly says "a complete user interface is not essential" for the prototype. The emphasis is on the data science pipeline and evaluation, not web dev polish.

---

## 2026-06-28

### Decision: Project template confirmed as 1.1 (Data Science)

Template 1.1 is the right fit — the project is fundamentally about ML/data pipelines, not web design. Examiners will prioritize the recommendation engine, evaluation metrics, and dataset quality.

---

### Dataset Build: Cross-Platform Unified Dataset (v1)

**Action:** Built fetchers for all 3 platforms and a normalization pipeline.

**Data collected:**

| Platform   | Problems | With Difficulty | With Tags |
|------------|----------|-----------------|-----------|
| Codeforces | 11,263   | 10,979 (97%)    | 11,083 (98%) |
| AtCoder    | 9,095    | 4,699 (52%)     | 2,926 (32%) |
| LeetCode   | 3,977    | 3,977 (100%)    | 3,857 (97%) |
| **Total**  | **24,335** | **19,655**    | **17,866** |

**Normalization applied:**
1. **Difficulty** — CF 800–3500, AtCoder -500 to 4000, LC Easy/Medium/Hard all mapped to [0.0, 1.0]
2. **Tags** — 50+ platform-specific tags mapped to a unified taxonomy of ~35 canonical topics
3. **IDs** — unified format: `cf:123A`, `ac:abc001_a`, `lc:42`

**Files produced:**
- `data/raw/` — raw API responses (cf_problems.json, ac_problems.json, lc_problems.json, ac_models.json)
- `data/cprs_unified.csv` — unified dataset, flat format
- `data/cprs_unified.json` — unified dataset, full fidelity

---

### Challenge: AtCoder problems have no topic tags

**Problem:** The kenkoooo AtCoder API provides difficulty estimates but **zero topic tags**. AtCoder itself does not categorize problems by algorithm/topic on its platform. This means 9,095 problems (37% of the dataset) lack topic metadata, making tag-based recommendations impossible for that slice.

By contrast, Codeforces has rich human-curated tags (98% coverage) and LeetCode has community-curated topic tags (97% coverage).

**Impact:** Without tags, AtCoder problems can only be used for difficulty-based recommendations, not topic-based ones. This limits cross-platform topic analysis.

**Options considered:**
1. **NLP auto-tagging** — scrape AtCoder problem statements, train a classifier on CF/LC tagged problems, predict tags for AtCoder. Adds a genuine NLP contribution (listed as required technique in template). Most technically impressive option.
2. **Community mappings** — some resources (e.g. AtCoder Tags by kenkoooo, community spreadsheets) partially map problems to topics. Spotty coverage.
3. **Accept the gap** — document as a dataset limitation; use AtCoder for difficulty-only recommendations.

**Decision: Implemented option 1 — NLP auto-tagging.** See entry below.

---

### NLP Auto-Tagger: Cross-Platform Tag Transfer

**Approach:** Train a multi-label text classifier on Codeforces problem statements (which have human-curated tags), then predict tags for AtCoder problems (which have none).

**Pipeline:**
1. Scraped 1,999 CF problem statements (training data with tags)
2. Scraped 795 AtCoder problem statements (inference targets — 4,205 blocked by 403)
3. TF-IDF vectorization (10K features, bigrams, sublinear TF)
4. OneVsRest Logistic Regression (balanced class weights)
5. 5-fold cross-validation for evaluation
6. Predict tags for AtCoder problems

**Cross-Validation Results (5-fold, on 1,999 CF problems):**

| Tag                 | Support | Precision | Recall | F1    |
|---------------------|---------|-----------|--------|-------|
| strings             | 206     | 0.640     | 0.864  | 0.736 |
| game_theory         | 54      | 0.667     | 0.815  | 0.733 |
| math                | 844     | 0.669     | 0.720  | 0.694 |
| greedy              | 868     | 0.646     | 0.680  | 0.663 |
| implementation      | 709     | 0.579     | 0.598  | 0.588 |
| bitmask             | 81      | 0.750     | 0.407  | 0.528 |
| constructive        | 417     | 0.465     | 0.556  | 0.507 |
| number_theory       | 175     | 0.521     | 0.491  | 0.506 |
| dfs                 | 53      | 0.529     | 0.340  | 0.414 |
| sorting             | 292     | 0.309     | 0.408  | 0.352 |
| data_structures     | 132     | 0.243     | 0.280  | 0.261 |
| brute_force         | 399     | 0.261     | 0.238  | 0.249 |
| binary_search       | 127     | 0.246     | 0.244  | 0.245 |
| dynamic_programming | 187     | 0.241     | 0.171  | 0.200 |
| two_pointers        | 100     | 0.188     | 0.180  | 0.184 |
| combinatorics       | 54      | 0.182     | 0.074  | 0.105 |
| **Micro avg**       |         |           |        | **0.530** |
| **Macro avg**       |         |           |        | **0.435** |

**AtCoder Prediction Results:**
- Tagged 775/795 scraped problems (97.5%)
- Average 1.9 tags per problem
- Top predicted: implementation (561), math (281), greedy (127), strings (88)

**Analysis — why some tags work better than others:**
- **High F1 tags** (strings, game_theory, math): These topics have distinctive vocabulary in problem statements ("string", "game", "integer", "divisor"). The text strongly signals the topic.
- **Low F1 tags** (dynamic_programming, two_pointers, binary_search): These describe *solution techniques*, not *problem content*. A DP problem might describe a grid, a sequence, or a game — the text doesn't reliably indicate the approach. This is a fundamental limitation of text-based tagging.
- **Implication:** Text-based auto-tagging works well for content-oriented tags but poorly for method-oriented tags. A more advanced approach would combine text features with structural features (difficulty, solve rate, editorial analysis).

**Challenge: AtCoder 403 blocks**
- Only 795/5,000 AtCoder problem pages were accessible (16% success rate)
- All failures were HTTP 403 — AtCoder blocks automated scraping
- The successfully scraped problems were mostly from recent ABC contests (newer pages)
- **Improvement for final project:** Use cached datasets (IBM CodeNet has AtCoder problem statements) or implement request throttling with session cookies

**Files produced:**
- `data/statements/cf_statements.json` — 1,999 CF problem statements
- `data/statements/ac_statements.json` — 795 AtCoder problem statements
- `data/ac_predicted_tags.json` — predicted tags with confidence scores for each AC problem
- `data/tagger_evaluation.json` — per-tag classification metrics
- `data/cprs_unified_tagged.json` / `.csv` — updated dataset with predicted tags

---

### Challenge: Difficulty scale differences across platforms

**Problem:** Each platform uses a completely different difficulty representation:
- **Codeforces:** integer rating 800–3500 (fine-grained, ~28 levels)
- **AtCoder:** float difficulty -∞ to +∞, estimated by kenkoooo's model (not official)
- **LeetCode:** 3 categorical levels (Easy, Medium, Hard)

These are not directly comparable. A "1500" on CF doesn't mean the same as "1500" on AtCoder.

**Solution:** Min-max normalization to [0, 1] with platform-specific bounds:
- CF: (rating - 800) / (3500 - 800)
- AtCoder: (difficulty - (-500)) / (4000 - (-500)), clipping outliers
- LeetCode: Easy → 0.2, Medium → 0.5, Hard → 0.85

**Limitations acknowledged:**
- This assumes difficulty distributions are roughly comparable across platforms, which is an approximation
- AtCoder's difficulty estimates are model-derived (kenkoooo), not official
- LeetCode's 3 levels are very coarse — a hard Easy and an easy Medium may be closer than their normalized values suggest
- A more principled approach would be to use problem solve rates or IRT models to calibrate cross-platform difficulty

**For the report:** This is a key design decision to discuss in the Design chapter. The normalization enables cross-platform recommendations but introduces assumptions that should be validated.

---

### Tagger Calibration: Determining Confidence Threshold

**Goal:** Determine the minimum confidence score at which a predicted tag should be shown to users. Too low = noisy, too high = too few tags.

**Method:** 5-fold cross-validation on 1,999 CF problems (ground truth tags), measured precision/recall/F1 at each threshold level.

**Global Threshold Analysis:**

| Threshold | Precision | Recall | F1    | Avg Tags/Problem | % Probs Tagged |
|-----------|-----------|--------|-------|------------------|----------------|
| 0.3       | 0.297     | 0.899  | 0.446 | 7.1              | 100%           |
| 0.4       | 0.391     | 0.746  | 0.513 | 4.5              | 100%           |
| **0.5**   | **0.514** | **0.536** | **0.525** | **2.5**     | **99%**        |
| 0.6       | 0.647     | 0.342  | 0.448 | 1.2              | 82%            |
| 0.7       | 0.759     | 0.172  | 0.280 | 0.5              | 45%            |
| 0.8       | 0.823     | 0.059  | 0.109 | 0.2              | 16%            |

**Decision: Global threshold = 0.5** (best F1). At this threshold:
- Half the predictions are correct (51.4% precision)
- Half the true tags are found (53.6% recall)
- 99% of problems get at least one tag
- Average 2.5 tags per problem

**Per-Tag Optimal Thresholds:**

| Tag                 | Best Threshold | Precision | Recall | F1    |
|---------------------|----------------|-----------|--------|-------|
| strings             | 0.55           | 0.657     | 0.854  | 0.743 |
| game_theory         | 0.60           | 0.719     | 0.759  | 0.739 |
| math                | 0.45           | 0.615     | 0.825  | 0.704 |
| greedy              | 0.40           | 0.565     | 0.854  | 0.680 |
| implementation      | 0.45           | 0.516     | 0.702  | 0.595 |
| bitmask             | 0.45           | 0.691     | 0.469  | 0.559 |
| number_theory       | 0.55           | 0.603     | 0.434  | 0.505 |
| constructive        | 0.45           | 0.412     | 0.638  | 0.500 |
| sorting             | 0.45           | 0.299     | 0.568  | 0.392 |
| dfs                 | 0.50           | 0.486     | 0.321  | 0.386 |
| brute_force         | 0.35           | 0.219     | 0.875  | 0.350 |
| data_structures     | 0.40           | 0.211     | 0.538  | 0.303 |
| two_pointers        | 0.45           | 0.198     | 0.330  | 0.247 |
| combinatorics       | 0.35           | 0.205     | 0.315  | 0.248 |
| binary_search       | 0.40           | 0.165     | 0.409  | 0.235 |
| dynamic_programming | 0.35           | 0.143     | 0.610  | 0.232 |

**Key Insight — Two Tiers of Tags:**

*Tier 1 — Content tags (F1 > 0.5):* strings, game_theory, math, greedy, implementation, bitmask, number_theory, constructive. These describe what the problem is *about* — the text contains distinctive vocabulary ("string", "game", "prime", "binary").

*Tier 2 — Method tags (F1 < 0.4):* dynamic_programming, binary_search, two_pointers, brute_force. These describe *how to solve it*, not what the problem says. A DP problem might describe a grid, a knapsack, or a sequence — the text doesn't reliably signal the approach.

**Confusion Analysis:** The most common confusion is between math/greedy/implementation/constructive — these are broad, overlapping categories. For example, many "constructive" problems are also "greedy" and "math". This reflects real ambiguity in the tagging conventions, not just classifier weakness.

**Recommendation for UI:**
- Show predicted tags at >= 0.5 confidence
- Optionally show confidence score (e.g. "math (0.82)", "greedy (0.53)")
- For critical decisions (e.g. study plan generation), use per-tag thresholds for higher accuracy
- Always label predicted tags as "predicted" vs "curated" so users know the source

---

### Challenge: User data accessibility varies by platform

**Investigation:** What user submission data can we access per platform?

| Platform     | API Endpoint                    | Data Available                              | Auth Required? |
|-------------|--------------------------------|---------------------------------------------|----------------|
| Codeforces  | `user.status`                  | Full history: problem, verdict, language, time | No — fully public |
| AtCoder     | kenkoooo `/v3/user/submissions`| Full history: problem, result, language, time  | No — public |
| LeetCode    | GraphQL                        | Very limited — solved count only, specific problems behind auth | Yes — session cookie |

**Impact:** We can build rich user profiles from CF and AtCoder submission history, but LeetCode user data is essentially inaccessible without the user's login credentials.

**Decision:** Profile building uses CF/AtCoder data. LeetCode problems are recommended based on cross-platform topic gap analysis (if user is weak at DP on CF, recommend DP problems from LC too). For the final project, a user could optionally provide their LC session cookie or manually import their solved list.

---

### Recommendation Engine: Content-Based Filtering Prototype

**Implemented:** Content-based recommender that scores problems using:

```
score = 0.4 * topic_gap + 0.4 * difficulty_fit + 0.1 * popularity + 0.1 * diversity
```

Where:
- **topic_gap**: weighted by inverse of user's solve rate per topic (weak topics score higher)
- **difficulty_fit**: gaussian centered slightly above user's current level (75th percentile of solved difficulties + 10% stretch)
- **popularity**: log-scaled solve count (prefer well-tested problems)
- **diversity**: small random factor to avoid monotonous recommendations

**User profile construction:**
1. Fetch all submissions via CF API
2. Track per-problem best verdict (solved/attempted)
3. Compute per-topic mastery (solve rate, avg difficulty, max difficulty)
4. Set difficulty level at 75th percentile of solved problems
5. Identify weak topics (low solve rate or never attempted)

**Tested with real users:**
- `tourist` (rating 3439, 3019 solved) — correctly recommends cross-platform problems in topics he hasn't seen on CF (array, hash_table, backtracking from LeetCode taxonomy)
- `jiangly` — similar cross-platform recommendations at appropriate difficulty

**Observation:** For top-rated users, "weak topics" are actually just tags from other platforms' taxonomy. This is the cross-platform value — discovering problems categorized differently across platforms.

---

### Web Application: Auth + Multi-Platform Profiles + Recent Performance

**Implemented:** Full-stack web application with:

**Backend (FastAPI):**
- User registration/login with hashed passwords (SHA-256 + salt), cookie-based sessions
- Platform handle management — users save their CF, AtCoder, (and eventually LC) handles
- Multi-platform profile building — fetches submissions from CF + AtCoder, merges into single profile
- Recent performance analysis — last N problems, solve rate, difficulty trend (improving/declining/stable)
- Two modes: quick mode (just pass a CF handle, no login) or full mode (logged in, multi-platform)

**Frontend (single-page HTML + vanilla JS):**
- Auth bar (register/login/logout)
- Platform handles settings panel
- Profile card with skill stats, topic mastery bars
- Recent performance section with per-platform breakdown and trend indicator
- Recommendation cards with platform badges, difficulty scores, topic tags (weak topics highlighted in yellow), direct links to solve on original platform

**Database (SQLite):**
- `users` — id, username, password_hash, salt
- `platform_handles` — user_id, platform, handle
- `sessions` — token, user_id

**Multi-platform profile merging:**
- Solved problem sets unioned across platforms
- Topic mastery merged (summing attempts/solves across platforms)
- Difficulty level computed as weighted average by solve count
- Rating taken as max across platforms

**Recent performance analysis:**
- Analyzes last N submissions (default 50) per platform
- Computes: solve rate, avg difficulty, most practiced topics
- Detects trend by comparing difficulty of recent half vs older half (>5% change = improving/declining)

**"Proxy" to judging systems:**
- Each recommendation includes a direct URL to the problem on its native platform
- User clicks → opens CF/AtCoder/LC in new tab → solves there
- On next profile refresh, their new submissions are picked up via API
- No actual submission proxying needed — this is the practical approach

---

### LeetCode API — Discovering Public Endpoints

**Initial assumption:** LeetCode has no public API for user submission data, making cross-platform analysis impossible for LC users. We initially disabled LC handle input in the UI with "API limited — coming soon".

**Investigation:** Tested LeetCode's GraphQL endpoint (`https://leetcode.com/graphql`) and found two public queries that work without authentication:

1. **`userPublicProfile`** — returns solve counts by difficulty (Easy/Medium/Hard/All)
2. **`recentAcSubmissionList`** — returns recent accepted submissions with title, titleSlug, and timestamp

**Test results with user `lee215`:**
- Profile: 642 total solved (122 Easy, 381 Medium, 139 Hard)
- Recent submissions: successfully returned 20 most recent AC submissions with slugs

**Implementation:**
- Added `fetch_user_profile()` and `fetch_user_submissions()` to `fetchers/leetcode.py`
- Added `build_leetcode_profile()` to `RecommenderEngine` — matches LC submissions to our unified dataset via `titleSlug`, extracts tags and difficulty for topic mastery analysis
- Wired into `app.py` recommend endpoint alongside CF and AtCoder
- Enabled LC handle input in the UI (was previously disabled)

**Limitations:**
- `recentAcSubmissionList` only returns accepted submissions (no failed attempts), so LC solve rates are always 100% for matched problems
- The endpoint returns at most ~200 recent submissions, not the full history
- Not all LC problems in our dataset may match by slug (naming changes, premium problems)

**Decision:** These limitations are acceptable for a recommendation system. We get enough signal from recent solves + difficulty distribution to build a useful LC profile. The cross-platform value is significant — users can now link all three major platforms.

---

## 2026-09-14

### Preliminary report feedback received + final build plan

**Feedback (TA):** solid pass; strongest on literature/concept, weakest on evaluation and workplan feasibility. Concrete asks: table of contents; explicit gap→contribution mapping in lit review; design figures + a cold-start solution; detailed eval metrics + a dataset to evaluate recommender quality; visualise + discuss results (not just descriptive). A stray "0 out of 6" line is treated as a form artifact — it contradicts every per-criterion mark above it.

**Stance:** keep feedback in mind, address organically as we build; do not reshape the project just to chase one TA's rubric.

### Decision: final build scope

- **Ambition:** aim for 1st class from the start.
- **Evaluation:** offline-only on real interaction data (no user study).
- **First thrust:** hybrid recommender (collaborative filtering + existing content-based) — chosen as highest grade-per-effort; also forces a genuine cold-start story.
- **Novelty framing:** cross-platform hybrid + rigorous comparative evaluation on the already-curated cross-platform dataset. RNN/sequence model is a stretch goal, not critical path.
- **Timeline:** ~1 week of everyday working days; must cover build + final report + MP4 demo.
- **#1 risk identified:** no user×problem interaction dataset on disk (only problem catalogues). Collaborative filtering needs it → Day 1 acquires the CF Open Dataset, with API-fetch and sub-sampling contingencies.

Full day-by-day plan: `cprs/FINAL_PROJECT_PLAN.md`.

### Next action: exam preparation before build execution.

---

## 2026-09-19

### Final rubric (v2) received + plan re-baselined against it

**New artifact:** `final_report_rubric_v2.pdf` — the official per-criterion grading breakdown for the final report + video (far more specific than the preliminary rubric). Point weights make priorities explicit: Implementation quality (22), Design (12), and Diagrams (10) are the fattest buckets; the evaluation cluster (strategy 6 + coverage 5 + results 5 + analysis 4 = 20) and Originality (10) are almost entirely unclaimed until the eval harness exists. Full mapping table in `FINAL_PROJECT_PLAN.md`.

### Q: Why do we actually need a user×problem interaction dataset? (raised during planning)

**A:** It is the linchpin for ~40+ of the ~120 report points:
1. **Collaborative filtering is a function of a user×problem matrix.** Content-only work is capped at "3rd (pass)" by the template rubric; a hybrid (CF + content) is the named "2:1" requirement. No matrix → no CF → no hybrid.
2. **Offline evaluation is impossible without held-out real solves.** Every ranking metric (HitRate@K, nDCG, MRR, Precision/Recall@K) hides a user's recent solves and checks whether the model predicts them back. Without interaction histories there is nothing to score against → the ~20-point evaluation cluster stays at zero.
3. It also lets us evaluate the *existing* content-based model properly (as a baseline) instead of anecdotally.

**Decision:** interaction data is on the critical path, not optional. Source = **download CF Open Dataset (primary)**; contingency = API-fetch `user.status` for ~2–5K sampled handles using existing fetchers; sub-sample to keep the matrix tractable.

### Decision: sequence/RNN model stays a cut-first stretch goal

In a 1-week window that also produces the report + demo, committing to an RNN risks the critical path for marginal originality gain (originality is already banked by the curated cross-platform dataset + comparative cross-platform evaluation). Build only if Days 1–4 finish clean.

### Deadline re-confirmed: ~1 week of working days (target ~2026-09-27)

Plan tightened to 7 working days covering build + report + demo; cut order and "never cut" list recorded in `FINAL_PROJECT_PLAN.md`.

---

## 2026-09-22

### Day 1 executed: cross-platform interaction dataset built

Took the API-fetch route (plan Contingency A) over the CF Open Dataset download — self-sufficient, deterministic, reuses existing fetchers, and downloads/Kaggle-auth aren't reliable unattended. Sampled 3,000 rated CF handles (seed 42) via `user.ratedList`, fetched full submission histories, kept accepted solves for problems in the unified dataset. After k-core filtering (≥5 solves/user, ≥5 solvers/problem): **2,459 users × 7,568 problems, 532,059 interactions, 2.86% density** (high for implicit feedback — good CF signal). Leave-last-N temporal split (max 10 / 20% per user) → 512,436 train / 19,623 test solves across all users. Raw `cf_submissions.jsonl` committed (frozen for reproducible evaluation; CF histories drift). Scripts: `fetch_interactions.py`, `build_interactions.py`, `split_interactions.py`.

### Day 2 executed: collaborative filtering + evaluation harness

- `implicit` (ALS) installed cleanly (prebuilt arm64 wheel); SVD fallback wired in `models/collaborative.py` so the pipeline never hard-depends on a native build.
- `models/content.py`: deterministic, vectorized offline content scorer (mirrors the deployed engine's topic-gap + difficulty-fit + popularity formula) — serves as a fair content-only baseline and the Day-3 hybrid ingredient.
- `scripts/evaluate.py`: harness computing HitRate/Precision/Recall/MRR/nDCG @{5,10,20} over the held-out temporal test, masking already-solved train items.

**Results (HitRate@10 / nDCG@10):**

| Model | HitRate@10 | nDCG@10 |
|---|---|---|
| random | 0.011 | 0.001 |
| content | 0.009 | 0.001 |
| popularity | 0.144 | 0.037 |
| **cf (ALS)** | **0.295** | **0.096** |

**Key finding:** CF wins decisively — ~2× popularity, ~26× random. **Content-based performs at ~chance for next-solve prediction** — verified as a genuine effect, not a bug (content deliberately recommends *unseen* topics at *stretch* difficulty, which anti-correlates with users' tendency to continue in familiar topics). This is the core motivation for the hybrid: content optimizes pedagogical spread, CF optimizes next-action likelihood — they answer different questions. CF recommendations also lean slightly toward recent, popular problems (expected, since the temporal test holds out recent solves). Metrics saved to `data/eval_results.json`; significance testing deferred to Day 5.

### Day 3 executed: hybrid model + cold-start — and a narrative-changing finding

Built `models/hybrid.py` (per-user min-max normalized, density-weighted blend of CF with a content/popularity "cold" ingredient; `alpha = density/(density+k0)`), added `hybrid` + `hybrid_pop` to the harness, and sliced evaluation by user-activity bucket.

**Finding 1 — on users with history, the hybrid cannot beat pure CF.** Overall nDCG@10: cf 0.096, hybrid_pop 0.090, hybrid(content) 0.081. Blending content *hurts* (it's at chance); blending popularity dilutes CF. This holds in **every** activity bucket, including low-activity [5,20) (cf 0.207 vs hybrid 0.143). Reason: the matrix is filtered to ≥5 solves, so it contains **no true cold-start users** — CF already has enough signal for everyone in it.

**Finding 2 — the hybrid's value is real but *only* at genuine cold-start**, shown by a fold-in simulation (`scripts/coldstart_sim.py`, 1,976 probe users, reveal only k most-recent solves, ALS fold-in against fixed item factors). nDCG@10 vs history size k:

| model | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|---|---|
| cf | 0.001 | 0.135 | 0.132 | 0.124 | 0.114 | 0.091 | 0.071 |
| popularity | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 | 0.011 | 0.012 |
| hybrid_pop (k0=10) | 0.010 | 0.014 | 0.019 | 0.026 | 0.038 | 0.065 | 0.069 |

At **k=0** (brand-new user) CF collapses to ~random (0.001) and popularity wins 10×; from **k=1 onward CF fold-in dominates** (a single solve is enough to place a user usefully in latent space). Higher-history users have *lower* nDCG because they've exhausted popular problems and their next solves are rarer/harder.

**Evidence-driven design conclusion:** the optimal cold-start policy is a **near-hard switch** — popularity for zero-history users, CF from the first interaction — not a slow smooth ramp. The k0=10 smooth blend is miscalibrated (under-weights CF at small k, where CF is already excellent). Recommended production model: **CF for any user with ≥1 solve; popularity fallback only at zero history.**

**Reframing the "novel contribution":** the data does not support "hybrid beats everything." The defensible headline is **cross-platform CF + a rigorous evaluation that characterises exactly when each method works (incl. a validated cold-start policy)**. The remaining originality lever is the **cross-platform value analysis** (Day 6: does merging AtCoder/LeetCode history improve CF recommendations vs single-platform?), which is untouched and independent of this finding.

### Decision: cross-platform value is the headline thrust (user choice, 2026-09-22)

Asked which direction to prioritise; user chose **cross-platform value** over "strengthen the hybrid" or "evaluation rigor as star." This yields a **unifying thesis that connects Day 3 and Day 6**:

> **Cross-platform history is a cold-start remedy.** Day 3 showed CF collapses for zero-history users. A user new to Codeforces but experienced on AtCoder/LeetCode is only "cold" if you look at one platform — merging their cross-platform profile makes them warm immediately. So the cross-platform contribution *is* the principled answer to the cold-start problem the evaluation exposed.

The hybrid is reframed as the cold-start **mechanism** (personalized once ≥1 signal exists, on any platform); cross-platform data is what *supplies* that signal for users cold on the target platform. Next action: acquire a multi-platform cohort by handle-matching the 3,000 CF users against AtCoder (kenkoooo), then compare CF-only vs cross-platform recommendation quality.

### Cross-platform value — thorough investigation, honest (mostly negative) result

Handle-matched the 3,000 CF users against AtCoder (kenkoooo); probe stopped at 80% (2,410 probed) → **392 cross-platform users, ~30K AtCoder solves** (220 with ≥10 AC solves). Built a merged CF+AtCoder matrix (2,459 × 10,239; 2,671 AtCoder problems with ≥3 cohort solvers). Tested three transfer mechanisms on the unchanged CF held-out test:

1. **Joint ALS (AtCoder as extra columns), fold-in with CF+AC items — robustly *hurts*.** Design B: cross 0.033 vs cf_only 0.089 nDCG@10 at k=1. AtCoder problems are solved only by the small cohort, so their latent factors sit in a subspace weakly aligned with CF; folding them in drags the user vector away from the CF-relevant region. Naive joint MF does not transfer across disjoint problem sets.

2. **AtCoder-neighborhood transfer (cosine over AtCoder solves → recommend neighbors' CF solves) — weak positive.** For proxy-cold users it beats the popularity baseline ~2× (0.0053 vs 0.0027 nDCG@10) using *no* CF history — but a single native CF solve (0.089) is ~17× better still.

3. **Decisive test on genuinely CF-cold users — the target population barely exists.** Users held out of training with few CF solves ([3,20]) *and* real AtCoder history (≥10): **zero** of 247 held-out users qualify (only 2 have ≥5 AC solves). **Structural finding:** handle-matched cross-platform users are experienced on *both* platforms — the "new to CF, veteran on AtCoder" user the thesis targets is empirically rare.

**Honest conclusion:** cross-platform *collaborative* transfer offers limited practical value here, for two compounding reasons — disjoint problem sets weaken latent transfer, and the cold-start-target population is rare because multi-platform users are broadly experienced. This is a legitimate, rigorous negative result (good for "critical evaluation"), but it is **not** the triumphant positive headline the thrust assumed. Scripts: `build_crossplatform.py`, `eval_crossplatform.py`, `eval_coldstart_xplat.py`. Results in `data/xplat_results.json`, `data/coldstart_xplat_results.json`.

**Decision pending (user):** reframe the project's headline contribution around what is robustly strong — the curated cross-platform dataset, the NLP auto-tagger, and a rigorous comparative evaluation with several well-explained (incl. counterintuitive) findings — rather than a cross-platform hybrid that wins. See next Q&A.

### BREAKTHROUGH: cross-platform value is in DIFFICULTY/SKILL transfer, not collaboration

User's insight: bet on unifying **difficulty + category** and *assume solvability from difficulty-vs-skill, correcting live* — rather than collaborative transfer. Two checks:

1. **Difficulty-match ranking is weak but validates the direction.** A difficulty-matched, *familiar*-category scorer (`ContentScorer.score_all_match`: at-level difficulty + topics the user already practises) scores nDCG@10 0.0061 — **5× the old pedagogical content scorer** (0.0012), confirming "at-level + familiar" > "stretch + unseen". Still below popularity/CF as a *next-item* predictor (predicting the exact next problem among hundreds at the right level is intractable) — but next-item ranking is the wrong metric for this idea.

2. **The unified difficulty scale is strongly validated.** For 186 users with ≥10 difficulty-bearing solves on both CF and AtCoder, **CF-skill vs AtCoder-skill correlates at Pearson r=0.773 (p≈4e-38), Spearman ρ=0.745**, with matched means (0.247 vs 0.260). Skill is a property of the person and the normalized difficulty makes it comparable across platforms.

**This is the positive cross-platform result.** Collaborative transfer fails (disjoint problem sets), but **difficulty/skill transfer works** — enabling cross-platform *difficulty calibration*: a user cold on CF but active on AtCoder has a known skill level, so we can recommend appropriately-hard CF problems immediately. That is the cross-platform cold-start remedy that survives the data (unlike collaborative transfer). The "correct live" idea becomes an **online adaptive skill estimate** grounded in this validated scale.

**Synthesised contribution (proposed):** CF ranks *which* problems; the validated cross-platform difficulty scale calibrates *at what level* and supplies a skill prior for cold users; an online skill update adapts as outcomes arrive. Cross-platform is now *essential* (normalized difficulty is what places an AtCoder user on the CF scale) and *validated* (r=0.77).

### Adaptive cross-platform skill model built + evaluated — the positive cross-platform result

`models/skill.py`: online skill estimator on the unified difficulty scale — a Robbins-Monro quantile tracker (`theta += lr*(q - 1[b<theta])`), O(1) per solve, self-limiting on solve-only data (no wins-only Elo drift), seedable from any platform. `scripts/eval_adaptive.py` tests it.

**(A) Cross-platform cold-start skill estimation (n=184), MAE to a user's future CF skill (75th-pct difficulty):**

| predictor | MAE | vs population |
|---|---|---|
| population prior | 0.111 | — |
| cf_1 (one CF solve) | 0.211 | −90% (worse) |
| cf_5 | 0.177 | −59% (worse) |
| **atcoder (0 CF solves)** | **0.072** | **+35%** |
| **atcoder + cf_5 (seed + live-correct)** | **0.066** | **+40%** |

**A brand-new Codeforces user's skill is estimated 35–40% more accurately from their AtCoder history than from the population prior — and better than their first several CF solves** (a few individual problem difficulties are noisy; AtCoder history is more data). Seed-then-correct (`atcoder+cf_k`) is best, realizing "seed cross-platform, correct live."

**(B) Online convergence (n=1524):** live estimate MAE falls monotonically 0.108 (1 solve) → 0.081 (20 solves) — it corrects as evidence accrues.

**This is the validated, positive cross-platform contribution:** collaborative transfer fails across disjoint problem sets, but **difficulty/skill transfer works** and is directly useful for cold-start skill calibration and difficulty-appropriate recommendation. Results in `data/adaptive_results.json`. Next: consolidation — significance tests, figures, report chapters, demo.
