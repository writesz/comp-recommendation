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
