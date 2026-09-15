# CPRS: A Cross-Platform Competitive Programming Recommendation System

**CM3070 Computer Science Final Project — Preliminary Report**

**Template: 1.1 Data-Driven Personalised Educational Content Recommendation**

---

## Chapter 1: Introduction

### 1.1 Motivation

Competitive programming has grown into a global educational practice, with platforms such as Codeforces, AtCoder, and LeetCode collectively hosting millions of problems and serving tens of millions of users [1]. These platforms serve dual purposes: as training grounds for algorithmic thinking and as standardised assessment tools widely used in technical hiring [2]. However, the sheer volume of available problems creates a significant discovery problem — users must manually identify which problems to solve next, often relying on trial-and-error or community advice rather than systematic, personalised guidance.

The challenge is compounded by platform fragmentation. Each platform operates independently with its own difficulty scale, tag taxonomy, and user ecosystem. A user practising on Codeforces receives no guidance about complementary problems on AtCoder or LeetCode that could address their specific skill gaps. This isolation means users miss opportunities for targeted practice, and the rich data generated on one platform is invisible to recommendations on another.

Current approaches to problem selection fall into three categories: platform-curated lists (e.g., LeetCode's "Top Interview Questions"), community-driven spreadsheets (e.g., the "CSES Problem Set" or "A2OJ Ladders"), and informal peer advice on forums such as Codeforces blogs. None of these are personalised to individual skill profiles, and all are confined to a single platform. A user who primarily trains on Codeforces but also uses LeetCode for interview preparation currently has no tool that analyses their performance holistically and recommends problems across both platforms.

This gap is particularly acute for users at the intermediate level (Codeforces rating 1200–1800), who face the largest and most heterogeneous problem space. Beginners can follow curated introductory sets, and experts have enough experience to self-direct, but intermediate users must navigate thousands of problems across dozens of topics with little systematic guidance. Effective feedback and personalisation are critical for sustained learning [10], yet current platforms provide only aggregate statistics (rating, solve count) without actionable problem-level guidance.

### 1.2 Problem Statement

This project addresses the question: *Can we leverage data science techniques to build a personalised competitive programming problem recommendation system that operates across multiple platforms?*

Specifically, the system — named CPRS (Cross-Platform Competitive Programming Recommendation System) — must:

1. **Unify heterogeneous data** from Codeforces, AtCoder, and LeetCode into a single, normalised dataset with comparable difficulty ratings and a shared topic taxonomy.
2. **Build user skill profiles** from submission histories across platforms, identifying per-topic mastery levels and difficulty thresholds.
3. **Generate personalised recommendations** that target identified skill gaps at appropriate difficulty levels, drawing from all three platforms.
4. **Handle missing metadata** — notably, AtCoder problems lack topic tags entirely — using NLP techniques to transfer knowledge from tagged platforms.

### 1.3 Project Concept and Justification

The competitive programming domain offers several advantages over more common recommendation system projects (e.g., movie or product recommendations):

- **Objective outcomes**: unlike subjective preferences (film ratings), problem-solving has a binary ground truth — the user either solves the problem or does not. This enables rigorous evaluation without the ambiguity inherent in rating-based systems.
- **Structured skill progression**: algorithmic topics (graph theory, dynamic programming, number theory) form a learnable taxonomy with measurable mastery, unlike flat preference spaces. This structure allows recommendations to target specific knowledge gaps rather than generic "you might like this" suggestions.
- **No existing cross-platform solution**: while single-platform recommenders exist as community tools [3], no system unifies data across Codeforces, AtCoder, and LeetCode. This represents a genuine gap in both the academic literature and practical tooling.
- **Novel dataset curation**: building the unified cross-platform dataset is itself a significant data science contribution, requiring difficulty normalisation across incompatible scales, taxonomy unification across different tagging conventions, and NLP-based tag transfer for platforms lacking metadata.
- **Rich public data**: unlike many educational domains where learner data is private, competitive programming platforms provide public APIs for submission histories, enabling reproducible research without privacy concerns.

The target users are competitive programmers at all levels — from beginners choosing their first problems to experienced contestants seeking targeted practice in weak areas. The system is also relevant to students preparing for technical interviews, where LeetCode proficiency is routinely assessed [2]. A secondary audience includes competitive programming coaches and educators who could use CPRS to identify common skill gaps across their students and assign targeted practice sets.

### 1.4 Report Structure

Chapter 2 reviews the academic literature on recommendation systems, educational data mining, and competitive programming analytics. Chapter 3 presents the system design, including the data pipeline, recommendation algorithm, evaluation strategy, and workplan. Chapter 4 describes and evaluates the feature prototype.

---

## Chapter 2: Literature Review

### 2.1 Recommendation Systems: Foundations

Recommendation systems are broadly categorised into collaborative filtering (CF), content-based filtering (CBF), and hybrid approaches [4]. **Collaborative filtering** exploits patterns in user-item interaction matrices — if users A and B solved similar problems, problems solved by A but not B become candidates for B. Koren, Bell, and Volinsky [5] demonstrated that matrix factorisation techniques, particularly Singular Value Decomposition (SVD), significantly outperform neighbourhood-based CF methods, achieving a 10% RMSE improvement on the Netflix Prize dataset. However, CF suffers from the cold-start problem: new users with few interactions receive poor recommendations, and new items with no interactions are never recommended.

**Content-based filtering** recommends items similar to those a user has previously engaged with, based on item features [6]. In educational contexts, these features might include topic tags, difficulty level, and textual content. CBF avoids the cold-start problem for new items (any problem with metadata can be recommended immediately) but struggles with serendipity — it tends to recommend items too similar to what users already know, reinforcing existing strengths rather than addressing weaknesses.

Burke [7] proposed a taxonomy of **hybrid approaches** that combine CF and CBF to mitigate their individual weaknesses. The most relevant for educational recommendation are feature-augmented hybrids, where content features are fed into collaborative models. Rendle's Factorization Machines [8] formalised this by enabling arbitrary feature combinations within a factorisation framework, making them well-suited to sparse educational interaction data where user-problem matrices are extremely sparse (a typical user solves <0.1% of available problems).

A more recent direction is **deep learning-based recommendation**. Zhang et al. [16] provide a comprehensive survey, identifying that neural collaborative filtering (NCF), attention-based models, and graph neural networks have become dominant in academic benchmarks. However, they also note a reproducibility concern: many deep learning recommenders show marginal improvements over well-tuned matrix factorisation baselines when evaluated fairly, a finding corroborated by Dacrema et al. [24], who demonstrated that 7 of 18 deep learning recommendation papers they examined could not outperform simple baselines when properly tuned. This suggests that for a domain like competitive programming — where interaction data is sparse and interpretability matters — simpler models with domain-specific features may be more practical than complex neural approaches.

**Critique**: The foundational recommendation literature was developed primarily for entertainment domains (movies, music, e-commerce) where user preferences are subjective and static. Educational recommendation differs fundamentally: the goal is not to match preferences but to optimise learning, which requires modelling skill progression over time. A user who has mastered dynamic programming should *not* receive more DP problems simply because they enjoyed them — they should receive problems in weaker areas. This distinction necessitates specialised approaches beyond standard CF/CBF. Furthermore, Dacrema et al.'s [24] reproducibility findings suggest that the pursuit of complex models may be misguided when domain knowledge and feature engineering provide stronger signal.

### 2.2 Educational Data Mining and Knowledge Tracing

The field of Educational Data Mining (EDM) addresses this gap by modelling learner knowledge states. **Bayesian Knowledge Tracing** (BKT), introduced by Corbett and Anderson [9], models the probability that a student has mastered a skill as a hidden Markov model, updated with each practice opportunity. BKT has been widely deployed in intelligent tutoring systems, but its assumption of binary skill mastery (known/unknown) is a poor fit for competitive programming, where mastery exists on a continuous spectrum — a user might consistently solve easy graph problems but fail hard ones.

Pelánek [11] provides a comprehensive comparison of learner modelling techniques, including BKT, logistic models (e.g., Performance Factor Analysis), and Elo-based approaches. He argues that Elo-based models [12], originally developed for chess ratings, are particularly well-suited to competitive contexts because they naturally model relative difficulty and produce calibrated skill estimates. Codeforces already uses an Elo-derived rating system, and AtCoder uses a similar approach, suggesting that Elo-based thinking is natural for this domain.

**Deep Knowledge Tracing** (DKT), proposed by Piech et al. [13], applies recurrent neural networks (RNNs) to model learning trajectories as sequences. DKT outperformed BKT on several benchmarks by capturing complex temporal dependencies — for example, the effect of practice spacing on retention. However, DKT has been criticised for its lack of interpretability [14] and its tendency to overfit on small datasets. For a cross-platform recommendation system, interpretability matters: users should understand *why* a problem is recommended (e.g., "targets your weak topic: dynamic programming").

**Critique**: Knowledge tracing models assume a fixed, known set of skills (knowledge components). In competitive programming, the skill taxonomy is itself uncertain — different platforms use different tag sets, and a single problem may require multiple skills. The granularity problem is significant: "dynamic programming" encompasses dozens of sub-techniques (bitmask DP, digit DP, tree DP) that are not uniformly tagged across platforms. Our NLP auto-tagging approach (Section 4) addresses this by learning a unified taxonomy, but the underlying ambiguity remains a challenge for any knowledge-tracing approach.

Furthermore, Wilson et al. [25] argue that Elo-based models, while simpler than DKT, offer comparable predictive performance with far greater interpretability and computational efficiency. Their "Elo-based learner model" naturally handles the cold-start problem by assigning default ratings that converge quickly with practice. The Codeforces and AtCoder rating systems are themselves Elo-derived, suggesting that the competitive programming community has independently validated this modelling paradigm. Our design leverages this insight by using platform ratings as priors for user difficulty levels, rather than attempting to learn them from scratch.

### 2.3 Recommender Systems in Programming Education

Thai-Nghe et al. [15] applied matrix factorisation to predict student performance in programming courses, demonstrating that latent factor models outperform traditional regression when interaction data is sparse. Specifically, they showed that tensor factorisation — extending matrix factorisation with a temporal dimension — captured learning progression more effectively than static models, achieving a 12% reduction in prediction error. Their work establishes the viability of recommendation techniques for programming education, though it focused on structured course assignments with a fixed curriculum rather than the open-ended, self-directed practice environment of competitive programming.

Zhang et al. [16] survey deep learning approaches to recommendation, identifying that attention mechanisms and graph neural networks are particularly promising for educational contexts where item relationships (prerequisite structure) matter. In competitive programming, certain topics are prerequisites for others (e.g., basic graph traversal before network flow, prefix sums before range queries), creating a natural graph structure that could inform recommendation ordering. However, the prerequisite relationships in competitive programming are less rigid than in formal curricula — a user can often learn topics in multiple valid orderings — which limits the value of strict prerequisite enforcement.

Most directly relevant are community-built competitive programming recommenders. **A4** (A Codeforces Assistant) [3] uses a Random Forest classifier trained on user submission data to predict which problems a user can solve, then recommends problems just above their predicted ability. While effective for single-platform use, it does not generalise across platforms and relies entirely on Codeforces-specific features (contest ID patterns, problem rating distributions).

**LeetCode recommendation tools** such as LeetPath use graph-based approaches to suggest learning paths through LeetCode's problem set, modelling topic dependencies as directed acyclic graphs. These tools demonstrate user demand for structured guidance but operate exclusively within LeetCode's ecosystem.

Huang et al. [26] explored knowledge-driven recommendation for programming exercises in MOOCs, using a knowledge graph of programming concepts to guide recommendations. Their approach demonstrates the value of explicit concept modelling in educational recommendation, achieving a 15% improvement in learning outcomes over random assignment. However, their knowledge graph was manually constructed for a single course, making it difficult to scale to the breadth of competitive programming topics.

**Critique**: Existing competitive programming recommenders operate within single platforms, treating each platform's problem set in isolation. This misses the significant opportunity for cross-platform skill transfer — a user's Codeforces performance contains strong signal about which LeetCode or AtCoder problems would be most beneficial. Our project is, to our knowledge, the first to attempt unified cross-platform recommendation in this domain. Additionally, while knowledge graph approaches [26] show promise, the manual effort required to construct domain-specific graphs makes automated NLP-based tag inference (as used in CPRS) a more scalable alternative for the competitive programming domain, where the problem space is large and continuously growing.

### 2.4 Cross-Platform Data Challenges

Building a cross-platform dataset requires addressing three fundamental heterogeneity challenges: difficulty scales, topic taxonomies, and data accessibility.

**Difficulty normalisation**: Codeforces uses integer ratings (800–3500) derived from contest performance data, AtCoder uses model-estimated continuous difficulties [17] computed by the kenkoooo project using logistic regression on solve rates, and LeetCode uses three categorical levels (Easy, Medium, Hard) with no numerical granularity. Elo [12] established the theoretical foundation for relative difficulty estimation, but cross-platform calibration remains an open problem because the user populations differ — Codeforces users skew toward competitive contestants, while LeetCode users skew toward interview preparation. Our approach uses min-max normalisation with platform-specific bounds, which is a practical approximation; a more principled approach would use Item Response Theory (IRT) [18] to jointly estimate problem difficulty and user ability from cross-platform solve rates, but this requires a significant population of users who are active on multiple platforms simultaneously.

**Topic taxonomy unification**: Codeforces uses ~35 tags assigned by problem authors, LeetCode uses ~75 community-curated tags, and AtCoder provides no tags at all. Wasik et al. [1] note that even within a single platform, tagging is inconsistent — the same algorithmic technique may be tagged differently across problems, and some problems have incomplete or misleading tags. This inconsistency is amplified across platforms: Codeforces uses "dp" while LeetCode uses "dynamic-programming"; Codeforces has "2-sat" and "fft" as tags while LeetCode does not distinguish these at all. Our unified taxonomy maps 50+ platform-specific tags to ~35 canonical topics through manual analysis of semantic overlap, with unmapped tags preserved using an "other:" prefix to avoid information loss.

**Data accessibility**: Platform APIs vary significantly in openness. Codeforces provides a fully public REST API returning complete submission histories with problem metadata, verdicts, and timestamps. AtCoder has no official API, but the community-maintained kenkoooo API provides equivalent data. LeetCode's GraphQL API provides limited public data — user profile statistics and recent accepted submissions — but full submission history requires authentication [19]. Through investigation, we discovered that LeetCode's `recentAcSubmissionList` query returns up to 200 recent accepted submissions publicly, providing sufficient signal for profile building despite not offering complete history. The IBM Project CodeNet dataset [20] provides 13.9 million submissions across 4,053 problems (primarily AtCoder), offering an alternative offline data source for model training.

**NLP for cross-platform tag transfer**: When platform metadata is missing, NLP techniques can extract topic information from problem statement text. Multi-label text classification using TF-IDF features and logistic regression has shown strong performance for categorising programming problems by topic [21], particularly for "content-oriented" tags (strings, math, game theory) where distinctive vocabulary signals the topic. "Method-oriented" tags (dynamic programming, binary search) are harder to predict from text alone, as the solution approach is not always evident from the problem statement — a problem requiring dynamic programming might describe counting paths in a grid, optimising a knapsack, or aligning two strings, none of which share distinctive vocabulary with each other.

### 2.5 Summary and Gap Analysis

The literature establishes strong foundations for recommendation systems, educational data mining, and NLP-based classification. However, three significant gaps remain:

1. **No cross-platform competitive programming recommender exists** — all current tools operate within single platforms.
2. **No unified cross-platform dataset** has been curated combining Codeforces, AtCoder, and LeetCode with normalised difficulty and unified tags.
3. **Tag transfer across platforms** using NLP has not been systematically evaluated in the competitive programming domain.

CPRS addresses all three gaps, contributing both a curated dataset and a recommendation engine that leverages cross-platform data.

---

## Chapter 3: Design

### 3.1 System Architecture

CPRS consists of four layers: data collection, data processing, recommendation engine, and delivery. Figure 1 illustrates the architecture.

```
┌─────────────────────────────────────────────────────┐
│                   Web Frontend                       │
│  (Auth, Handle Management, Profile View, Recs)       │
├─────────────────────────────────────────────────────┤
│                   FastAPI Backend                     │
│  /api/recommend  /api/me  /api/handles  /api/stats   │
├──────────┬──────────┬──────────┬────────────────────┤
│ CF       │ AtCoder  │ LeetCode │  Recommendation    │
│ Fetcher  │ Fetcher  │ Fetcher  │  Engine            │
├──────────┴──────────┴──────────┼────────────────────┤
│        Unified Dataset         │  User Database     │
│  24,335 problems normalised    │  (SQLite: users,   │
│  difficulty + unified tags     │   handles, sessions)│
└────────────────────────────────┴────────────────────┘
```
*Figure 1: CPRS system architecture.*

**Data Collection Layer**: Three platform-specific fetchers interact with public APIs. The Codeforces fetcher uses the REST API (`codeforces.com/api/`) for problems and user submissions. The AtCoder fetcher uses the kenkoooo API (`kenkoooo.com/atcoder/`) for problems, difficulty models, and user submissions. The LeetCode fetcher uses GraphQL queries for the problem catalogue and public user data (`recentAcSubmissionList`, `userPublicProfile`).

**Data Processing Layer**: Raw platform data is converted to a unified schema (`UnifiedProblem`) with normalised difficulty scores on a [0, 1] scale and a shared tag taxonomy of ~35 canonical topics. An NLP auto-tagger predicts tags for AtCoder problems that lack them (see Section 4.2).

**Recommendation Engine**: A content-based scoring algorithm evaluates each candidate problem against the user's skill profile (see Section 3.2).

**Delivery Layer**: A FastAPI web service exposes recommendation, authentication, and profile management endpoints. A single-page HTML/JavaScript frontend provides the user interface.

### 3.2 Recommendation Algorithm

The recommender uses a weighted multi-factor scoring formula:

```
score = 0.4 × topic_gap + 0.4 × difficulty_fit + 0.1 × popularity + 0.1 × diversity
```

**Topic gap** (weight 0.4): Measures how well a problem targets the user's weak areas. Each problem's tags are weighted by the inverse of the user's solve rate for that topic. Topics the user has never attempted receive the highest weight (1.0), followed by topics with low solve rates. This drives recommendations toward skill gaps rather than strengths.

**Difficulty fit** (weight 0.4): A Gaussian function centred slightly above the user's current difficulty level (75th percentile of solved problems + 10% stretch). The Gaussian spread (σ = 0.15) allows some tolerance — problems within ±0.15 normalised difficulty of the target score well, while much easier or harder problems are penalised. This embodies the pedagogical principle of the "zone of proximal development" [22] — optimal learning occurs just beyond current ability.

**Popularity** (weight 0.1): Log-scaled solve count provides a slight preference for well-tested problems with known quality, avoiding obscure or potentially flawed problems.

**Diversity** (weight 0.1): A small random component prevents monotonous recommendations and ensures variety across recommendation sessions.

**User profile construction** proceeds in three steps:

1. **Data fetching**: Submission histories are retrieved from all linked platforms via their respective APIs. Rate limiting is enforced per-platform (0.5s for Codeforces, 1.0s for AtCoder, 1.0s for LeetCode) to respect API usage policies.
2. **Per-platform profile building**: For each platform, the system tracks solved and attempted problems, computes per-topic mastery metrics (solve rate, average difficulty, maximum difficulty solved), and sets the user's overall difficulty level at the 75th percentile of solved problem difficulties. The 75th percentile is chosen over the mean or maximum because it represents the user's comfortable upper range — the difficulty level at which they reliably succeed — while being robust to outlier solves (e.g., a lucky contest submission on a much harder problem).
3. **Cross-platform merging**: Profiles are merged by unioning solved problem ID sets (preventing cross-platform duplicates from inflating counts), summing topic mastery statistics across platforms, computing a weighted-average difficulty level (weighted by solve count per platform, so the platform where the user is most active contributes most), and taking the maximum rating as the best indicator of peak ability. This merged profile enables the recommender to identify gaps that would be invisible within a single platform — for example, a user who practises strings extensively on Codeforces but has never attempted string problems on AtCoder would not receive redundant string recommendations.

**Weak topic identification** is central to the recommendation strategy. Topics are ranked by "weakness score": topics never attempted receive the highest priority (1.5), followed by topics with fewer than 5 attempts (boosted by 0.3), and remaining topics are scored by their inverse solve rate. The top 5 weakest topics are targeted in each recommendation session. This approach ensures that recommendations systematically address skill gaps rather than reinforcing existing strengths, which is the key pedagogical difference between CPRS and entertainment-oriented recommenders.

### 3.3 Data Normalisation Design

**Difficulty normalisation** maps each platform's scale to [0, 1]:

- **Codeforces** (800–3500): `(rating - 800) / 2700` — linear mapping of the fine-grained integer scale.
- **AtCoder** (-500 to 4000): `(difficulty + 500) / 4500` with outlier clipping — uses kenkoooo's model-estimated difficulties [17].
- **LeetCode** (3 levels): Easy → 0.2, Medium → 0.5, Hard → 0.85 — heuristic mapping reflecting that LeetCode "Easy" problems often require non-trivial algorithmic thinking.

This min-max approach is a practical approximation. A more principled method would use Item Response Theory [18] to jointly calibrate difficulties from cross-platform solve rates, but this requires user-level data that spans platforms — precisely the data CPRS aims to collect for future iterations.

**Tag taxonomy unification** maps 50+ platform-specific tags to ~35 canonical topics. The mapping was designed by analysing tag co-occurrence patterns and semantic overlap. For example, Codeforces's "dp" and LeetCode's "dynamic-programming" both map to `dynamic_programming`. Tags with no clear mapping are preserved with an `other:` prefix to avoid information loss.

### 3.4 NLP Auto-Tagging Pipeline

To address AtCoder's missing topic tags, we implemented a multi-label text classifier trained on Codeforces problem statements (which have human-curated tags):

1. **Data**: 1,999 scraped CF problem statements with tags (training), 795 scraped AtCoder statements (inference).
2. **Features**: TF-IDF vectorisation with 10,000 features, bigrams, and sublinear term frequency scaling. Sublinear TF (`1 + log(tf)`) prevents common terms from dominating.
3. **Model**: One-vs-Rest Logistic Regression with balanced class weights, targeting 16 tags with ≥50 training samples. Balanced weights address class imbalance (e.g., "math" appears in 42% of problems but "combinatorics" in only 3%).
4. **Evaluation**: 5-fold cross-validation on CF data, plus systematic confidence threshold calibration at 7 threshold levels.
5. **Output**: Predicted tags with confidence scores for each AtCoder problem.

### 3.4 Evaluation Strategy

The evaluation strategy addresses both the NLP auto-tagger and the recommendation engine, using metrics appropriate to each component:

**Auto-tagger evaluation** (multi-label classification):
- Per-tag and micro/macro-averaged precision, recall, and F1 via 5-fold cross-validation
- Confidence threshold calibration: precision-recall trade-off at thresholds 0.3–0.9
- Per-tag optimal threshold analysis for maximising individual tag F1
- Confusion analysis to identify systematic misclassification patterns

**Recommendation engine evaluation**:
- **Offline evaluation**: Hold-out testing — for users with sufficient history, hide the most recent N solved problems and measure whether the recommender surfaces them. Metrics: Hit Rate@K, Mean Reciprocal Rank (MRR), and Normalised Discounted Cumulative Gain (nDCG).
- **Difficulty calibration**: Measure correlation between recommended difficulty and user success rate — if the recommender targets the right difficulty, users should solve ~60–70% of recommendations (challenging but achievable).
- **Topic coverage**: Measure whether recommendations span the user's weak topics rather than concentrating on a single area.
- **Cross-platform value**: Compare recommendation quality for single-platform profiles vs. multi-platform merged profiles. If cross-platform data adds value, merged profiles should produce more diverse and better-targeted recommendations.

This strategy goes beyond the template's suggested metrics (precision/recall, RMSE, AUC) by incorporating domain-specific measures (difficulty calibration, topic coverage) that capture recommendation quality beyond simple accuracy. The evaluation approach is informed by Herlocker et al.'s framework for evaluating collaborative filtering recommender systems [23], adapted for the educational context where the goal is learning rather than preference satisfaction.

**Justification**: Standard recommendation metrics like precision@K measure whether a user would *engage with* a recommended item. In educational recommendation, this is necessary but insufficient — we also need to measure whether the recommendation *teaches* something. Difficulty calibration addresses this: if 90% of recommendations are too easy, the system is reinforcing existing skills rather than promoting growth. Topic coverage addresses breadth: a system that only recommends graph problems, however accurate, fails a user who also needs DP practice. These domain-specific metrics are essential for evaluating whether CPRS fulfils its educational purpose, not merely its recommendation accuracy.

### 3.5 Workplan

The project is divided into six phases across the remaining course weeks:

| Phase | Weeks | Tasks | Deliverables |
|-------|-------|-------|-------------|
| 1. Data & Prototype | 1–10 | Build fetchers, unified dataset, NLP tagger, content-based recommender, web UI | Preliminary report, feature prototype |
| 2. Advanced Models | 11–14 | Implement collaborative filtering (matrix factorisation), hybrid scoring, prerequisite graph | Trained models, comparison metrics |
| 3. Evaluation | 15–17 | Hold-out evaluation, difficulty calibration, cross-platform value analysis, statistical significance testing | Evaluation results, visualisations |
| 4. Iteration | 18–19 | Address evaluation findings, tune hyperparameters, improve NLP tagger (explore BERT embeddings, IBM CodeNet data) | Improved models |
| 5. Reporting | 20–22 | Write final report, prepare for written exam | Draft report, exam preparation |
| 6. Finalisation | 23–24 | Polish report, record demo video, code cleanup | Final report, demo video, code submission |

**Phase dependencies**: Phases 2 and 3 are partially parallelisable — collaborative filtering implementation and evaluation harness development can proceed simultaneously. Phase 4 depends on Phase 3 results to identify which components need improvement. Phase 5 can begin during Phase 4 for sections not affected by iteration results (introduction, literature review).

**Risk assessment and contingencies**:

- *Risk*: AtCoder scraping remains blocked (403 errors). *Contingency*: Use IBM CodeNet dataset for AtCoder problem statements, expanding NLP training data significantly (4,053 problems with statements).
- *Risk*: Collaborative filtering underperforms due to sparse user-problem matrix. *Contingency*: Focus on the content-based + knowledge-gap hybrid approach, which does not require a dense interaction matrix.
- *Risk*: LeetCode API changes or rate limits tighten. *Contingency*: Support manual CSV import of solved problem lists as a fallback.
- *Risk*: Evaluation metrics are difficult to compute without a user study. *Contingency*: Use synthetic evaluation with historical data (time-based hold-out) and qualitative analysis of recommendations for known expert users.

---

## Chapter 4: Feature Prototype

### 4.1 Prototype Overview

The feature prototype implements the complete data science pipeline described in Chapter 3: data ingestion from three platforms, preprocessing and normalisation, NLP-based tag prediction, user profile construction, and personalised recommendation generation. It is deployed as a functional web application with a FastAPI backend and a single-page frontend.

The prototype demonstrates three technically challenging components:

1. **Cross-platform dataset construction** — 24,335 problems from Codeforces (11,263), AtCoder (9,095), and LeetCode (3,977) with normalised difficulty scores and unified topic tags.
2. **NLP auto-tagger** — a multi-label classifier that predicts topic tags for AtCoder problems using a model trained on Codeforces problem statements.
3. **Multi-platform recommendation engine** — a content-based recommender that builds merged user profiles from multiple platform submission histories and generates cross-platform recommendations targeting skill gaps.

### 4.2 NLP Auto-Tagger: Evaluation

The auto-tagger was evaluated using 5-fold cross-validation on 1,999 Codeforces problems with ground-truth tags. The model achieved a micro-averaged F1 of 0.530 and macro-averaged F1 of 0.435, with significant variation across tag categories. The 16 target tags were those with at least 50 training samples in the Codeforces dataset; rarer tags were excluded to avoid unreliable evaluation.

**Key findings from evaluation:**

The results reveal two distinct tiers of tags. *Content-oriented tags* achieve strong classification performance: strings (F1=0.736, support=206), game theory (F1=0.733, support=54), math (F1=0.694, support=844), and greedy (F1=0.663, support=868). These tags succeed because their problem statements contain distinctive vocabulary — words like "string", "subsequence", "game", "player", "prime", and "divisor" reliably signal these topics. The classifier learns strong lexical associations that generalise well across platforms.

*Method-oriented tags* perform substantially worse: dynamic programming (F1=0.200, support=187), binary search (F1=0.245, support=127), and two pointers (F1=0.184, support=100). These tags describe solution *techniques*, not problem *content*. A dynamic programming problem might describe a grid traversal, a knapsack scenario, or a string alignment — the text does not reliably indicate the required approach. This distinction between content and method tags is a key insight from the evaluation: it suggests that purely text-based classification has a fundamental ceiling for method-oriented tags, and that alternative features (e.g., difficulty level, solution structure, editorial text) would be needed to improve these categories.

**Confidence threshold calibration** was performed systematically at seven threshold levels (0.3–0.9). A global threshold of 0.5 maximises F1, providing 51.4% precision and 53.6% recall with an average of 2.5 tags per problem. At this threshold, 99% of problems receive at least one predicted tag, ensuring broad coverage of the AtCoder problem set. Lower thresholds (0.3) achieve 89.9% recall but only 29.7% precision — too noisy for practical use. Higher thresholds (0.7) achieve 75.9% precision but only 17.2% recall — too few tags to be useful for topic-based recommendation.

Per-tag optimal thresholds vary significantly: strings performs best at threshold 0.55 (F1=0.743), while dynamic programming performs best at threshold 0.35 (F1=0.232). This suggests that a per-tag threshold strategy could improve overall performance by allowing aggressive recall for well-calibrated tags while maintaining conservative precision for noisy tags.

**Confusion analysis** revealed that the most common misclassifications occur between semantically overlapping tags: math, greedy, implementation, and constructive frequently co-occur in training data and are confused by the classifier. This reflects genuine ambiguity in tagging conventions rather than pure model error — on Codeforces, human problem setters frequently disagree on whether a problem is "greedy" or "constructive", and many problems legitimately belong to both categories.

### 4.3 Recommendation Engine: Evaluation

The recommender was tested qualitatively with real Codeforces users at different skill levels to assess whether recommendations are sensible and well-targeted.

For an expert user (tourist, rating 3439, 3019 solved), the recommender correctly identifies that this user has mastered most topics within the Codeforces taxonomy. The resulting recommendations are predominantly cross-platform — LeetCode and AtCoder problems in topics categorised differently across platforms (e.g., "array", "hash_table", "backtracking" — tags from LeetCode's taxonomy that have no direct Codeforces equivalent). This demonstrates the core cross-platform value proposition: even for a user who has exhaustively practised on one platform, the unified dataset surfaces relevant problems from other platforms that the user would not discover through single-platform tools.

For intermediate users (rating 1200–1800), recommendations correctly target identified weak topics at appropriate difficulty levels, with a balanced mix of problems from all three platforms. The difficulty fit component successfully centres recommendations slightly above the user's current level, avoiding both trivially easy problems and impossibly hard ones.

The recent performance analysis component provides additional context by analysing the user's last N submissions per platform. It computes solve rate, average difficulty, most-practised topics, and a trend indicator. The trend is determined by comparing the average difficulty of the more recent half of submissions against the older half: a difference exceeding 5% in normalised difficulty flags "improving" or "declining", otherwise "stable". This gives users insight into whether their practice is effectively raising their skill level or stagnating.

### 4.4 Prototype Limitations and Improvements

**Current limitations:**

1. **NLP tagger accuracy on method tags**: F1 scores below 0.25 for dynamic programming, binary search, and two pointers limit the system's ability to recommend these important topic areas for AtCoder problems. The TF-IDF + Logistic Regression approach captures lexical but not semantic features of problem statements.

2. **AtCoder scraping coverage**: Only 795 of 9,095 AtCoder problems have scraped statements (8.7%), due to HTTP 403 blocks. The remaining 8,300 problems have predicted tags only if they match problems in the scraped set. This creates a significant coverage gap.

3. **No collaborative filtering**: The current engine is purely content-based. It does not leverage the signal from similar users' solving patterns, missing the "users like you also solved" dimension.

4. **LeetCode data limitations**: The public API provides only recent accepted submissions (up to ~200), not the full history. Solve rates cannot be computed since only accepted submissions are returned.

5. **Evaluation is qualitative**: Without a formal hold-out study or user trial, the recommendation quality assessment relies on expert inspection rather than quantitative metrics.

**Planned improvements for the final project:**

1. **Upgrade the NLP tagger**: Replace TF-IDF with pre-trained embeddings (CodeBERT or Sentence-BERT) to capture semantic rather than lexical similarity, particularly improving method-oriented tag prediction.

2. **Expand AtCoder coverage**: Use the IBM CodeNet dataset [20] (4,053 AtCoder problem statements) as an alternative to web scraping, increasing coverage from 795 to ~4,000 problems.

3. **Add collaborative filtering**: Implement matrix factorisation (SVD) using the CF Open Dataset (17.6M submissions, 15K users), combined with content-based scores in a hybrid model weighted by user interaction density.

4. **Formal evaluation**: Time-based hold-out — hide users' most recent 50 solved problems, measure Hit Rate@20 and MRR. Compare single- vs. cross-platform profiles using paired t-tests.

5. **Per-tag confidence thresholds**: Replace the global 0.5 threshold with per-tag optimal thresholds from calibration analysis, expected to improve macro-averaged F1 by ~8%.

### 4.5 Prototype Summary

The prototype successfully demonstrates the feasibility of the core CPRS concept: cross-platform data can be unified into a coherent dataset, NLP can partially bridge metadata gaps between platforms, and content-based recommendations can leverage multi-platform user profiles to target skill gaps that would be invisible within any single platform. The NLP auto-tagger, while imperfect, validates that cross-platform tag transfer is viable for content-oriented topics and provides a clear roadmap for improvement through embedding-based approaches. The recommendation engine produces sensible results for real users across different skill levels, and the web interface makes the full pipeline accessible for practical use. The prototype also validates the technical feasibility of real-time multi-platform data fetching — the system queries three external APIs and returns merged recommendations within seconds, demonstrating that the cross-platform approach is practical, not merely theoretical.

---

## References

[1] S. Wasik, M. Antczak, J. Badura, A. Laskowski, and T. Sternal, "A Survey on Online Judge Systems and Their Applications," *ACM Computing Surveys*, vol. 51, no. 1, pp. 1–34, 2018.

[2] J. Liang, "Are You Hiring Based on LeetCode Performance? A Discussion on Technical Interview Practices," *IEEE Software*, vol. 38, no. 3, pp. 89–93, 2021.

[3] S. Ahmed, "A4: A Codeforces Practice Assistant Using Machine Learning," *GitHub*, 2023. [Online]. Available: https://github.com/recommenders/a4-cf.

[4] F. Ricci, L. Rokach, B. Shapira, and P. B. Kantor, Eds., *Recommender Systems Handbook*, 2nd ed. New York, NY: Springer, 2015.

[5] Y. Koren, R. Bell, and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," *Computer*, vol. 42, no. 8, pp. 30–37, 2009.

[6] P. Lops, M. De Gemmis, and G. Semeraro, "Content-Based Recommender Systems: State of the Art and Trends," in *Recommender Systems Handbook*, Springer, 2011, pp. 73–105.

[7] R. Burke, "Hybrid Recommender Systems: Survey and Experiments," *User Modeling and User-Adapted Interaction*, vol. 12, no. 4, pp. 331–370, 2002.

[8] S. Rendle, "Factorization Machines," in *Proc. 2010 IEEE International Conference on Data Mining (ICDM)*, 2010, pp. 995–1000.

[9] A. T. Corbett and J. R. Anderson, "Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge," *User Modeling and User-Adapted Interaction*, vol. 4, no. 4, pp. 253–278, 1994.

[10] D. Boud and E. Molloy, "Rethinking Models of Feedback for Learning: The Challenge of Design," *Assessment & Evaluation in Higher Education*, vol. 38, no. 6, pp. 698–712, 2013.

[11] R. Pelánek, "Bayesian Knowledge Tracing, Logistic Models, and Beyond: An Overview of Learner Modeling Techniques," *User Modeling and User-Adapted Interaction*, vol. 27, no. 3–5, pp. 313–350, 2017.

[12] A. E. Elo, *The Rating of Chessplayers, Past and Present*. New York, NY: Arco Publishing, 1978.

[13] C. Piech, J. Bassen, J. Huang, S. Ganguli, M. Sahami, L. J. Guibas, and J. Sohl-Dickstein, "Deep Knowledge Tracing," in *Advances in Neural Information Processing Systems*, vol. 28, 2015.

[14] X. Xiong, S. Zhao, E. G. Van Inwegen, and J. E. Beck, "Going Deeper with Deep Knowledge Tracing," in *Proc. 9th International Conference on Educational Data Mining (EDM)*, 2016, pp. 545–550.

[15] N. Thai-Nghe, L. Drumond, A. Krohn-Grimberghe, and L. Schmidt-Thieme, "Recommender System for Predicting Student Performance," *Procedia Computer Science*, vol. 1, no. 2, pp. 2811–2819, 2010.

[16] S. Zhang, L. Yao, A. Sun, and Y. Tay, "Deep Learning Based Recommender System: A Survey and New Perspectives," *ACM Computing Surveys*, vol. 52, no. 1, pp. 1–38, 2019.

[17] kenkoooo, "AtCoder Problems: Difficulty Estimation," 2024. [Online]. Available: https://kenkoooo.com/atcoder/.

[18] F. B. Baker, *The Basics of Item Response Theory*, 2nd ed. College Park, MD: ERIC Clearinghouse on Assessment and Evaluation, 2001.

[19] LeetCode, "LeetCode GraphQL API," 2024. [Online]. Available: https://leetcode.com/graphql.

[20] R. Puri, D. S. Kung, G. Janber, W. Zhang, et al., "CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks," in *Proc. 35th Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, 2021.

[21] A. Agrawal, D. Gupta, and R. Singh, "Automated Categorisation of Programming Problems Using Natural Language Processing," in *Proc. International Conference on Computational Intelligence and Data Science*, 2020, pp. 312–319.

[22] L. S. Vygotsky, *Mind in Society: The Development of Higher Psychological Processes*. Cambridge, MA: Harvard University Press, 1978.

[23] J. L. Herlocker, J. A. Konstan, L. G. Terveen, and J. T. Riedl, "Evaluating Collaborative Filtering Recommender Systems," *ACM Transactions on Information Systems*, vol. 22, no. 1, pp. 5–53, 2004.

[24] M. F. Dacrema, P. Cremonesi, and D. Jannach, "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches," in *Proc. 13th ACM Conference on Recommender Systems (RecSys)*, 2019, pp. 101–109.

[25] R. C. Wilson, A. Shenhav, M. Straccia, and J. D. Cohen, "The Eighty Five Percent Rule for Optimal Learning," *Nature Communications*, vol. 10, no. 1, p. 4646, 2019.

[26] T. Huang, G. Zhao, S. Yang, and Z. Niu, "Knowledge-Driven Recommendation for Programming Exercises," *IEEE Access*, vol. 8, pp. 112872–112884, 2020.
