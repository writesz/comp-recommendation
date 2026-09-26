# CM3070 Final Project — CPRS

Cross-platform competitive programming recommendation system (Codeforces / AtCoder /
LeetCode). **Template 1.1: Data-Driven Personalised Educational Content Recommendation.**

Report source: `cprs/final_report.tex` · Build plan: `cprs/FINAL_PROJECT_PLAN.md` ·
Decisions: `cprs/DECISIONS_LOG.md`

---

## Marking structure

| Submission | Weight | Rubric total |
|---|---:|---:|
| Final Project Report **& Code** | 60% | 120 points |
| Presentation Video | 5% | 10 points |

Source of truth: `final_report_rubric_v2.pdf` (6 pages; report rubric pp. 1–5, video
rubric p. 6). Template bands: `cprs/REQUIREMENTS.md` §C, verbatim from
`FinalProjectTemplates (1).pdf`.

## Report rubric — 14 criteria, 120 points

| # | Criterion | Max | Top band requires |
|---|---|---:|---|
| 1 | Clearly written and presented | 10 | "very professional level of language to academic or industry standards" |
| 2 | Diagrams and images appropriate and clear | 10 | "very professional level of visual materials" |
| 3 | Knowledge of area, previous work, literature | 10 | "detailed and insightful… goes **considerably beyond** the template" |
| 4 | Critically evaluates previous work | 6 | same "considerably beyond" wording |
| 5 | Appropriate citations | 4 | "**all** work mentioned… cited in correct ACM style" |
| 6 | Design clear and of high quality | 12 | "very clear and coherent… described well **in both visuals and writing**" |
| 7 | Concept justified by domain and users | 8 | "detailed, insightful and **innovative** analysis of the users and/or domain" |
| 8 | **Final implementation of high quality** | **22** | working prototype at the template's **"excellent"** level |
| 9 | Implementation **that is described** technically challenging | 8 | "close to cutting edge research (advanced masters or PhD)" |
| 10 | Evaluation strategy appropriate to aims | 6 | "very detailed and well thought out, going considerably beyond the template" |
| 11 | Evaluation coverage | 5 | "very detailed and covers a number of **subtle** issues" |
| 12 | Results presented well | 5 | "very professional standard, either of academic or industry research" |
| 13 | Results → critical analysis vs objectives | 4 | "**potentially generalisable conclusions**" |
| 14 | Originality | 10 | "PhD level, or a highly innovative commercial product" |

### Video rubric — 10 points

| Criterion | Max |
|---|---:|
| Is the final product of high quality? | 7 |
| Is the implementation that is described technically challenging? | 3 |

## The code rubric

**There is no separate rubric for the code.** Code is marked through report criteria
**8 (22 pts)** and **9 (8 pts)**, plus the two video criteria — 30 of 120 report points.

Every band in criterion 8 is defined *by reference to the template*: "a working prototype
that would correspond to part of a project at the 'excellent' level of project described
in the template." So the de-facto code rubric is Template 1.1's own grading bands:

- **3rd (pass):** basic recommendation algorithm (e.g. CF with simple matrix
  factorisation); public dataset; basic evaluation on a single metric; clear
  documentation of the data science pipeline.
- **2:2–2:1 (good):** + a more advanced model (e.g. hybrid); data preprocessing and
  feature engineering; multiple metrics with cross-validation; clear explanation of the
  strengths and weaknesses of the approach.
- **1st (outstanding):** + a novel approach or adaptation of a state-of-the-art
  technique; creation/curation of a high-quality dataset; comprehensive evaluation with
  baseline comparisons **and statistical significance testing**; detailed analysis of
  model performance and insights; potentially publishable results.

There are **no criteria for code style, test coverage or architecture as such**. Those
pay off only insofar as they push the implementation into a higher template band.

## Structural notes that change how to prioritise

**Bands are non-linear; the jumps are where the marks are.** Criterion 8 runs
0 → 6 → 12 → 15 → 18 → 22 with nothing in between, and 12 points is merely "a working
prototype at *acceptable* level" — the real contest is 18 ("good") → 22 ("excellent").
Criterion 6 runs 0 → 2 → 4 → 5 → 6 → 8 → 10 → 12.

**Citations are effectively binary.** 2 points for "some previous work without proper
citations"; 4 for *all* work cited correctly in ACM style. One bad entry costs the same
as ten, so every reference must be real and verifiable. Three fabricated references were
found and replaced during drafting — verify any new one against dblp or the ACM DL
before adding it.

**Criterion 9 says "that is described".** Technical difficulty is scored on what the
report and video describe, not on what sits in the repo. Undocumented cleverness scores
zero. The same pressure applies to criterion 8, since markers reach the code through the
report.

**Design at 8+ explicitly requires "suitable visual communication".** Weak diagrams cap
criterion 6 (12 pts) as well as criterion 2 (10 pts).

**The evaluation cluster is worth 20** (criteria 10–13), nearly as much as
implementation. Criterion 13's top band wants *generalisable conclusions*, not a list of
pros and cons.

## Working conventions

- Exploration and data work goes in Python scripts under `cprs/scripts/`, not inline
  shell.
- Numbers quoted in the report are regenerated by `scripts/report_stats.py` rather than
  transcribed by hand.
- ToC word counts go stale on every edit; regenerate with
  `scripts/update_word_counts.py`.
- Tests: `python -m pytest cprs/tests -q`.
- Build the report: `cd cprs && latexmk -pdf final_report.tex`.
- Do **not** add `Co-Authored-By` or any AI attribution to commit messages.
