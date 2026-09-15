# CPRS — Competitive Programming Recommendation System

## Project Requirements

> Adapted from CM3005 Data Science — Project Idea 1:
> "Data-Driven Personalised Educational Content Recommendation"
> (CM3070 Computer Science Final Project, University of London)

---

## 1. Problem Statement

Build a data-driven recommendation system that leverages competitive programming
platform data (Codeforces, AtCoder, LeetCode) to provide personalised problem
recommendations tailored to individual user learning patterns and skill levels.

---

## 2. Functional Requirements

### 2.1 Data Pipeline

- [ ] Collect data from competitive programming platforms (Codeforces, AtCoder, LeetCode)
  - User submission histories
  - Problem metadata (tags, difficulty, categories)
  - Contest/performance data
- [ ] Clean and preprocess raw data
  - Handle missing values, duplicates, inconsistencies
  - Normalise difficulty ratings across platforms
- [ ] Engineer features from raw data
  - User skill profiles (per-topic strengths, solve rates, progression over time)
  - Problem features (difficulty, topic tags, acceptance rates)

### 2.2 Recommendation Engine

- [ ] Implement at least one core recommendation algorithm:
  - Collaborative Filtering (user-based or item-based)
  - Content-Based Filtering (problem metadata similarity)
  - Matrix Factorisation (e.g. SVD, ALS)
- [ ] *(For higher grade)* Implement a hybrid approach combining multiple techniques
- [ ] *(For higher grade)* Explore deep learning methods (e.g. RNNs for sequence modelling)
- [ ] Generate ranked list of recommended problems per user
- [ ] Build and maintain user profiles with learned preferences

### 2.3 NLP / Text Analysis *(optional, for higher grade)*

- [ ] Topic modelling on problem statements
- [ ] Sentiment analysis on editorial/discussion data
- [ ] Text embeddings (Word2Vec, BERT) for problem similarity

### 2.4 Delivery

- [ ] Expose recommendations via a Python API or web service
- [ ] Accept user history/preferences as input
- [ ] Return personalised problem recommendations with scores

---

## 3. Evaluation Requirements

- [ ] Precision and recall of recommendations
- [ ] RMSE or other relevant metrics for rating/difficulty prediction
- [ ] Area under the ROC curve (AUC) for binary classification tasks
- [ ] Cross-validation to ensure model generalisation
- [ ] *(For higher grade)* Statistical significance testing (e.g. A/B testing)
- [ ] *(For higher grade)* Comparison against baseline models

---

## 4. Output Artefacts

- [ ] Trained and serialised ML model files
- [ ] Recommendation scores for each content item
- [ ] User profiles with learned preferences
- [ ] Visualisations of model performance:
  - Precision-recall curves
  - ROC curves
  - Learning curves
- [ ] Processed dataframes and analysis notebooks/scripts

---

## 5. Relevant Techniques

| Area | Techniques |
|------|-----------|
| Machine Learning | Collaborative Filtering, Content-Based Filtering, Matrix Factorisation, Deep Learning (RNNs) |
| NLP | Topic Modelling, Sentiment Analysis, Text Embeddings (Word2Vec, BERT) |
| Data Mining | Clustering, Classification, Association Rule Mining |
| Statistics | Hypothesis Testing, A/B Testing, Cross-Validation |
| Visualisation | Matplotlib, Seaborn, Plotly |

---

## 6. Grading Criteria

### 3rd Class (Pass)

- [ ] Basic recommendation algorithm (e.g. collaborative filtering with matrix factorisation)
- [ ] Use of a publicly available dataset
- [ ] Evaluation using a single metric
- [ ] Clear documentation of the data science pipeline

### 2:2 – 2:1 (Good)

All of the above, plus:

- [ ] More advanced or hybrid recommendation model
- [ ] Data preprocessing and feature engineering
- [ ] Evaluation using multiple metrics and cross-validation
- [ ] Clear explanation of strengths and weaknesses of the chosen approach

### 1st Class (Outstanding)

All of the above, plus:

- [ ] Novel data science approach or adaptation of a state-of-the-art technique
- [ ] Creation or curation of a high-quality dataset
- [ ] Comprehensive evaluation with baseline comparisons and statistical significance testing
- [ ] Detailed analysis of model performance and insights
- [ ] Potentially publishable results

---

## 7. Course Deliverables & Deadlines

| Deliverable | Summative? | Deadline | Weight |
|-------------|-----------|----------|--------|
| Project Proposal | Formative | ~Week 4 | 0% |
| Preliminary Project Report | Yes | ~Week 10 | 10% |
| Check-in Quizzes | Yes | Weeks 1–20 | 5% |
| Draft Report | Formative | ~Week 18 | 0% |
| Written Examination | Yes | ~Week 22 | 20% |
| Presentation Video | Yes | ~Week 24 | 5% |
| Final Project Report & Code | Yes | ~Week 24 | 60% |

---

## 8. Recommended Resources

- Google ML Recommendation Systems documentation
- Scikit-learn documentation
- TensorFlow documentation
- NLTK documentation
- Research papers on "Educational Data Mining" and "Learning Analytics"
  (Google Scholar, ACM Digital Library)
