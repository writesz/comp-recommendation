# CPRS — Project Requirements (Extracted from Official Documents)

> All quotes below are verbatim from the course documents.
> Sources: `CM3070 Computer Science Final Project 2025 (1).pdf` and `FinalProjectTemplates (1).pdf`

---

## A. Course-Level Requirements (CM3070)

### Course Goals — What You Must Demonstrate

1. Select and apply appropriate Computer Science techniques to a particular problem
2. Develop a project proposal that can be addressed using Computer Science techniques
3. Evaluate previous work in areas related to your chosen project and write a literature review
4. Design and develop a substantial piece of software that matches a project brief
5. Test and evaluate a software project in terms of how it matches a project brief including such factors as user needs, software correctness and efficiency
6. Report the results of the project in written, oral and visual form

### Assessment Breakdown

| Activity | Required? | Deadline | Est. Time | Weight |
|----------|-----------|----------|-----------|--------|
| Project Proposal | Yes (formative) | ~Week 4 | 10 hours | 0% |
| Preliminary Project Report | Yes | ~Week 10 | 50 hours | 10% |
| Check-in Quizzes | Yes | Weeks 1–20 | 2 hours | 5% |
| Draft Report | Yes (formative) | ~Week 18 | 3 hours | 0% |
| Written Examination | Yes | ~Week 22 | 3–4 hours | 20% |
| Project Presentation Video | Yes | ~Week 24 | 4 hours | 5% |
| Final Project Report & Code | Yes | ~Week 24 | 150 hours | 60% |

> "Two submissions are formative ones that do not count towards the final grade but are required to receive timely feedback from your project supervisor."

> "Two submissions are summative ones that do count towards the final grade and take place in the mid-term and at the end of the course."

### 10-Topic Course Structure (what examiners expect to see covered)

| Topic | What It Covers |
|-------|---------------|
| 1. Project Concept | Choose project idea and template |
| 2. Project Proposal | Scope, ethics quiz |
| 3. Background Research | Literature review, evaluate previous work |
| 4. Design | Design the project, prototype key functionality |
| 5. Planning and Evaluation | Project planning, prototyping, evaluation criteria |
| 6. Development | Dev environment setup, first iteration |
| 7. Testing and Iteration | Test criteria, peer feedback |
| 8. Academic Writing | Report outline, draft report |
| 9. Your Project and Career | Portfolio piece, draft report |
| 10. Completing Your Project | Demo video, final written report, evaluate project as a whole |

---

## B. Template-Specific Requirements (1.1 Data-Driven Personalised Educational Content Recommendation)

### Problem Statement (verbatim)

> "Can we leverage advanced data science techniques to build a highly accurate and personalised educational content recommendation system?"

### Background (verbatim)

> "The explosion of online learning resources presents a challenge: how do learners find the most relevant materials? Traditional recommendation systems often rely on simplistic metrics. This project aims to apply sophisticated data science methods to understand individual learning patterns and preferences. By analysing large datasets of user interactions, content metadata, and learning outcomes, we can build models that predict optimal learning pathways. This involves exploring techniques from machine learning, natural language processing, and network analysis to create a system that truly personalises the educational experience, improving knowledge retention and learning efficiency."

### Final Product (verbatim)

> "The final product should be a data-driven recommendation engine, likely implemented as a Python API or a web service. It should demonstrate the ability to process user data, apply machine learning models, and generate personalised recommendations. The project should include a thorough evaluation of the model's performance using relevant data science metrics."

### Prototype (verbatim)

> "A prototype would demonstrate the core data science pipeline: data ingestion, preprocessing, model training, and recommendation generation. It needs to prove that the chosen algorithms can learn from data and produce meaningful recommendations. It's important to clearly show the data transformations and model outputs. A complete user interface is not essential at this stage."

### Required Techniques/CS Fundamentals

- **Machine Learning**: Collaborative Filtering, Content-Based Filtering, Matrix Factorisation, Deep Learning (e.g. RNNs for sequence modelling)
- **NLP**: Topic Modelling, Sentiment Analysis, Text Embeddings (Word2Vec, BERT)
- **Data Mining**: Clustering, Classification, Association Rule Mining
- **Statistical Analysis**: Hypothesis testing, A/B testing
- **Data Visualization**: Matplotlib, Seaborn, or Plotly for data trends and model performance

### Expected Outputs

- Trained machine learning models (e.g. serialized model files)
- Recommendation scores for each content item
- User profiles with learned preferences
- Visualisations of model performance (e.g. precision-recall curves, ROC curves)
- Dataframes containing processed and analysed data

### Evaluation Criteria (verbatim)

> Evaluation will focus on:
> - Precision and recall of recommendations
> - Root mean squared error (RMSE) or other relevant metrics for rating prediction
> - Area under the ROC curve (AUC) for binary classification tasks
> - Statistical significance of A/B testing results
> - The use of cross validation to ensure model generalisation

---

## C. Grading Rubric (verbatim from template)

### 3rd Class (Minimum Pass)

- Implementation of a basic recommendation algorithm (e.g. collaborative filtering using a simple matrix factorisation technique)
- Use of a publicly available dataset
- Basic evaluation using a single metric
- Clear documentation of the data science pipeline

### 2:2 – 2:1 (Good)

All of the above, plus:

- Implementation of a more advanced machine learning model (e.g. a hybrid recommendation system)
- Data preprocessing and feature engineering
- Evaluation using multiple metrics and cross-validation
- Clear explanation of the strengths and weaknesses of the chosen approach

### 1st Class (Outstanding)

All of the above, plus:

- Development of a novel data science approach or adaptation of a state-of-the-art technique
- Creation or curation of a high-quality dataset
- Comprehensive evaluation, including comparisons with baseline models and statistical significance testing
- Detailed analysis of model performance and insights
- Potentially publishable results in a data science or educational technology venue

---

## D. Recommended Starting Resources (from template)

- Google's resources on recommendation systems
- Scikit-learn documentation for machine learning
- TensorFlow documentation for deep learning
- NLTK documentation for natural language processing
- Research papers on "Educational Data Mining" and "Learning Analytics" (Google Scholar or ACM Digital Library)
