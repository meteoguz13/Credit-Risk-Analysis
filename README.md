# Credit Risk Assessment Tool

An end-to-end credit default risk model with an interactive web app. The model predicts the probability that a borrower will experience serious delinquency (90+ days past due), and explains *why* through individual factor analysis.

**Live app:** [Streamlit link here]

---

## What this project does

Given a borrower's profile (income, credit usage, payment history, etc.), the tool returns:
- A **risk score** (probability of default)
- A **risk level** (low / medium / high)
- The **factors** that pushed the score up or down

It is built to mirror the kind of analysis done in a real bank credit department, with a focus on **sound methodology** and **explainability** rather than just a high score.

---

## The data

I used the **Give Me Some Credit** dataset (Kaggle) — real anonymized US consumer credit data, ~150,000 records. I chose real data over synthetic data on purpose: synthetic data tends to be circular (a model learns the rules you used to generate it), while real data forces the model to learn genuine risk relationships.

**Target:** `SeriousDlqin2yrs` (1 = serious delinquency, 0 = clean). The data is imbalanced — only ~6.7% are defaults.

### Adapting to a Turkish context

The original income figures are in US dollars. To make the tool intuitive for a Turkish audience, I scaled income into Turkish lira terms **without touching its relationship to default**. Only a monotonic transformation was applied (multiply by a constant derived from the 2026 Turkish minimum wage and TÜİK average/minimum income ratio), so the ranking of borrowers by income — and therefore the risk relationship — is preserved. Income was floored at the minimum wage and capped at the 99th percentile.

This is disclosed honestly: the app uses **US data scaled to a Turkish income context, with risk relationships preserved** — not actual Turkish credit data.

---

## Key decisions (and why)

Every cleaning and modeling decision was tested against the data, not made by rule of thumb.

### Outliers — tested, not blindly removed
For each suspicious value, I checked its actual default behavior before deciding. Example: `DebtRatio` had absurd values (e.g. 1159). I confirmed these came almost entirely from rows with missing income (when income is missing, the debt-to-income ratio can't be computed properly). These rows did **not** default more than average, so the high values were not a genuine risk signal — they were artifacts. I replaced them with the median of the valid group.

### Missing values — informed imputation
- **Income** (~20% missing): filled with the **median income of the borrower's age group**, not a global median. Missing-income borrowers turned out to be a distinct group (older, lower default rate), so age-group imputation respects that structure.
- **Dependents**: filled with 0 (the most common value; "unspecified" usually means none).

### Data leakage — prevented
This is the single most important methodological point. All imputation values, caps, and medians are learned **only from the training set**, then applied to both train and test. Computing them on the full dataset before splitting would leak test information and inflate the score. (I verified the leakage effect was small here — AUC moved by ~0.001 — but the pipeline is now methodologically clean and defensible.)

### Multicollinearity — resolved
The three delinquency columns (30-59, 60-89, 90+ days) were ~0.99 correlated. I combined them into a single `toplam_gecikme` (total delinquencies) feature and dropped the originals, plus added a binary `gecikme_var` (any delinquency). This matters for the logistic regression baseline.

### Feature engineering — tested, then trimmed
I engineered several features (age buckets, usage categories, delinquency flag) and tested each against default behavior. Only the ones that actually helped were kept. The binary "has any past delinquency" flag was the strongest single signal (clean: ~2.7% default, with delinquency: ~22%).

---

## Modeling

I built a baseline first, then a stronger model — to measure whether complexity actually helps.

| Model | Test ROC-AUC | Notes |
|---|---|---|
| Logistic Regression (baseline) | ~0.855 | scaled, class_weight balanced |
| **XGBoost (tuned)** | **~0.867** | GridSearch, scale_pos_weight |

- **Metric:** ROC-AUC and precision/recall — **not accuracy**, which is misleading on imbalanced data (predicting "everyone clean" would score 93% accuracy but catch zero defaults).
- **Class imbalance:** handled with `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (logistic).
- XGBoost won. ~0.867 is a solid score for this dataset — the realistic ceiling is around 0.87.

### Why no single cutoff
A credit score isn't a yes/no. Choosing a threshold is a **business decision**: catching a defaulter you missed is usually more expensive than rejecting a good borrower. Rather than hard-coding one threshold, the app shows a **graded score + level + reasons**, preserving nuance (e.g. someone with a maxed-out card but no missed payments isn't simply "rejected").

---

## Explainability (SHAP)

I used **SHAP** to verify the model learned sensible things — and it did. The top drivers are exactly what a credit analyst would expect: past delinquency, credit utilization, age, and number of 90-day delinquencies. The model isn't a black box guessing randomly; it reflects genuine risk logic. In the app, this is translated into plain-language factor explanations for non-technical users.

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit

## Project structure

```
├── app.py              # Streamlit web app
├── credit_risk.py      # data prep, modeling, evaluation pipeline
├── kredi_model.pkl     # trained model
├── feature_isimleri.pkl
├── Data/               # raw dataset
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## A note on honesty

This is a portfolio / learning project. It uses US data scaled to a Turkish context, and it's a risk **indicator**, not a production credit-decision system. The value is in the methodology: tested decisions, leakage-free pipeline, proper metrics, and explainability.
