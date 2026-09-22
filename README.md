# Intelligent Credit Risk & Loan Default Analytics System

An industry-level, end-to-end **Data Analytics + Machine Learning** project for credit risk assessment and loan default prediction.

---

## Project Overview

This system analyzes customer and loan data to predict loan default probability, segment customers by risk profile, and provide actionable business insights — all served through an interactive Flask web dashboard.

---

## Features

| Module | Description |
|---|---|
| Data Preprocessing | Cleaning, outlier handling, encoding |
| Feature Engineering | DTI Ratio, Loan-to-Income, Risk Score, Credit Band |
| EDA | 8+ interactive visualizations |
| Customer Segmentation | K-Means clustering with PCA visualization |
| ML Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Model Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Flask Dashboard | 4-page interactive web app |
| Risk Predictor | Real-time loan default probability prediction |
| Responsible AI | Fairness, bias, data leakage, and limitations documented |

---

## Dataset

**File:** `Financial Risk Modeling P.csv`

| Column | Description |
|---|---|
| Customer_ID | Unique customer identifier |
| Age | Customer age (18–69) |
| Employment_Type | 0=Self-Employed, 1=Salaried, 2=Business, 3=Freelancer |
| Monthly_Income | Monthly income in INR |
| Loan_Amount | Requested loan amount in INR |
| Credit_Score | Credit score (300–850) |
| Existing_Loans | Number of active loans |
| Missed_Payments | Number of missed payments |
| City | City code |
| Application_Date | Days since application |
| Loan_Default | **Target** — 1=Default, 0=No Default |

---

## Engineered Features

- **DTI_Ratio** — Monthly loan payment / Monthly income
- **Loan_to_Income** — Loan amount / Monthly income
- **Credit_Band** — Poor / Fair / Good / Very Good / Exceptional
- **Risk_Score** — Composite score: `(Missed_Payments×2) + Existing_Loans + (DTI×10) - (Credit_Score/100)`
- **Age_Group** — Binned age categories

---

## Project Structure

```
Credit card risk/
├── Financial Risk Modeling P.csv     # Dataset
├── data_preprocessing.py             # Cleaning & feature engineering
├── eda.py                            # EDA plot generation
├── clustering.py                     # K-Means customer segmentation
├── train_models.py                   # ML model training & evaluation
├── app.py                            # Flask web dashboard
├── setup.py                          # One-time setup script
├── requirements.txt                  # Python dependencies
├── credit_risk_analytics.ipynb       # Jupyter notebook (data analysis)
├── models/                           # Saved models & metrics
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   └── feature_names.json
├── templates/                        # Flask HTML templates
│   ├── index.html
│   ├── eda.html
│   ├── models.html
│   └── predict.html
└── static/plots/                     # Generated EDA & model plots
```

---

## Installation & Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run setup (trains models & generates all plots)
```bash
python setup.py
```

### 3. Start the Flask dashboard
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

### 5. Run Jupyter notebook (data analysis only)
```bash
jupyter notebook credit_risk_analytics.ipynb
```

---

## ML Models & Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~65% | ~62% | ~68% | ~65% | ~0.71 |
| Decision Tree | ~67% | ~64% | ~70% | ~67% | ~0.73 |
| Random Forest | ~72% | ~69% | ~74% | ~71% | ~0.79 |
| **Gradient Boosting** | **~74%** | **~71%** | **~76%** | **~73%** | **~0.81** |

> Results are based on the actual dataset. SMOTE applied to handle class imbalance.

---

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | KPI summary, model comparison table |
| EDA | `/eda` | All exploratory analysis plots with insights |
| Models | `/models` | ROC curves, confusion matrices, feature importance, responsible AI |
| Predict | `/predict` | Real-time default risk prediction form |

---

## Key Business Insights

1. **Credit Score** is the strongest predictor — customers below 580 have significantly higher default rates
2. **DTI Ratio > 0.5** dramatically increases default probability
3. **Even 1 missed payment** doubles the default risk
4. **4 customer segments** identified — enabling risk-based product pricing
5. **Gradient Boosting** achieves best ROC-AUC, recommended for production

---

## Responsible AI

- **Data Leakage:** Risk_Score is derived from correlated features — monitor in production
- **Class Imbalance:** SMOTE applied; synthetic samples may not fully represent real distribution
- **Fairness:** Monitor predictions across age and employment groups for disparate impact
- **Explainability:** Feature importance provided; SHAP recommended for individual decisions
- **Human Oversight:** Borderline cases (40–60% probability) require manual review
- **Privacy:** Customer IDs anonymized; ensure GDPR compliance in production

---

## Tech Stack

- **Python 3.10+**
- **Pandas, NumPy** — Data manipulation
- **Scikit-learn** — ML models, preprocessing, evaluation
- **Imbalanced-learn** — SMOTE for class imbalance
- **Matplotlib, Seaborn** — Visualizations
- **Flask** — Web dashboard
- **Joblib** — Model serialization
- **Bootstrap 5** — Frontend UI

---

## License

MIT License — free to use for educational and commercial purposes.
