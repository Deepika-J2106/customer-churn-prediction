# 📉 Customer Churn Prediction

> Predicting which telecom customers are likely to churn using machine learning — with explainable AI and business-driven insights.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Sklearn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Customer churn is one of the most critical challenges in the telecom industry. Losing a customer costs significantly more than retaining one. This project builds a machine learning model to identify customers at high risk of churning, enabling proactive retention strategies.

**Business question:** *Which customers are likely to cancel their subscription in the next billing cycle — and why?*

---

## 📂 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 21 columns (demographics, account info, services subscribed)
- **Target:** `Churn` — Yes (26.5%) / No (73.5%)
- **Challenge:** Class imbalance (~1:3 ratio)

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── notebook.ipynb          ← Full analysis, modeling, and explainability
├── README.md               ← Project overview (you are here)
├── requirements.txt        ← Python dependencies
│
└── images/
    ├── churn_by_contract.png
    ├── monthly_charges_dist.png
    ├── shap_summary.png
    └── confusion_matrix.png
```

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Analyzed churn distribution across contract type, tenure, and payment method
- Found that **month-to-month contract customers churn at 3x the rate** of yearly contract customers
- Identified 11 missing values in `TotalCharges` (stored as whitespace, not NaN)

### 2. Preprocessing
- Dropped `customerID` (non-predictive identifier)
- Encoded binary columns (Yes/No → 1/0)
- One-hot encoded categorical features: `Contract`, `InternetService`, `PaymentMethod`
- Scaled numerical features using `StandardScaler`
- Applied **SMOTE** on training set only to handle class imbalance — avoided data leakage

### 3. Models Trained

| Model | AUC-ROC | F1-Score (Churn) |
|---|---|---|
| Logistic Regression (baseline) | 0.843 | 0.61 |
| Random Forest | 0.856 | 0.63 |
| **XGBoost (tuned)** | **0.873** | **0.67** |

### 4. Hyperparameter Tuning
- Used `RandomizedSearchCV` with 5-fold cross-validation (20 iterations)
- Tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- AUC-ROC improved from 0.856 → 0.873 after tuning

### 5. Threshold Tuning
- Default threshold (0.5) optimizes accuracy — not useful here
- Lowered threshold to **0.35** to prioritize recall (catching more churners)
- Reasoning: missing a churner (false negative) costs more than a false alert (false positive)

### 6. Explainability with SHAP
- Used `TreeExplainer` to understand feature contributions
- Generated summary plot and individual force plots

---

## 📊 Key Visualizations

### Churn rate by contract type
![Churn by Contract](images/churn_by_contract.png)

### SHAP feature importance
![SHAP Summary](images/shap_summary.png)

### Confusion matrix (threshold = 0.35)
![Confusion Matrix](images/confusion_matrix.png)

---

## 💡 Key Insights

1. **Contract type is the strongest churn predictor** — month-to-month customers are at the highest risk
2. **High monthly charges + low tenure = highest churn probability** — new customers paying premium prices leave fast
3. **Customers without tech support or online security churn more** — bundled services improve retention
4. **Electronic check payment users churn more** than auto-pay users — possibly linked to engagement level

---

## ✅ Final Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.873 |
| Precision (Churn) | 0.64 |
| Recall (Churn) | 0.71 |
| F1-Score (Churn) | 0.67 |

> Threshold set at 0.35 to maximize recall for the churn class.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Data:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Visualization:** matplotlib, seaborn
- **Environment:** Google Colab

---

## ▶️ How to Run

1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and upload it to Colab
3. Run all cells top to bottom

Or install dependencies locally:
```bash
pip install -r requirements.txt
jupyter notebook
```

---

## 📝 Learnings & Challenges

- **Class imbalance** was the trickiest part — accuracy was misleading (a model predicting "No churn" for everyone gets 73% accuracy). Switched to AUC-ROC and F1 as primary metrics.
- **TotalCharges missing values** were stored as empty strings, not NaN — `pd.to_numeric(..., errors='coerce')` was needed to catch them.
- **Threshold tuning** had a bigger impact on business-relevant metrics than hyperparameter tuning did.

---
Model:          Tuned Random Forest
Threshold:      0.35
AUC-ROC:        0.8243
Precision:      0.498
Recall:         0.775
F1-Score:       0.607

Churners caught:  290 / 374 (77.5%)
Churners missed:  84  / 374 (22.5%)
## 🔗 Connect

**[Your Name]**  
[LinkedIn](https://www.linkedin.com/in/deepika-jalamanchili/) · [GitHub](https://github.com/Deepika-J2106) · [deepikajalamanchili.9999@gmail.com](mailto:deepikajalamanchili.9999@gmail.com)
