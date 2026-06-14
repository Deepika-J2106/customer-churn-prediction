# 📉 Customer Churn Prediction

> Predicting which telecom customers are likely to churn using machine learning — with explainable AI and business-driven insights.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Sklearn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![RandomForest](https://img.shields.io/badge/Random%20Forest-Tuned-green) ![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

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
├── notebook.ipynb                  ← Full analysis, modeling, and explainability
├── README.md                       ← Project overview (you are here)
├── requirements.txt                ← Python dependencies
│
└── images/
    ├── churn_by_contract.png           ← EDA: churn rate by contract type
    ├── monthly_charges_dist.png        ← EDA: monthly charges distribution
    ├── tenure_churn.png                ← EDA: tenure vs churn
    ├── shap_bar.png                    ← SHAP: mean feature importance
    ├── shap_summary.png                ← SHAP: beeswarm feature impact
    ├── shap_force.png                  ← SHAP: single customer explanation
    ├── precision_recall_curve.png      ← Threshold tuning curve
    └── confusion_matrix.png            ← Final confusion matrix
```

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Analyzed churn distribution across contract type, tenure, payment method, and monthly charges
- Found that **month-to-month contract customers churn at 3x the rate** of yearly contract customers
- Discovered that **electronic check users** churn significantly more than auto-pay users
- Identified 11 hidden missing values in `TotalCharges` stored as whitespace strings (not NaN) — caught using `pd.to_numeric(..., errors='coerce')`
- Churned customers cluster at **higher monthly charges ($70–100)** vs loyal customers who cluster at **~$20/month**

### 2. Preprocessing & Feature Engineering
- Dropped `customerID` (non-predictive identifier)
- Mapped target variable `Churn` → Yes/No to 1/0
- One-hot encoded all categorical features using `pd.get_dummies(drop_first=True)` — expanded from 20 to 30 features
- Applied **stratified train-test split** (80/20) to preserve class ratio
- Applied **SMOTE on training data only** to handle class imbalance — avoided data leakage by not applying to test set
- Deliberately **excluded StandardScaler** for tree-based models — scaling degraded XGBoost AUC-ROC from 0.81 to 0.63 during debugging

### 3. Model Training & Comparison

| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.8109 |
| XGBoost | 0.8144 |
| **Random Forest** | **0.8210 ✅ winner** |

> Tree-based models (Random Forest, XGBoost) do not require feature scaling as they split on thresholds, not distances. Applying StandardScaler before SMOTE distorted synthetic samples and significantly degraded model performance — identified and resolved during debugging.

### 4. Hyperparameter Tuning
- Tuned Random Forest using `RandomizedSearchCV` (50 iterations, 5-fold cross-validation)
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- AUC-ROC improved from **0.8210 → 0.8243** after tuning

### 5. Model Explainability with SHAP
- Used `shap.TreeExplainer` to compute feature contributions for every prediction
- Generated summary bar plot, beeswarm plot, and individual force plot

### 6. Threshold Tuning
- Default threshold (0.5) optimizes accuracy — not suitable for imbalanced churn problems
- Evaluated thresholds from 0.3 to 0.5 on precision, recall, and F1

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.478 | 0.818 | 0.604 |
| **0.35** | **0.498** | **0.775** | **0.607 ✅** |
| 0.40 | 0.518 | 0.727 | 0.605 |
| 0.45 | 0.556 | 0.668 | 0.607 |
| 0.50 | 0.565 | 0.604 | 0.584 |

> Chose **0.35** — missing a churner (false negative) costs more than a false alert (false positive). Higher recall means more churners caught for the retention team to act on.

---

## 📊 Key Visualizations

### Churn rate by contract type
![Churn by Contract](images/churn_by_contract.png)

### Monthly charges distribution by churn
![Monthly Charges](images/monthly_charges_dist.png)

### SHAP beeswarm — feature impact on churn
![SHAP Beeswarm](images/shap_summary.png)

### Confusion matrix (threshold = 0.35)
![Confusion Matrix](images/confusion_matrix.png)

---

## 💡 Key Insights from SHAP Analysis

1. **Electronic check payment is the strongest churn predictor** — customers on manual payment methods are significantly more likely to churn than those on automatic payments (credit card, bank transfer)

2. **Fiber optic internet customers churn at higher rates** despite it being a premium service — likely driven by higher monthly charges and more competitive alternatives in the market

3. **Tenure is the clearest churn signal** — new customers (low tenure) are at the highest risk. The first 12 months are the most critical retention window

4. **Two-year contracts strongly reduce churn** — customers on longer contracts almost never leave. Incentivizing contract upgrades is the single most effective retention lever

5. **High monthly charges correlate with churn** — customers paying $70–100/month churn more, especially when combined with month-to-month contracts

### Business Recommendation
Target retention campaigns at:
- New customers (tenure < 12 months)
- Month-to-month contract holders
- Fiber optic subscribers paying via electronic check
- Customers without online backup or device protection (lower engagement = higher churn risk)

---

## ✅ Final Model Performance

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.8243** |
| Precision (Churn) | 0.498 |
| **Recall (Churn)** | **0.775** |
| F1-Score (Churn) | 0.607 |
| Threshold | 0.35 |

### Confusion Matrix Breakdown
| | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actually No Churn** | 743 ✅ | 292 ❌ |
| **Actually Churn** | 84 ❌ | 290 ✅ |

> Out of **374 actual churners** — model caught **290 (77.5%)**, missed only 84 (22.5%)

---

## 🐛 Debugging Note

During model comparison, XGBoost scored unexpectedly low (AUC-ROC: 0.63) compared to Logistic Regression (0.81). Root cause identified: `StandardScaler` was applied before SMOTE, which distorted the synthetic minority samples generated by SMOTE — specifically hurting tree-based models that are sensitive to the feature space distribution used during resampling. Removing scaling restored XGBoost to expected performance (0.81+). This highlights the importance of understanding the order of operations in ML pipelines.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab |

---

## ▶️ How to Run

1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
3. Upload the CSV to Colab using:
```python
from google.colab import files
files.upload()
```
4. Run all cells top to bottom

Or install dependencies locally:
```bash
pip install -r requirements.txt
jupyter notebook
```

---

## 📝 Learnings & Challenges

- **Class imbalance** was the first hurdle — accuracy was misleading (a model predicting "No churn" for everyone gets 73% accuracy). Switched to AUC-ROC and recall as primary metrics.

- **Hidden missing values** in `TotalCharges` were stored as empty strings, not NaN — `pd.to_numeric(..., errors='coerce')` was required to catch them. A standard `.isnull()` check alone would have missed these.

- **Scaling hurt tree models** — applying StandardScaler before SMOTE caused XGBoost AUC-ROC to drop from 0.81 to 0.63. Identifying and removing the scaler restored expected performance. Lesson: always understand which preprocessing steps are appropriate for which model families.

- **Threshold tuning had more business impact than hyperparameter tuning** — tuning improved AUC-ROC by only +0.003, but moving the threshold from 0.5 to 0.35 increased churner recall from 60% to 77.5% — catching 64 additional churners in the test set alone.

---

## requirements.txt

```
pandas==2.1.0
numpy==1.25.0
scikit-learn==1.3.0
xgboost==1.7.6
imbalanced-learn==0.11.0
shap==0.43.0
matplotlib==3.7.2
seaborn==0.12.2
```

---

## 🔗 Connect

**[Your Name]**  
[LinkedIn](https://www.linkedin.com/in/deepika-jalamanchili/) · [GitHub](https://github.com/Deepika-J2106) · [deepikajalamanchili.9999@gmail.com](mailto:deepikajalamanchili.9999@gmail.com)
