# From Cart to Counter — Consumer Behaviour & Churn Analysis at DMart

This repository contains an applied statistics + machine learning project analyzing **customer shopping behaviour** at DMart.  
It demonstrates data preprocessing, hypothesis testing, churn-prediction modeling, market-basket analysis, classification of shopping preference, and visualization of results.

---

## 📌 Features
- Data cleaning and feature engineering for survey & transactional data  
- Chi-Square tests for demographic association with spending  
- Logistic Regression (and robust checks) for churn prediction  
- Random Forest & KNN for classifying In-store vs Online preference (with SMOTE)  
- Market Basket Analysis (Apriori + association rules) to find frequently-bought itemsets  
- Factor Analysis (KMO, Bartlett, Varimax) to extract latent factors  
- Pareto analysis to prioritize improvements (80/20)  
- Jupyter notebooks for step-by-step reproducibility  
- Exportable results (plots, tables, and model artifacts)

---

# --- Dataset ---
├─ Dmart_dataset.xlsx # main DMart survey data (315 responses)
├─ OtherSupermarkets_dataset.xlsx # other supermarkets (110 responses)


---

# MODEL — example runnable pipelines

Below are compact, copy-ready Python snippets you can paste into `src/` scripts (or run in notebooks). They assume your data columns match the project (e.g. `Avg_amount_spent`, `Shopping_duration`, `CheckoutEfficiency`, `PA`, `Overall_rating`, `Churn`, `Preference`, `Transaction_Items`).

> Save each snippet as a `.py` in `src/` (e.g. `logistic_regression.py`, `preference_classification.py`, `market_basket_analysis.py`) and run from repo root.

---

## --- LOGISTIC REGRESSION (Churn Prediction) ---
```python
# src/logistic_regression.py
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare(path):
    df = pd.read_excel(path, engine='openpyxl')
    # Example preprocessing (adapt to your columns)
    df = df.dropna(subset=['Churn'])
    # Encode categorical variables used as features
    df['Gender_m'] = (df['Gender']=='Male').astype(int)
    # Convert shopping duration categories to numeric (example mapping)
    dur_map = {'<6 months':0, '6 months - 1 year':1, '1-2 years':2, '2-5 years':3, '>5 years':4}
    if 'Shopping_duration' in df.columns:
        df['Shopping_duration_num'] = df['Shopping_duration'].map(dur_map).fillna(0)
    # Numeric feature set (tweak list as needed)
    features = ['Avg_amount_spent', 'Shopping_duration_num', 'CheckoutEfficiency', 'PA', 'Overall_rating', 'Gender_m']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df, features

def train_and_evaluate(df, features, target='Churn'):
    X = df[features]
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:,1]

    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.show()

    # Save artifacts
    joblib.dump(model, "results/models/logistic_model.joblib")
    joblib.dump(scaler, "results/models/logistic_scaler.joblib")
    print("Saved model & scaler to results/models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Dmart_dataset.xlsx")
    args = parser.parse_args()
    df, features = load_and_prepare(args.data)
    train_and_evaluate(df, features)

'''
## --- LOGISTIC REGRESSION (Churn Prediction) ---
--- RANDOM FOREST (Preference Classification; SMOTE for imbalance) ---
