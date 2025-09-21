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


    df, features = load_and_prepare(args.data)
    train_and_evaluate(df, features)

'''
## --- LOGISTIC REGRESSION (Churn Prediction) ---
--- RANDOM FOREST (Preference Classification; SMOTE for imbalance) ---
