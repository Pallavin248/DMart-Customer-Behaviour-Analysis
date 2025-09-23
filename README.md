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


---

## 📊 Methodology
1. **Data Collection**: 425 responses collected via structured survey.  
2. **Analysis Tools**: SPSS, Python (sklearn, statsmodels, mlxtend for MBA).  
3. **Steps**:
   - Descriptive analysis of demographics & shopping trends  
   - Chi-Square Tests for independence  
   - Logistic Regression for churn prediction  
   - Factor Analysis (KMO = 0.691, Bartlett’s p < 0.001)  
   - KNN & Random Forest for classification  
   - Market Basket Analysis using Apriori algorithm  

---

## 📈 Results
- **Chi-Square Test**: Income influences spending, Gender does not.  
- **Logistic Regression**: Product availability, checkout efficiency, and occupation increase churn risk.  
- **Factor Analysis**: 3 key factors → Socioeconomic traits, Basket & duration, Store efficiency.  
- **Random Forest**: Best model for customer preference classification (80.95%).  
- **MBA Findings**: Strong product bundles (Groceries + Packaged Food, Dairy + Beverages).  

---

## 📑 Files
- `docs/DMart-Consumer-Behaviour.pptx` → Research presentation  
- `analysis/` → Chi-Square, Logistic Regression, FA results  
- `models/` → ML models in Jupyter Notebooks  
- `images/` → Graphs (ROC, Scree Plot, MBA visualization)  

---


# --- Dataset ---
├─ Dmart_dataset.xlsx # main DMart survey data (315 responses)
├─ OtherSupermarkets_dataset.xlsx # other supermarkets (110 responses)

