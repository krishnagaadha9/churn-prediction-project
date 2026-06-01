# Customer Churn Prediction

## Problem
A bank loses revenue when customers cancel their credit cards. 
This project predicts which customers are likely to churn so 
the bank can intervene early.

## Dataset
BankChurners dataset — 10,127 customers, 21 features
Source: Kaggle (sakshigoyal7/credit-card-customers)

## Approach
- SQL EDA to identify churn patterns
- SMOTE to handle class imbalance (16% churn rate)
- Compared 4 models: Logistic Regression, Random Forest, 
  XGBoost, LightGBM

## Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 85.5% | 0.905 |
| Random Forest | 95.2% | 0.983 |
| XGBoost | 96.5% | 0.990 |
| LightGBM | 96.3% | 0.991 |

**Best model: LightGBM — ROC-AUC 0.9906**

## Key Findings
- Transaction count and amount are the strongest churn signals
- Demographics (age, gender) are weak predictors
- High risk threshold: <51 transactions AND <$2,772 spend
- Customers with fewer bank products churn significantly more

## Tech Stack
Python, Pandas, SQL (SQLite), Scikit-learn, 
XGBoost, LightGBM, SMOTE, Matplotlib, Seaborn

## Files
- churn_prediction_project.ipynb — full analysis
- lightgbm_churn_model.pkl — saved model
