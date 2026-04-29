# Bank Customer Churn Prediction

A machine learning project that predicts whether a bank customer will exit using structured customer data. The notebook demonstrates data cleaning, feature engineering, model selection, and evaluation using F1 score and AUC-ROC.

## Project Overview

- Objective: predict customer churn for a retail bank.
- Primary metric: F1 score.
- Secondary metric: AUC-ROC.
- Data source: `Churn.csv`.

## Data Preparation

- Dropped irrelevant identifiers: `RowNumber`, `CustomerId`, and `Surname`.
- Encoded categorical features:
  - `Geography`: one-hot encoding.
  - `Gender`: binary encoding.
- Scaled numerical features: `CreditScore`, `Age`, `Tenure`, `Balance`, and `EstimatedSalary`.
- Split data into train/validation/test sets with a 60/20/20 ratio.

## Modeling Approach

- Evaluated baseline models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Addressed class imbalance with both upsampling and `class_weight='balanced'`.
- Selected the best-performing model based on validation F1 score.

## Final Model & Results

- Best model: `RandomForestClassifier(random_state=12345, n_estimators=40, max_depth=8, class_weight='balanced')`
- Final F1 score: above the project requirement and approximately `0.639`.
- AUC-ROC shows how well the model ranks customers by churn probability.

## Key Insights

- Churn appears more pronounced among customers aged 38–51.
- Higher account balances may correlate with increased churn risk.
- Credit score outliers below 400 show a higher proportion of churned customers.
- Using balanced class weights improved model performance more than simple upsampling.

## Usage

1. Open `Churn_ml.ipynb`.
2. Install the required packages if needed: `pandas`, `scikit-learn`, `matplotlib`.
3. Run the notebook cells sequentially to reproduce the analysis and final evaluation.

## Notes

- The notebook includes visual analysis of age, credit score, salary, and balance distributions by churn status.
- The final evaluation uses the held-out test set to verify both F1 and AUC-ROC performance.
