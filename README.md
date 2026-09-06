# 🏦 Loan Approval Prediction

A machine learning classification project that predicts whether a loan application will be approved or rejected based on applicant financial metrics and demographics.

---

### 🔑 Key Highlights

* **Objective:** Binary classification to predict loan approval (`Y` or `N`).
* **Feature Engineering:** Derived yearly total income, normalized loan amounts, converted loan terms to years, and applied log transformations to fix right-skewed financial metrics.
* **Data Preprocessing:** Handled missing data via median/mode imputation, standard feature scaling, and one-hot encoding for categorical variables.
* **Leakage Prevention:** Managed preprocessing and model fitting strictly through `scikit-learn` pipelines to ensure zero data leakage between training and testing sets.
* **Model Handling:** Evaluated Logistic Regression with balanced class weights to address target dataset imbalance.
