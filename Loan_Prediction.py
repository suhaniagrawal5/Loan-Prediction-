import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
df = pd.read_csv('train_ctrUa4K.csv')
df.rename(columns={'total_incone_per_annum': 'total_income_per_annum'}, inplace=True)

# 2. Derive/Fix Engineered Features
if 'total_income_per_annum' not in df.columns:
    df['total_income_per_annum'] = (df['ApplicantIncome'] + df['CoapplicantIncome']) * 12

if 'Loan_total' not in df.columns:
    df['Loan_total'] = df['LoanAmount'] * 1000  # LoanAmount is usually in thousands

if 'loan_in_years' not in df.columns:
    df['loan_in_years'] = df['Loan_Amount_Term'] / 12

# Apply Log Transformations to skewed income/loan columns to handle outliers
df['log_total_income'] = np.log1p(df['total_income_per_annum'])
df['log_loan_total'] = np.log1p(df['Loan_total'])

# 3. Define Features and Target
numerical_cols = ['loan_in_years', 'log_loan_total', 'log_total_income']
categorical_cols = ['Education', 'Self_Employed', 'Married']

X = df[numerical_cols + categorical_cols]
y = df['Loan_Status'].map({'Y': 1, 'N': 0}) # Map target to numeric binary values

# Split Data clean into train and test sets FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build Preprocessing Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# 5. Full Model Pipeline with Logistic Regression
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
])

# Fit on training set only
full_pipeline.fit(X_train, y_train)

# Evaluate on training and validation set
train_acc = accuracy_score(y_train, full_pipeline.predict(X_train))
test_acc = accuracy_score(y_test, full_pipeline.predict(X_test))

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}") # Quick test using Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
])

gb_pipeline.fit(X_train, y_train)
print(f"Gradient Boosting Test Accuracy: {accuracy_score(y_test, gb_pipeline.predict(X_test)):.4f}")
