"""
Task 8: End-to-End ML Pipeline for Customer Churn Prediction
Author: Muhammad Latif
Date: May 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 8: CUSTOMER CHURN PREDICTION PIPELINE")
print("=" * 60)

# ========== 1. LOAD AND EXPLORE DATA ==========
print("\n[1] Loading Telco Churn Dataset...")

# Try multiple URLs
urls = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/csalinasonline/Telco-Customer-Churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "telco_churn.csv"  # Local file
]

df = None
for url in urls:
    try:
        print(f"   Trying: {url[:50]}...")
        if url.endswith('.csv') and os.path.exists(url):
            df = pd.read_csv(url)
        else:
            df = pd.read_csv(url)
        print(f"   ✅ Success!")
        break
    except:
        continue

if df is None:
    print("\n❌ Could not load dataset. Please download manually:")
    print("1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("2. Download and save as 'telco_churn.csv' in this folder")
    print("3. Run this script again")
    exit()

print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns[:5])}...")

# ========== 2. DATA PREPROCESSING ==========
print("\n[2] Data Preprocessing...")

# Remove customerID column if exists
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Handle TotalCharges
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert SeniorCitizen to categorical
if 'SeniorCitizen' in df.columns:
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')

# Encode target
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"✅ Preprocessing complete!")

# ========== 3. PREPARE FEATURES ==========
print("\n[3] Preparing features...")

X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"   Features: {X.shape[1]}")
print(f"   Churn rate: {y.mean():.2%}")

# ========== 4. TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ========== 5. COLUMN TYPES ==========
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"   Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

# ========== 6. BUILD PIPELINE ==========
print("\n[4] Building pipeline...")

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_cols),
    ('categorical', categorical_pipeline, categorical_cols)
])

# Create pipelines
log_reg = Pipeline([('preprocessor', preprocessor), 
                   ('classifier', LogisticRegression(random_state=42, max_iter=1000))])

rf = Pipeline([('preprocessor', preprocessor), 
              ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))])

# ========== 7. TRAIN BASELINE ==========
print("\n[5] Training baseline models...")

log_reg.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, log_reg.predict(X_test))
print(f"   Logistic Regression: {lr_acc:.4f}")

rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"   Random Forest: {rf_acc:.4f}")

# ========== 8. GRID SEARCH ==========
print("\n[6] Hyperparameter tuning...")

param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\n   Best params: {grid_search.best_params_}")
print(f"   Best CV score: {grid_search.best_score_:.4f}")

# ========== 9. EVALUATE ==========
print("\n[7] Evaluating best model...")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n📊 Final Results:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   ROC-AUC: {roc_auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ========== 10. SAVE MODEL ==========
print("\n[8] Saving model...")

os.makedirs('outputs', exist_ok=True)
joblib.dump(best_model, 'outputs/churn_pipeline.pkl')
print(f"   ✅ Saved to: outputs/churn_pipeline.pkl")

# Save parameters
import json
with open('outputs/best_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f, indent=4)

print("\n" + "=" * 60)
print("✅ TASK 8 COMPLETED SUCCESSFULLY!")
print("=" * 60)