# Task 8: Customer Churn Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📋 Project Overview

This project implements an **end-to-end machine learning pipeline** for predicting customer churn using the Telco Customer Churn dataset. The pipeline includes data preprocessing, model training, hyperparameter tuning, and model export for production use.

## 🎯 Objective

Build a reusable and production-ready machine learning pipeline that can predict whether a customer will churn (cancel their service) based on their account information, demographics, and service usage patterns.

## 📊 Dataset

**Telco Customer Churn Dataset**
- **Source:** IBM Telco Customer Churn Data
- **Samples:** 7,043 customer records
- **Features:** 20 features (demographics, account info, services subscribed)
- **Target:** Churn (Yes/No)
- **Churn Rate:** 26.5% (imbalanced dataset)

### Features Include:
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account Info:** tenure, Contract, PaperlessBilling, PaymentMethod
- **Services:** PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV
- **Charges:** MonthlyCharges, TotalCharges

## 🏆 Results

### Model Performance

| Model | Accuracy | ROC-AUC | Best Parameters |
|-------|----------|---------|-----------------|
| Logistic Regression (Baseline) | 80.55% | - | - |
| Random Forest (Baseline) | 78.57% | - | - |
| **Random Forest (Tuned)** | **80.13%** | **84.27%** | `max_depth=10, min_samples_split=5, n_estimators=150` |

### Classification Report (Tuned Model)

```
              precision    recall  f1-score   support
           0       0.84      0.90      0.87      1035
           1       0.66      0.52      0.58       374

    accuracy                           0.80      1409
```

- **Class 0 (No Churn):** 90% recall - Model is excellent at identifying loyal customers
- **Class 1 (Churn):** 52% recall - Moderate ability to predict churn (can be improved with imbalanced data techniques)

## 🏗️ Pipeline Architecture

```
Raw Data
    ↓
Preprocessing Pipeline
├── Numeric Features → Median Imputation → StandardScaler
└── Categorical Features → Constant Imputation → OneHotEncoder
    ↓
ColumnTransformer
    ↓
Random Forest Classifier
    ↓
GridSearchCV (Hyperparameter Tuning)
    ↓
Best Model Pipeline (saved with joblib)
```

## 📁 Project Structure

```
Task8-Customer-Churn-Pipeline/
│
├── train_pipeline.py          # Main training script
├── predict.py                 # Prediction script for single/batch predictions
├── test_customers.py          # Test with custom customer data
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── outputs/                   # Generated files (after running)
│   ├── churn_pipeline.pkl     # Saved pipeline model
│   ├── best_params.json       # Best hyperparameters
│   └── .gitignore            # Excludes large .pkl files
│
└── telco_churn.csv            # Dataset (downloaded automatically)
```

## 🚀 Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_pipeline.py
```

This will:
- Load and preprocess the Telco dataset
- Build preprocessing pipelines
- Train Logistic Regression and Random Forest
- Perform GridSearchCV hyperparameter tuning
- Save the best model to `outputs/churn_pipeline.pkl`

### 3. Make Predictions

**Single customer prediction:**
```bash
python predict.py
```

**Custom batch predictions:**
```bash
python test_customers.py
```

### 4. Use in Your Own Code

```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('outputs/churn_pipeline.pkl')

# Predict for new customer
customer_data = pd.DataFrame([{
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'tenure': 24,
    'InternetService': 'Fiber optic',
    'Contract': 'Month-to-month',
    'MonthlyCharges': 85.5,
    'TotalCharges': 1500.0
}])

prediction = pipeline.predict(customer_data)  # 0 = No Churn, 1 = Churn
probability = pipeline.predict_proba(customer_data)[:, 1]  # Churn probability
```

## 🔧 Pipeline Components

### Preprocessing Steps

| Feature Type | Steps |
|--------------|-------|
| **Numeric** | Median imputation → Standard scaling |
| **Categorical** | Constant imputation → One-hot encoding (drop first) |

### Models Tested

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble method with hyperparameter tuning

### Hyperparameter Tuning (GridSearchCV)

```python
param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5]
}
# 3-fold cross-validation, ROC-AUC scoring
```

## 📈 Key Insights

1. **Most Important Features (from Random Forest):**
   - Tenure (longer tenure → less likely to churn)
   - Contract type (month-to-month customers churn more)
   - Monthly charges (higher charges may increase churn risk)

2. **Business Recommendations:**
   - Target month-to-month customers for retention offers
   - Focus on early tenure customers (first 12 months)
   - Consider loyalty programs for high-risk segments

## 🛠️ Technologies Used

- **Python 3.11** - Core programming language
- **Pandas/NumPy** - Data manipulation
- **Scikit-learn** - ML pipeline, preprocessing, models
- **Joblib** - Model serialization
- **Matplotlib/Seaborn** - Visualization (optional)

## 📦 Dependencies

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
```

## 🎓 Skills Gained

- ✅ Building end-to-end ML pipelines with `Pipeline` and `ColumnTransformer`
- ✅ Implementing preprocessing (scaling, encoding, imputation)
- ✅ Hyperparameter tuning with `GridSearchCV`
- ✅ Model export and reusability with `joblib`
- ✅ Production-ready practices for ML deployment
- ✅ Handling imbalanced datasets
- ✅ Creating reusable prediction interfaces

## 🔄 Future Improvements

- Handle class imbalance with SMOTE or class weights
- Add more models (XGBoost, Gradient Boosting)
- Implement model monitoring and retraining pipeline
- Add SHAP explainability
- Create FastAPI web service
- Add Docker containerization

## 📧 Contact

**Muhammad Latif**  
AI/ML Developer

- **GitHub:** [github.com/Muhammadlatifkhan](https://github.com/Muhammadlatifkhan)
- **Email:** muhammad.latif@example.com
- **LinkedIn:** [linkedin.com/in/muhammad-latif-khan](https://linkedin.com/in/your-profile)

## 📅 Project Status

✅ **COMPLETED** - May 2026

---

## 📝 Sample Output

### Training Output:
```
============================================================
TASK 8: CUSTOMER CHURN PREDICTION PIPELINE
============================================================

[1] Loading Telco Churn Dataset...
✅ Dataset loaded successfully! Shape: (7043, 21)

[2] Data Preprocessing...
✅ Preprocessing complete!

[3] Preparing features...
   Features: 19
   Churn rate: 26.54%

[4] Training baseline models...
   Logistic Regression: 0.8055
   Random Forest: 0.7857

[5] Hyperparameter tuning...
   Best params: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 150}
   Best CV score: 0.8440

[6] Final Results:
   Accuracy: 80.13%
   ROC-AUC: 84.27%

✅ TASK 8 COMPLETED SUCCESSFULLY!
```

### Prediction Output:
```
Loading pipeline...
✅ Pipeline loaded successfully!

📊 Sample Prediction:
   Churn Prediction: Yes
   Churn Probability: 54.36%
```

---

⭐ **If you found this project helpful, please give it a star on GitHub!**
```
