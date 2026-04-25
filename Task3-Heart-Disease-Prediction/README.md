# Task 3: Heart Disease Prediction

## 🎯 Objective
Build a classification model to predict whether a person is at risk of heart disease based on their health data using medical parameters.

## 📊 Dataset
**UCI Heart Disease Dataset** - 303 patient records with 14 features:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Gender (0=female, 1=male) | Binary |
| cp | Chest pain type (1-4) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl (0/1) | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina (0/1) | Binary |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment (1-3) | Categorical |
| ca | Number of major vessels (0-3) | Numerical |
| thal | Thalassemia (3=normal, 6=defect, 7=reversible) | Categorical |
| target | Heart disease (0=No, 1=Yes) | Target |

**Dataset Statistics:**
- Total samples: 303
- After cleaning: 297 samples
- Heart disease positive: 46.1%
- Heart disease negative: 53.9%

## 🛠️ Models Used

| Model | Type | Strengths |
|-------|------|-----------|
| Logistic Regression | Baseline Classification | Interpretable, fast, good for linear relationships |
| Random Forest | Ensemble Learning | Handles non-linear patterns, feature importance |

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions |
| **ROC-AUC** | Model's ability to distinguish between classes |
| **Confusion Matrix** | Breakdown of correct/incorrect predictions |
| **Cross-Validation** | 5-fold CV to ensure model robustness |
| **Precision/Recall** | Per-class performance metrics |

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python task3_heart_disease.py
```

## 📊 Visualizations Generated

| File | Description |
|------|-------------|
| `correlation_heatmap.png` | Correlation between all features |
| `feature_distributions.png` | Age, cholesterol, heart rate distributions by disease status |
| `confusion_matrices.png` | Confusion matrices for both models |
| `roc_curves.png` | ROC curves with AUC scores |
| `feature_importance.png` | Top predictors from Random Forest |

## 🔍 Results Summary

### Model Performance

| Model | Accuracy | ROC-AUC | CV Score |
|-------|----------|---------|----------|
| Logistic Regression | 83.33% | 0.9498 | 0.8221 |
| Random Forest | 83.33% | 0.9442 | 0.8348 |

### Top 5 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Chest pain type (cp) | 16.48% |
| 2 | Thalassemia (thal) | 14.26% |
| 3 | Major vessels count (ca) | 11.96% |
| 4 | Max heart rate (thalach) | 11.53% |
| 5 | ST depression (oldpeak) | 11.01% |

## 💡 Clinical Insights

- **Chest pain type** is the strongest predictor of heart disease
- **Higher maximum heart rate** indicates lower risk
- **ST depression (oldpeak)** during exercise indicates higher risk
- **Exercise-induced angina** significantly increases risk
- **Number of major vessels** colored by fluoroscopy is a strong indicator

## 📊 Sample Output

```
============================================================
MODEL EVALUATION
============================================================

LOGISTIC REGRESSION RESULTS:
   Accuracy: 0.8333 (83.33%)
   ROC-AUC:  0.9498
   CV Score: 0.8221 (+/- 0.0764)

RANDOM FOREST RESULTS:
   Accuracy: 0.8333 (83.33%)
   ROC-AUC:  0.9442
   CV Score: 0.8348 (+/- 0.0737)

============================================================
KEY FINDINGS & INSIGHTS
============================================================

🏥 Top 5 Most Important Features:
   1. cp (chest pain type): 16.48%
   2. thal (thalassemia): 14.26%
   3. ca (major vessels): 11.96%
   4. thalach (max heart rate): 11.53%
   5. oldpeak (ST depression): 11.01%
```

## ✅ Task Completion Checklist

- [x] Load dataset from UCI repository
- [x] Clean dataset (handle missing values)
- [x] Perform Exploratory Data Analysis (EDA)
- [x] Train classification models (Logistic Regression + Random Forest)
- [x] Evaluate with accuracy, ROC-AUC, and confusion matrix
- [x] Analyze feature importance
- [x] Generate and save all visualizations
- [x] Document clinical insights

## 📝 Notes

- Dataset automatically downloaded from UCI Machine Learning Repository
- Missing values in 'ca' and 'thal' columns were removed
- Target variable was binarized (0 = no disease, 1 = disease)
- Features were standardized for Logistic Regression
- Both models achieved identical accuracy (83.33%)

## 🔗 Dataset Source
[UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
## 📚 Technologies Used
- Python 3.11
- Pandas (Data manipulation)
- NumPy (Numerical operations)
- Scikit-learn (Machine learning models)
- Matplotlib & Seaborn (Visualizations)

## 👨‍💻 Author
Muhammad Latif - AI/ML Engineering Intern

## 📅 Date
April 25, 2026