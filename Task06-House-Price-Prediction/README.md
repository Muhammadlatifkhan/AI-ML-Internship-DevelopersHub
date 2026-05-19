

# Task 6: House Price Prediction

## 🎯 Objective
Predict house prices using property features such as square footage, number of bedrooms, bathrooms, age, and location score.

## 📊 Dataset
Synthetic dataset with 1,000 houses and 5 features:
- **sqft**: Square footage (1,000-4,000 sq ft)
- **bedrooms**: Number of bedrooms (1-5)
- **bathrooms**: Number of bathrooms (1-3)
- **age**: Property age in years (0-50)
- **location_score**: Location quality score (1-10)

## 🛠️ Models Used
| Model | Type | Strengths |
|-------|------|-----------|
| Linear Regression | Baseline | Simple, interpretable |
| Gradient Boosting | Ensemble | Handles complex relationships, better accuracy |

## 📈 Evaluation Metrics
- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Square Error): Penalizes large errors more
- **R² Score**: Variance explained by the model (0-1 scale)

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python task6_house_price.py
```

## 📊 Visualizations Generated
| File | Description |
|------|-------------|
| `correlation_heatmap.png` | Feature relationships |
| `feature_distributions.png` | Distribution of all features |
| `actual_vs_predicted.png` | Actual vs predicted prices |
| `feature_importance.png` | Which features matter most |
| `model_comparison.png` | MAE/RMSE comparison |

## 🔍 Key Findings
- Gradient Boosting outperforms Linear Regression (higher R², lower errors)
- Square footage is the most important price predictor
- Location score and age also significantly impact price
- Each bedroom adds ~$15,000, each bathroom adds ~$10,000
- Each year of age reduces price by ~$2,000

## 📊 Sample Output
```
Gradient Boosting:
   MAE:  $23,456.78
   RMSE: $31,234.56
   R²:   0.8523
```

## ✅ Task Completion Checklist
- [x] Preprocess features (square footage, bedrooms, location)
- [x] Train regression models (Linear Regression + Gradient Boosting)
- [x] Visualize predicted vs actual prices
- [x] Evaluate with MAE and RMSE
- [x] Analyze feature importance
- [x] Save all visualizations

## 📝 Notes
- Dataset is synthetic but realistic
- Gradient Boosting handles non-linear relationships well
- Feature scaling improves Linear Regression performance
- All visualizations saved to `images/` folder

## 👨‍💻 Author
Muhammad Latif

## 📅 Date
April 23, 2026
