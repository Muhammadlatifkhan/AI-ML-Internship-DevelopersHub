## 🎯 Objective
Predict next day's closing price using historical stock data from Yahoo Finance with regression models.

## 📊 Dataset
**Apple Inc. (AAPL)** - 2 years of daily stock data (501 trading days)

| Feature | Description |
|---------|-------------|
| Open | Opening price |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Close | Closing price |
| Volume | Number of shares traded |

**Engineered Features:**
- Daily returns, volume changes, high/low ratio
- 5, 10, 20-day moving averages
- Price-to-moving-average ratios

## 🛠️ Models Used

| Model | Type | Strengths |
|-------|------|-----------|
| Linear Regression | Baseline | Simple, interpretable, fast |
| Random Forest | Ensemble | Handles non-linear patterns |

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (average prediction error in dollars) |
| **RMSE** | Root Mean Square Error (penalizes large errors) |
| **R² Score** | Variance explained by the model |

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python task2_stock_prediction.py
```

## 📊 Visualizations Generated

| File | Description |
|------|-------------|
| `actual_vs_predicted.png` | Actual vs predicted prices over time |
| `scatter_plot.png` | Predicted vs actual price scatter |
| `residuals.png` | Prediction errors over time |
| `feature_importance.png` | Most important features |
| `model_comparison.png` | MAE/RMSE comparison |

## 🔍 Results Summary

### Model Performance (Apple Stock)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | $3.23 | $4.23 | 0.7920 |
| Random Forest | $5.99 | $7.08 | 0.4174 |

### Top 5 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Low price | 67.80% |
| 2 | High price | 19.02% |
| 3 | 20-day MA | 5.02% |
| 4 | Open price | 4.77% |
| 5 | 5-day MA | 0.79% |

## 💡 Trading Insights

- **Low price** is the strongest predictor of next day's close
- **Linear Regression outperformed Random Forest** (R²: 0.792 vs 0.417)
- Average prediction error is only **$3.23** per share
- Moving averages help identify price trends

## 📊 Sample Output

```
============================================================
MODEL EVALUATION
============================================================

LINEAR REGRESSION RESULTS:
   MAE:  $3.23
   RMSE: $4.23
   R²:   0.7920

RANDOM FOREST RESULTS:
   MAE:  $5.99
   RMSE: $7.08
   R²:   0.4174

============================================================
KEY FINDINGS & INSIGHTS
============================================================

🏆 Best Model: Linear Regression
   R² Score: 0.7920
   Prediction Error: $3.23
```

## ✅ Task Completion Checklist

- [x] Load historical stock data using yfinance
- [x] Create features (Open, High, Low, Volume, moving averages)
- [x] Train regression models (Linear Regression + Random Forest)
- [x] Plot actual vs predicted prices
- [x] Evaluate with MAE and RMSE
- [x] Analyze feature importance
- [x] Save all visualizations

## 📝 Notes

- Data period: April 2024 - April 2026 (2 years)
- Linear Regression performed better on this linear trend data
- Features were engineered including moving averages and ratios
- Time series split maintained chronological order

## 🔗 Data Source
Yahoo Finance (yfinance library)

## 📚 Technologies Used
- Python 3.11
- yfinance (Stock data)
- Pandas & NumPy (Data manipulation)
- Scikit-learn (Machine learning)
- Matplotlib & Seaborn (Visualizations)


## 👨‍💻 Author
Muhammad Latif - AI/ML Engineering Intern

## 📅 Date
April 25, 2026

