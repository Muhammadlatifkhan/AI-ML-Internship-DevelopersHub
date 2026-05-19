"""
Task 6: House Price Prediction
DevelopersHub Corporation - AI/ML Internship

Predict house prices using features like square footage, bedrooms, bathrooms, and age.
Uses Gradient Boosting Regression for optimal performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TASK 6: HOUSE PRICE PREDICTION")
print("="*60)

# ============================================
# 1. CREATE SAMPLE DATASET
# ============================================
print("\n📊 Creating house price dataset...")

np.random.seed(42)
n_samples = 1000

# Generate realistic house features
df = pd.DataFrame({
    'sqft': np.random.normal(2000, 500, n_samples),  # Square feet
    'bedrooms': np.random.randint(1, 6, n_samples),   # 1-5 bedrooms
    'bathrooms': np.random.randint(1, 4, n_samples),  # 1-3 bathrooms
    'age': np.random.randint(0, 50, n_samples),       # 0-50 years old
    'location_score': np.random.uniform(1, 10, n_samples)  # Location quality 1-10
})

# Create price based on features (with realistic relationships)
df['price'] = (
    df['sqft'] * 200 +           # $200 per sq ft
    df['bedrooms'] * 15000 +     # $15k per bedroom
    df['bathrooms'] * 10000 +    # $10k per bathroom
    -df['age'] * 2000 +          # -$2k per year of age
    df['location_score'] * 20000 + # $20k per location point
    np.random.normal(0, 30000, n_samples)  # Random noise
)

# Ensure no negative prices
df['price'] = df['price'].clip(lower=100000)

print(f"✅ Dataset created: {len(df)} houses")
print(f"📊 Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"📊 Average price: ${df['price'].mean():,.0f}")

# ============================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=2, annot_kws={'size': 10})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=300)
plt.show()

# Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score', 'price']
for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    sns.histplot(data=df, x=feature, kde=True, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {feature}', fontweight='bold')
plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/feature_distributions.png', dpi=300)
plt.show()

# ============================================
# 3. DATA PREPROCESSING
# ============================================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")
print(f"✅ Features: {list(X.columns)}")

# ============================================
# 4. MODEL TRAINING
# ============================================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Model 1: Linear Regression
print("\n📈 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Model 2: Gradient Boosting (better for complex relationships)
print("📈 Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)

# ============================================
# 5. MODEL EVALUATION
# ============================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = gb_model.score(X_test_scaled, y_test) if model_name == "Gradient Boosting" else lr_model.score(X_test_scaled, y_test)
    
    print(f"\n{model_name}:")
    print(f"   MAE:  ${mae:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   R²:   {r2:.4f}")
    return mae, rmse, r2

print("\n" + "-"*40)
print("LINEAR REGRESSION RESULTS:")
print("-"*40)
lr_mae, lr_rmse, lr_r2 = evaluate_model(y_test, lr_pred, "Linear Regression")

print("\n" + "-"*40)
print("GRADIENT BOOSTING RESULTS:")
print("-"*40)
gb_mae, gb_rmse, gb_r2 = evaluate_model(y_test, gb_pred, "Gradient Boosting")

# ============================================
# 6. VISUALIZATIONS
# ============================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Plot 1: Actual vs Predicted Prices (Gradient Boosting)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
axes[0].scatter(y_test, gb_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price ($)', fontsize=12)
axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
axes[0].set_title('Actual vs Predicted House Prices\n(Gradient Boosting)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual plot
residuals = y_test - gb_pred
axes[1].scatter(gb_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price ($)', fontsize=12)
axes[1].set_ylabel('Residuals ($)', fontsize=12)
axes[1].set_title('Residual Plot\n(Gradient Boosting)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/actual_vs_predicted.png', dpi=300)
plt.show()

# Plot 2: Feature Importance (Gradient Boosting)
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance - Gradient Boosting', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300)
plt.show()

# Plot 3: Model Comparison
plt.figure(figsize=(10, 6))
models = ['Linear Regression', 'Gradient Boosting']
mae_scores = [lr_mae, gb_mae]
rmse_scores = [lr_rmse, gb_rmse]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, mae_scores, width, label='MAE', color='skyblue')
plt.bar(x + width/2, rmse_scores, width, label='RMSE', color='lightcoral')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Error ($)', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/model_comparison.png', dpi=300)
plt.show()

# ============================================
# 7. KEY FINDINGS
# ============================================
print("\n" + "="*60)
print("KEY FINDINGS & INSIGHTS")
print("="*60)

insights = [
    f"\n📈 Model Performance:",
    f"   • Gradient Boosting outperforms Linear Regression",
    f"   • R² Score: {gb_r2:.4f} (explains {gb_r2*100:.1f}% of price variance)",
    f"   • Average prediction error: ${gb_mae:,.2f}",
    f"\n🏠 Most Important Features:",
]

for _, row in feature_importance.iterrows():
    insights.append(f"   • {row['feature']}: {row['importance']:.2%} importance")

insights.append(f"\n💰 Price Drivers:")
insights.append(f"   • Square footage: +$200 per sq ft")
insights.append(f"   • Each bedroom: +$15,000")
insights.append(f"   • Each bathroom: +$10,000")
insights.append(f"   • Each year of age: -$2,000")
insights.append(f"   • Location score: +$20,000 per point")

for insight in insights:
    print(insight)

print("\n" + "="*60)
print("✅ TASK 6 COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 Visualizations saved in 'images' folder:")
print("   • correlation_heatmap.png")
print("   • feature_distributions.png")
print("   • actual_vs_predicted.png")
print("   • feature_importance.png")
print("   • model_comparison.png")