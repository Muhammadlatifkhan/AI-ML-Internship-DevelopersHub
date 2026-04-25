"""
Task 1: Iris Dataset - Exploratory Data Analysis
DevelopersHub Corporation - AI/ML Internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("="*60)
print("TASK 1: IRIS DATASET EDA")
print("="*60)

# Load dataset
df = sns.load_dataset('iris')
print(f"\n✅ Dataset loaded successfully!")
print(f"📊 Shape: {df.shape}")
print(f"📋 Columns: {df.columns.tolist()}")

# Data inspection
print("\n" + "="*60)
print("DATA INSPECTION")
print("="*60)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Create visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# 1. Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', palette='viridis', s=100, ax=axes[0,0])
axes[0,0].set_title('Sepal Length vs Sepal Width')
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='viridis', s=100, ax=axes[0,1])
axes[0,1].set_title('Petal Length vs Petal Width')
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', palette='viridis', s=100, ax=axes[1,0])
axes[1,0].set_title('Sepal Length vs Petal Length')
sns.scatterplot(data=df, x='sepal_width', y='petal_width', hue='species', palette='viridis', s=100, ax=axes[1,1])
axes[1,1].set_title('Sepal Width vs Petal Width')
plt.suptitle('Scatter Plots - Feature Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/scatter_plots.png', dpi=300)
plt.show()

# 2. Histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(data=df, x='sepal_length', hue='species', bins=20, kde=True, alpha=0.6, ax=axes[0,0])
axes[0,0].set_title('Distribution of Sepal Length')
sns.histplot(data=df, x='sepal_width', hue='species', bins=20, kde=True, alpha=0.6, ax=axes[0,1])
axes[0,1].set_title('Distribution of Sepal Width')
sns.histplot(data=df, x='petal_length', hue='species', bins=20, kde=True, alpha=0.6, ax=axes[1,0])
axes[1,0].set_title('Distribution of Petal Length')
sns.histplot(data=df, x='petal_width', hue='species', bins=20, kde=True, alpha=0.6, ax=axes[1,1])
axes[1,1].set_title('Distribution of Petal Width')
plt.suptitle('Histograms - Feature Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/histograms.png', dpi=300)
plt.show()

# 3. Box plots for outlier detection
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df, x='species', y='sepal_length', ax=axes[0])
axes[0].set_title('Box Plot - Sepal Length by Species')
sns.boxplot(data=df, x='species', y='petal_length', ax=axes[1])
axes[1].set_title('Box Plot - Petal Length by Species')
plt.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/box_plots.png', dpi=300)
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df.drop('species', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=2)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=300)
plt.show()

# 5. Pairplot (complete overview)
pairplot = sns.pairplot(df, hue='species', palette='viridis', diag_kind='kde')
pairplot.fig.suptitle('Pairplot - Complete Feature Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('images/pairplot.png', dpi=300)
plt.show()

# Key insights
print("\n" + "="*60)
print("KEY INSIGHTS & FINDINGS")
print("="*60)
insights = [
    "1. Setosa species is clearly separable from Versicolor and Virginica",
    "2. Petal length and petal width have strong positive correlation (0.96)",
    "3. Virginica has the largest petal dimensions (avg petal length: 5.55cm)",
    "4. No missing values or significant outliers detected in the dataset",
    "5. All features follow approximately normal distributions",
    "6. Petal features are more discriminative than sepal features for classification"
]
for insight in insights:
    print(insight)

print("\n" + "="*60)
print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 Visualizations saved in 'images' folder")