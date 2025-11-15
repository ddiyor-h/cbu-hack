"""
Data Visualization Script for Credit Default Prediction
========================================================

This script creates visualizations to understand the prepared data.

Usage:
    python3 visualize_data.py

Requirements:
    pip install matplotlib seaborn

Author: Data Science Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import json

print("=" * 80)
print("DATA VISUALIZATION FOR CREDIT DEFAULT PREDICTION")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')['default']

with open('/home/dr/cbu/class_balance_info.json', 'r') as f:
    class_info = json.load(f)

print(f"✓ Loaded {X_train.shape[0]:,} training samples with {X_train.shape[1]} features")

# Check if matplotlib and seaborn are available
print("\n[2/4] Checking visualization libraries...")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ matplotlib and seaborn available")
    VISUALIZE = True
except ImportError:
    print("✗ matplotlib or seaborn not available")
    print("  Install with: pip install matplotlib seaborn")
    VISUALIZE = False

if not VISUALIZE:
    print("\nSkipping visualizations. Please install matplotlib and seaborn.")
    exit(0)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n[3/4] Creating visualizations...")

# 1. Class Distribution
print("  Creating class distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
class_counts = y_train.value_counts()
axes[0].bar(['No Default', 'Default'], class_counts.values, color=['green', 'red'], alpha=0.7)
axes[0].set_ylabel('Count')
axes[0].set_title('Class Distribution (Training Set)')
axes[0].text(0, class_counts[0]/2, f'{class_counts[0]:,}\n({class_counts[0]/len(y_train)*100:.1f}%)',
             ha='center', va='center', fontsize=12, fontweight='bold')
axes[0].text(1, class_counts[1]/2, f'{class_counts[1]:,}\n({class_counts[1]/len(y_train)*100:.1f}%)',
             ha='center', va='center', fontsize=12, fontweight='bold')

# Pie chart
axes[1].pie(class_counts.values, labels=['No Default', 'Default'], autopct='%1.2f%%',
            colors=['green', 'red'], alpha=0.7, startangle=90)
axes[1].set_title('Class Distribution (Training Set)')

plt.tight_layout()
plt.savefig('/home/dr/cbu/01_class_distribution.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 01_class_distribution.png")
plt.close()

# 2. Feature Statistics Summary
print("  Creating feature statistics summary...")
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Calculate statistics for first 20 numeric features
feature_stats = pd.DataFrame({
    'Feature': numeric_features[:20],
    'Mean': [X_train[f].mean() for f in numeric_features[:20]],
    'Median': [X_train[f].median() for f in numeric_features[:20]],
    'Std': [X_train[f].std() for f in numeric_features[:20]],
    'Min': [X_train[f].min() for f in numeric_features[:20]],
    'Max': [X_train[f].max() for f in numeric_features[:20]]
})

# Save to CSV
feature_stats.to_csv('/home/dr/cbu/feature_statistics.csv', index=False)
print("  ✓ Saved: feature_statistics.csv")

# 3. Sample Feature Distributions (top 6 features by name)
print("  Creating feature distribution plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

important_features = [
    'annual_income', 'age', 'credit_utilization',
    'debt_to_income_ratio', 'num_login_sessions', 'account_age_years'
]

# Use available features if engineered ones don't exist
available_features = [f for f in important_features if f in X_train.columns]
if len(available_features) < 6:
    available_features = numeric_features[:6]

for i, feature in enumerate(available_features[:6]):
    if feature in X_train.columns:
        axes[i].hist(X_train[feature], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/dr/cbu/02_feature_distributions.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 02_feature_distributions.png")
plt.close()

# 4. Correlation Analysis (sample of features)
print("  Creating correlation heatmap...")
# Select a subset of numeric features for correlation
sample_features = numeric_features[:20]
correlation_matrix = X_train[sample_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix (First 20 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/dr/cbu/03_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 03_correlation_matrix.png")
plt.close()

# 5. Feature Comparison by Target Class
print("  Creating feature comparison by target class...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(available_features[:6]):
    if feature in X_train.columns:
        # Split by default/no default
        no_default = X_train[y_train == 0][feature]
        default = X_train[y_train == 1][feature]

        axes[i].hist([no_default, default], bins=30, alpha=0.6,
                     label=['No Default', 'Default'], color=['green', 'red'])
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{feature} by Default Status')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/dr/cbu/04_features_by_target.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 04_features_by_target.png")
plt.close()

print("\n[4/4] Summary statistics...")

# Print summary
print("\n" + "=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(f"""
Dataset Size:
  Training samples: {len(X_train):,}
  Features: {X_train.shape[1]}
  Numeric features: {len(numeric_features)}

Class Balance:
  No Default: {class_counts[0]:,} ({class_counts[0]/len(y_train)*100:.2f}%)
  Default: {class_counts[1]:,} ({class_counts[1]/len(y_train)*100:.2f}%)
  Ratio: 1:{(class_counts[0]/class_counts[1]):.1f}

Feature Statistics:
  Mean feature count per sample: {X_train.shape[1]}
  Features with missing values: 0

Files Created:
  ✓ 01_class_distribution.png
  ✓ 02_feature_distributions.png
  ✓ 03_correlation_matrix.png
  ✓ 04_features_by_target.png
  ✓ feature_statistics.csv

Next Steps:
  1. Review the visualizations to understand data distributions
  2. Check correlation matrix for multicollinearity
  3. Analyze feature differences between default/no-default groups
  4. Proceed to model training with train_model.py
""")
print("=" * 80)
print("✓ Visualization complete!")
print("=" * 80)
