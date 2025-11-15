#!/usr/bin/env python3
"""
Generate comprehensive feature engineering report with statistics
"""

import pandas as pd
import numpy as np

print("="*80)
print("FEATURE ENGINEERING SUMMARY REPORT")
print("="*80)
print()

# Load results
feature_importance = pd.read_csv('/home/dr/cbu/feature_importance_optimized.csv')
model_comparison = pd.read_csv('/home/dr/cbu/model_configurations_comparison.csv')

# Try to load new features correlations (may not exist)
try:
    new_features_corr = pd.read_csv('/home/dr/cbu/new_features_correlations.csv')
except FileNotFoundError:
    new_features_corr = pd.DataFrame()

# Load original statistics
numeric_stats = pd.read_csv('/home/dr/cbu/numeric_features_statistics.csv', index_col=0)

print("FINAL MODEL PERFORMANCE")
print("-"*80)
print(model_comparison.to_string(index=False))
print()

best_config = model_comparison.iloc[0]
print(f"Best Configuration: {best_config['config']}")
print(f"Cross-validation AUC: {best_config['cv_auc']:.4f} ± {best_config['std']:.4f}")
print(f"Training AUC: {best_config['train_auc']:.4f}")
print(f"Overfitting Gap: {best_config['overfit_gap']:.4f}")
print(f"GINI Coefficient: {2*best_config['cv_auc'] - 1:.4f}")
print()

# Improvement analysis
baseline_auc = 0.7889
improvement = ((best_config['cv_auc'] - baseline_auc) / baseline_auc) * 100
print(f"Baseline AUC: {baseline_auc:.4f}")
print(f"Improvement: {improvement:+.2f}%")
print(f"Absolute gain: {best_config['cv_auc'] - baseline_auc:+.4f}")
print()

# Feature categorization
print("="*80)
print("FEATURE ANALYSIS")
print("="*80)
print()

# Categorize features
def categorize_feature(name):
    if '_X_' in name or '_div_' in name:
        return 'Interaction'
    elif any(suffix in name for suffix in ['_squared', '_cubed', '_sqrt', '_log']):
        return 'Polynomial'
    elif '_binned' in name:
        return 'Binned'
    elif '_has_value' in name or '_magnitude' in name:
        return 'Sparse Indicator'
    elif name.startswith('ratio_'):
        return 'Ratio'
    elif any(term in name for term in ['score', 'capacity', 'category', 'stress', 'bucket']):
        return 'Domain-Specific'
    elif any(term in name for term in ['_mean', '_std', '_max', '_sum']):
        return 'Aggregation'
    else:
        return 'Original'

feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)

# Category statistics
category_stats = feature_importance.groupby('category').agg({
    'importance': ['count', 'mean', 'sum', 'max']
}).round(4)
category_stats.columns = ['Count', 'Avg_Importance', 'Total_Importance', 'Max_Importance']
category_stats = category_stats.sort_values('Total_Importance', ascending=False)

print("FEATURE CATEGORIES SUMMARY")
print("-"*80)
print(category_stats.to_string())
print()

# Top features by category
print("TOP 5 FEATURES BY CATEGORY")
print("-"*80)

for category in ['Interaction', 'Polynomial', 'Domain-Specific', 'Original']:
    cat_features = feature_importance[feature_importance['category'] == category].head(5)
    if len(cat_features) > 0:
        print(f"\n{category}:")
        for idx, row in cat_features.iterrows():
            print(f"  {row['feature']:50s} {row['importance']:.4f}")

print()

# Engineered vs Original features
print("="*80)
print("ENGINEERED VS ORIGINAL FEATURES")
print("="*80)
print()

engineered_categories = ['Interaction', 'Polynomial', 'Binned', 'Sparse Indicator', 'Ratio', 'Domain-Specific', 'Aggregation']
original_features = feature_importance[feature_importance['category'] == 'Original']
engineered_features = feature_importance[feature_importance['category'].isin(engineered_categories)]

print(f"Original features: {len(original_features)}")
print(f"Engineered features: {len(engineered_features)}")
print()

print("Importance distribution:")
print(f"  Original total importance: {original_features['importance'].sum():.4f}")
print(f"  Engineered total importance: {engineered_features['importance'].sum():.4f}")
print(f"  Engineered contribution: {engineered_features['importance'].sum() / feature_importance['importance'].sum():.2%}")
print()

# Top features overall
print("="*80)
print("TOP 40 FEATURES BY IMPORTANCE")
print("="*80)
print()

top_features = feature_importance.head(40)
for idx, row in top_features.iterrows():
    marker = "★" if row['category'] != 'Original' else " "
    print(f"{idx+1:2d}. {marker} {row['feature']:55s} {row['importance']:.4f} ({row['category']})")

print()
print("★ = Engineered feature")
print()

# Engineered features in top N
for n in [10, 20, 30, 40]:
    top_n = feature_importance.head(n)
    engineered_count = len(top_n[top_n['category'].isin(engineered_categories)])
    print(f"Engineered features in top {n}: {engineered_count}/{n} ({engineered_count/n:.0%})")

print()

# New feature correlations
print("="*80)
print("NEW FEATURES: TARGET CORRELATIONS")
print("="*80)
print()

if len(new_features_corr) > 0:
    top_corr = new_features_corr.nlargest(20, 'abs_correlation')
    print("Top 20 new features by target correlation:")
    for idx, row in top_corr.iterrows():
        print(f"  {row['feature']:50s} {row['abs_correlation']:.4f}")
    print()

    print(f"New features with |corr| > 0.10: {len(new_features_corr[new_features_corr['abs_correlation'] > 0.10])}")
    print(f"New features with |corr| > 0.05: {len(new_features_corr[new_features_corr['abs_correlation'] > 0.05])}")
    print(f"New features with |corr| > 0.01: {len(new_features_corr[new_features_corr['abs_correlation'] > 0.01])}")
    print()

# Feature importance distribution
print("="*80)
print("FEATURE IMPORTANCE DISTRIBUTION")
print("="*80)
print()

importance_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
for threshold in importance_thresholds:
    count = len(feature_importance[feature_importance['importance'] > threshold])
    pct = count / len(feature_importance) * 100
    print(f"Features with importance > {threshold:.3f}: {count} ({pct:.1f}%)")

print()

# Summary statistics
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print()

summary = {
    'Total features': len(feature_importance),
    'Engineered features': len(engineered_features),
    'Original features': len(original_features),
    'Feature categories': len(feature_importance['category'].unique()),
    'Best CV AUC': best_config['cv_auc'],
    'Baseline AUC': baseline_auc,
    'AUC improvement': best_config['cv_auc'] - baseline_auc,
    'Improvement percentage': improvement,
    'Training AUC': best_config['train_auc'],
    'Overfitting gap': best_config['overfit_gap'],
    'GINI coefficient': 2*best_config['cv_auc'] - 1,
    'Target achieved': 'YES' if best_config['cv_auc'] >= 0.80 else 'NO'
}

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key:.<50s} {value:.4f}")
    else:
        print(f"{key:.<50s} {value}")

print()
print("="*80)
print("REPORT COMPLETE")
print("="*80)

# Save summary to file
summary_df = pd.DataFrame([summary])
summary_df.to_csv('/home/dr/cbu/feature_engineering_summary.csv', index=False)
print("\nSummary saved to: feature_engineering_summary.csv")
