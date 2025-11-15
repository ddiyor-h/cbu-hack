#!/usr/bin/env python3
"""
Optimized Model Training with Feature Selection and Regularization
Goal: Achieve AUC > 0.80
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("OPTIMIZED MODEL TRAINING WITH FEATURE SELECTION")
print("="*80)
print()

# Load engineered features
print("Loading engineered features...")
X_train = pd.read_parquet('/home/dr/cbu/X_train_engineered.parquet')
X_test = pd.read_parquet('/home/dr/cbu/X_test_engineered.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet').values.ravel()

print(f"Initial shape - Train: {X_train.shape}, Test: {X_test.shape}")
print()

# Load feature importance from previous run
feature_importance = pd.read_csv('/home/dr/cbu/feature_importance_engineered.csv')

# Feature selection strategy 1: Remove low importance features
importance_threshold = 0.0015  # Keep features with importance > 0.15%
selected_features_by_importance = feature_importance[
    feature_importance['importance'] > importance_threshold
]['feature'].tolist()

print(f"Features selected by importance (>{importance_threshold}): {len(selected_features_by_importance)}")

# Feature selection strategy 2: Keep highly correlated features with target
try:
    new_features_corr = pd.read_csv('/home/dr/cbu/new_features_correlations.csv')
    high_corr_features = new_features_corr[
        new_features_corr['abs_correlation'] > 0.01
    ]['feature'].tolist()
    print(f"New features with |corr| > 0.01: {len(high_corr_features)}")
except:
    high_corr_features = []

# Combine selection strategies
selected_features = list(set(selected_features_by_importance) | set(high_corr_features))
selected_features = [f for f in selected_features if f in X_train.columns]

print(f"Total features after selection: {len(selected_features)}")
print(f"Features removed: {X_train.shape[1] - len(selected_features)}")
print()

# Apply feature selection
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"Selected shape - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")
print()

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Model configurations to test
configs = [
    {
        'name': 'Conservative (High Regularization)',
        'params': {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'gamma': 0.5,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
    },
    {
        'name': 'Moderate (Balanced)',
        'params': {
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.04,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 4,
            'gamma': 0.3,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'scale_pos_weight': scale_pos_weight,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
    },
    {
        'name': 'Aggressive (Lower Regularization)',
        'params': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.2,
            'reg_alpha': 0.3,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
    }
]

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Test each configuration
results = []
best_auc = 0
best_config = None
best_model = None

for config in configs:
    print(f"Testing configuration: {config['name']}")
    print("-" * 60)

    model = XGBClassifier(**config['params'])

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_selected, y_train,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )

    mean_cv_auc = cv_scores.mean()
    std_cv_auc = cv_scores.std()

    print(f"CV AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")

    # Train on full set to check overfitting
    model.fit(X_train_selected, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_selected)[:, 1])
    overfit_gap = train_auc - mean_cv_auc

    print(f"Train AUC: {train_auc:.4f}")
    print(f"Overfitting gap: {overfit_gap:.4f}")
    print()

    results.append({
        'config': config['name'],
        'cv_auc': mean_cv_auc,
        'std': std_cv_auc,
        'train_auc': train_auc,
        'overfit_gap': overfit_gap
    })

    if mean_cv_auc > best_auc:
        best_auc = mean_cv_auc
        best_config = config
        best_model = model

# Results summary
print("="*80)
print("CONFIGURATION COMPARISON")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# Best model details
print("="*80)
print("BEST MODEL")
print("="*80)
print(f"Configuration: {best_config['name']}")
print(f"CV AUC: {best_auc:.4f}")
print()

# Feature importance from best model
feature_importance_selected = pd.DataFrame({
    'feature': selected_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 30 features:")
for i, row in feature_importance_selected.head(30).iterrows():
    print(f"{row['feature']:60s} {row['importance']:.4f}")

# Save results
feature_importance_selected.to_csv('/home/dr/cbu/feature_importance_optimized.csv', index=False)
results_df.to_csv('/home/dr/cbu/model_configurations_comparison.csv', index=False)

# Save selected features list
with open('/home/dr/cbu/selected_features_optimized.txt', 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

# Save test predictions from best model
test_predictions = pd.DataFrame({
    'default_probability': best_model.predict_proba(X_test_selected)[:, 1]
})
test_predictions.to_csv('/home/dr/cbu/test_predictions_optimized.csv', index=False)

print()
print("="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Best CV AUC: {best_auc:.4f}")
print(f"Features used: {len(selected_features)}")
print(f"Baseline AUC: 0.7889")
print(f"Improvement: {((best_auc - 0.7889) / 0.7889) * 100:+.2f}%")
print()

if best_auc >= 0.80:
    print("SUCCESS: Achieved AUC >= 0.80!")
else:
    gap = 0.80 - best_auc
    print(f"Gap to target (0.80): {gap:.4f}")
    print("\nRecommendations:")
    print("1. Try ensemble methods (XGBoost + LightGBM + RF)")
    print("2. Apply SMOTE for class balancing")
    print("3. Optimize prediction threshold")
    print("4. Try stacking classifier")

print("="*80)
print("\nFiles saved:")
print("- feature_importance_optimized.csv")
print("- model_configurations_comparison.csv")
print("- selected_features_optimized.txt")
print("- test_predictions_optimized.csv")
print("="*80)
