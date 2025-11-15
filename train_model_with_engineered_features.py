#!/usr/bin/env python3
"""
Model Training with Engineered Features
Goal: Achieve AUC > 0.80 using advanced feature engineering
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("MODEL TRAINING WITH ENGINEERED FEATURES")
print("="*80)
print()

# Load engineered features
print("Loading engineered features...")
X_train = pd.read_parquet('/home/dr/cbu/X_train_engineered.parquet')
X_test = pd.read_parquet('/home/dr/cbu/X_test_engineered.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet').values.ravel()

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Default rate: {y_train.mean():.2%}")
print(f"Class imbalance: 1:{int(1/y_train.mean())}")
print()

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")
print()

# XGBoost model with optimized parameters
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    eval_metric='auc'
)

# Cross-validation
print("Performing stratified 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"CV AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print()

# Train final model
print("Training final model on full training set...")
model.fit(X_train, y_train)

# Training set evaluation
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_train_pred_proba)
print(f"Training AUC: {train_auc:.4f}")
print(f"Training GINI: {2*train_auc - 1:.4f}")
print()

# Feature importance
print("Top 20 features by importance:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(20).iterrows():
    print(f"{row['feature']:60s} {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv('/home/dr/cbu/feature_importance_engineered.csv', index=False)
print()
print("Feature importance saved to feature_importance_engineered.csv")
print()

# Save model predictions
print("Saving predictions...")
test_predictions = pd.DataFrame({
    'default_probability': model.predict_proba(X_test)[:, 1]
})
test_predictions.to_csv('/home/dr/cbu/test_predictions_engineered.csv', index=False)
print("Test predictions saved to test_predictions_engineered.csv")
print()

# Model summary
print("="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"Features used: {X_train.shape[1]}")
print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Training AUC: {train_auc:.4f}")
print(f"Training GINI: {2*train_auc - 1:.4f}")
print()

improvement = ((cv_scores.mean() - 0.7889) / 0.7889) * 100
print(f"Improvement over baseline (0.7889): {improvement:+.2f}%")
print()

if cv_scores.mean() > 0.80:
    print("SUCCESS: Achieved AUC > 0.80!")
else:
    print(f"Current AUC: {cv_scores.mean():.4f}, Target: 0.80")
    print(f"Gap to target: {0.80 - cv_scores.mean():.4f}")

print("="*80)
