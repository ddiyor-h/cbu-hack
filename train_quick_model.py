#!/usr/bin/env python3
"""
QUICK MODEL TRAINING
Быстрое обучение XGBoost модели для inference
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUICK XGBoost MODEL TRAINING")
print("="*80)

# Load data
print("\n[1/4] Loading leak-free data...")
X_train = pd.read_parquet('task_result1/X_train_leak_free_v3.parquet')
X_test = pd.read_parquet('task_result1/X_test_leak_free_v3.parquet')
y_train = pd.read_parquet('task_result1/y_train_leak_free_v3.parquet').values.ravel()
y_test = pd.read_parquet('task_result1/y_test_leak_free_v3.parquet').values.ravel()

print(f"  Training: {X_train.shape}")
print(f"  Test: {X_test.shape}")
print(f"  Default rate: {y_train.mean():.2%}")

# Calculate class weight
scale_pos_weight = (1 - y_train.mean()) / y_train.mean()
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# Train model with good parameters
print("\n[2/4] Training XGBoost model...")
print("  Using optimized hyperparameters...")

model = XGBClassifier(
    max_depth=5,
    learning_rate=0.03,
    n_estimators=1000,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=3.0,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("✓ Model trained")

# Evaluate
print("\n[3/4] Evaluating model...")
train_pred = model.predict_proba(X_train)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
test_auc = roc_auc_score(y_test, test_pred)
gap = train_auc - test_auc

print(f"  Train AUC: {train_auc:.4f}")
print(f"  Test AUC:  {test_auc:.4f}")
print(f"  Gap:       {gap:.4f}")

# Save model
print("\n[4/4] Saving model...")
model_path = 'task_result1/model/xgboost_model_v3.pkl'
joblib.dump(model, model_path)
print(f"✓ Model saved to: {model_path}")

# Save feature names
feature_names_path = 'task_result1/model/feature_names_v3.txt'
with open(feature_names_path, 'w') as f:
    for feat in X_train.columns:
        f.write(f"{feat}\n")
print(f"✓ Feature names saved to: {feature_names_path}")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)
print(f"\nModel ready for inference!")
print(f"  Features: {len(X_train.columns)}")
print(f"  AUC: {test_auc:.4f}")
