"""
Approach #3: CatBoost with Ordered Boosting (Quick Test)

Expected AUC: 0.81-0.83
Implementation time: 30 minutes - 1 hour
Target improvement: +0.010-0.025 AUC

This is a quick test script to evaluate CatBoost's performance.
CatBoost advantages:
- Ordered boosting (prevents target leakage)
- Symmetric trees (better generalization)
- Auto class weights (handles imbalance)
- Fast inference

Use this as a quick baseline before trying more complex approaches.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5

# Paths
DATA_DIR = '/home/dr/cbu'
X_TRAIN_PATH = f'{DATA_DIR}/X_train_engineered.parquet'
Y_TRAIN_PATH = f'{DATA_DIR}/y_train.parquet'
X_TEST_PATH = f'{DATA_DIR}/X_test_engineered.parquet'

print("="*80)
print("APPROACH #3: CatBoost Quick Test")
print("="*80)
print()

# Load data
print("[1/3] Loading data...")
X_train = pd.read_parquet(X_TRAIN_PATH)
y_train = pd.read_parquet(Y_TRAIN_PATH).values.ravel()
X_test = pd.read_parquet(X_TEST_PATH)

print(f"  Training set: {X_train.shape}")
print(f"  Class distribution: {np.sum(y_train==0)} / {np.sum(y_train==1)} (ratio: {np.sum(y_train==0)/np.sum(y_train==1):.1f}:1)")
print()

# Configure CatBoost
print("[2/3] Training CatBoost with cross-validation...")
print()

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 5,
    'auto_class_weights': 'Balanced',  # Key parameter for imbalanced data
    'eval_metric': 'AUC',
    'random_seed': RANDOM_STATE,
    'verbose': 100,
    'early_stopping_rounds': 50,
    'thread_count': -1
}

print("  Parameters:")
for param, value in catboost_params.items():
    if param not in ['verbose', 'thread_count']:
        print(f"    {param}: {value}")
print()

# Cross-validation
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"  Fold {fold+1}/{N_FOLDS}:")
    start_time = time.time()

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Create pools
    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)

    # Train model
    model = CatBoostClassifier(**catboost_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=False
    )

    # Evaluate
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    cv_scores.append(auc)
    models.append(model)

    print(f"    AUC: {auc:.4f}, Best iteration: {model.best_iteration_}, Time: {time.time()-start_time:.1f}s")

print()
print(f"  Mean CV AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"  GINI: {2*np.mean(cv_scores)-1:.4f}")
print()

# Generate predictions
print("[3/3] Generating test predictions...")

# Train final model on full dataset
final_model = CatBoostClassifier(**catboost_params)
final_model.fit(X_train, y_train, verbose=100)

y_test_pred = final_model.predict_proba(X_test)[:, 1]

# Save
predictions_df = pd.DataFrame({'prediction': y_test_pred})
predictions_df.to_csv(f'{DATA_DIR}/predictions_catboost.csv', index=False)
final_model.save_model(f'{DATA_DIR}/model_catboost.cbm')

print()
print("="*80)
print("RESULTS")
print("="*80)
print()
print(f"Cross-validation AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"GINI coefficient: {2*np.mean(cv_scores)-1:.4f}")
print()
print("Comparison with baseline:")
print(f"  Baseline XGBoost: AUC = 0.8047, GINI = 0.6094")
print(f"  CatBoost:         AUC = {np.mean(cv_scores):.4f}, GINI = {2*np.mean(cv_scores)-1:.4f}")
print(f"  Improvement:      {np.mean(cv_scores)-0.8047:+.4f} AUC ({(np.mean(cv_scores)-0.8047)*100:+.2f}%)")
print()
print("Files saved:")
print(f"  - {DATA_DIR}/predictions_catboost.csv")
print(f"  - {DATA_DIR}/model_catboost.cbm")
print()
print("Next steps:")
if np.mean(cv_scores) >= 0.82:
    print("  GREAT! CatBoost works well. You can stop here or try:")
else:
    print("  CatBoost didn't improve much. Try:")
print("  1. Approach #1 (LightGBM + ADASYN + Optuna)")
print("  2. Approach #2 (Ensemble Stacking)")
print()
print("="*80)
