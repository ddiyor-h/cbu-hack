"""
Approach #1: LightGBM + ADASYN + Optuna Hyperparameter Tuning

Expected AUC: 0.825-0.840
Implementation time: 3-4 hours
Target improvement: +0.020-0.035 AUC

This script implements the state-of-the-art approach for imbalanced credit default prediction:
1. ADASYN over-sampling with optimal 6.6:1 ratio (not 1:1!)
2. LightGBM with is_unbalance parameter
3. Optuna Bayesian optimization for hyperparameters
4. 5-fold stratified cross-validation
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import ADASYN
import warnings
import time
import json

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
N_OPTUNA_TRIALS = 100  # Reduce to 50 for faster testing
OPTUNA_TIMEOUT = 10800  # 3 hours in seconds

# Paths
DATA_DIR = '/home/dr/cbu'
X_TRAIN_PATH = f'{DATA_DIR}/X_train_engineered.parquet'
Y_TRAIN_PATH = f'{DATA_DIR}/y_train.parquet'
X_TEST_PATH = f'{DATA_DIR}/X_test_engineered.parquet'

print("="*80)
print("APPROACH #1: LightGBM + ADASYN + Optuna")
print("="*80)
print()

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("[Step 1/5] Loading data...")
start_time = time.time()

X_train = pd.read_parquet(X_TRAIN_PATH)
y_train = pd.read_parquet(Y_TRAIN_PATH).values.ravel()
X_test = pd.read_parquet(X_TEST_PATH)

print(f"  Training set: {X_train.shape}")
print(f"  Test set: {X_test.shape}")
print(f"  Features: {X_train.shape[1]}")

# Check class distribution
unique, counts = np.unique(y_train, dtype=int)
class_ratio = counts[0] / counts[1]
minority_pct = (counts[1] / len(y_train)) * 100

print(f"  Class distribution:")
print(f"    Class 0 (non-default): {counts[0]:,} ({100-minority_pct:.2f}%)")
print(f"    Class 1 (default): {counts[1]:,} ({minority_pct:.2f}%)")
print(f"    Imbalance ratio: {class_ratio:.1f}:1")
print(f"  Time: {time.time() - start_time:.1f}s")
print()

# ============================================================================
# STEP 2: Apply ADASYN with Optimal Ratio
# ============================================================================
print("[Step 2/5] Applying ADASYN over-sampling...")
print("  Research finding: Optimal ratio is 6.6:1, NOT 1:1!")
print("  Source: 'Finding the Sweet Spot' (arXiv:2510.18252, 2024)")
start_time = time.time()

# Calculate target ratio
# Current: 5.11% minority (1:18.6)
# Target: 13.2% minority (1:6.6)
# sampling_strategy = minority_after / majority = 0.132 / (1 - 0.132) = 0.152
TARGET_MINORITY_PCT = 13.2
target_sampling_ratio = (TARGET_MINORITY_PCT / 100) / (1 - TARGET_MINORITY_PCT / 100)

print(f"  Current minority: {minority_pct:.2f}%")
print(f"  Target minority: {TARGET_MINORITY_PCT:.2f}% (6.6:1 ratio)")
print(f"  ADASYN sampling_strategy: {target_sampling_ratio:.3f}")

adasyn = ADASYN(
    sampling_strategy=target_sampling_ratio,
    random_state=RANDOM_STATE,
    n_neighbors=5,
    n_jobs=-1
)

X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Check new distribution
unique_res, counts_res = np.unique(y_train_resampled, dtype=int)
minority_pct_res = (counts_res[1] / len(y_train_resampled)) * 100
ratio_res = counts_res[0] / counts_res[1]

print(f"  Resampled data:")
print(f"    Total samples: {len(y_train):,} -> {len(y_train_resampled):,} (+{len(y_train_resampled)-len(y_train):,})")
print(f"    Class 0: {counts_res[0]:,}")
print(f"    Class 1: {counts_res[1]:,} (+{counts_res[1]-counts[1]:,} synthetic)")
print(f"    New minority: {minority_pct_res:.2f}%")
print(f"    New ratio: {ratio_res:.1f}:1")
print(f"  Time: {time.time() - start_time:.1f}s")
print()

# ============================================================================
# STEP 3: Define Optuna Objective Function
# ============================================================================
print("[Step 3/5] Setting up Optuna hyperparameter optimization...")

def objective(trial):
    """Optuna objective function for LightGBM hyperparameter tuning"""

    # Suggest hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,

        # Key hyperparameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

        # Class imbalance handling
        'is_unbalance': True,  # Alternative: use scale_pos_weight
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_resampled, y_train_resampled)):
        X_tr, X_val = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
        y_tr, y_val = y_train_resampled[train_idx], y_train_resampled[val_idx]

        # Create datasets
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                LightGBMPruningCallback(trial, 'auc')  # Optuna pruning
            ]
        )

        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred)
        cv_scores.append(auc)

    return np.mean(cv_scores)

print(f"  Optimization settings:")
print(f"    Algorithm: TPE (Tree-structured Parzen Estimator)")
print(f"    Trials: {N_OPTUNA_TRIALS}")
print(f"    Timeout: {OPTUNA_TIMEOUT/3600:.1f} hours")
print(f"    Cross-validation: {N_FOLDS}-fold StratifiedKFold")
print(f"    Pruning: MedianPruner (early stopping for bad trials)")
print()

# ============================================================================
# STEP 4: Run Optuna Optimization
# ============================================================================
print("[Step 4/5] Running Optuna optimization...")
print("  This will take ~2-3 hours. Progress will be shown every 10 trials.")
print("  You can stop early with Ctrl+C - best parameters will be saved.")
print()
start_time = time.time()

# Create study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

# Optimize
try:
    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True,
        callbacks=[lambda study, trial: print(f"  Trial {trial.number}: AUC = {trial.value:.4f}") if trial.number % 10 == 0 else None]
    )
except KeyboardInterrupt:
    print("\n  Optimization interrupted by user. Using best parameters found so far.")

optimization_time = time.time() - start_time

print()
print("  Optimization complete!")
print(f"  Total time: {optimization_time/3600:.2f} hours")
print(f"  Trials completed: {len(study.trials)}")
print(f"  Best trial: #{study.best_trial.number}")
print(f"  Best CV AUC: {study.best_value:.4f}")
print()
print("  Best hyperparameters:")
for param, value in study.best_params.items():
    print(f"    {param}: {value}")
print()

# Save best parameters
best_params_full = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': RANDOM_STATE,
    'is_unbalance': True,
    **study.best_params
}

with open(f'{DATA_DIR}/best_params_lightgbm_adasyn.json', 'w') as f:
    json.dump(best_params_full, f, indent=2)
print(f"  Saved best parameters to: {DATA_DIR}/best_params_lightgbm_adasyn.json")
print()

# ============================================================================
# STEP 5: Train Final Model and Generate Predictions
# ============================================================================
print("[Step 5/5] Training final model with best parameters...")
start_time = time.time()

# Train on full resampled training set
train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)

final_model = lgb.train(
    best_params_full,
    train_data,
    num_boost_round=2000,
    callbacks=[lgb.log_evaluation(period=100)]
)

print(f"  Training complete! Best iteration: {final_model.best_iteration}")
print(f"  Time: {time.time() - start_time:.1f}s")
print()

# Cross-validation evaluation
print("  Evaluating final model with 5-fold CV...")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_resampled, y_train_resampled)):
    X_tr, X_val = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
    y_tr, y_val = y_train_resampled[train_idx], y_train_resampled[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)

    model = lgb.train(
        best_params_full,
        train_data,
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    cv_scores.append(auc)
    print(f"    Fold {fold+1}: AUC = {auc:.4f}")

print(f"  Mean CV AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"  GINI: {2*np.mean(cv_scores)-1:.4f}")
print()

# Generate test predictions
print("  Generating test set predictions...")
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Save predictions
predictions_df = pd.DataFrame({
    'prediction': y_test_pred
})
predictions_df.to_csv(f'{DATA_DIR}/predictions_lightgbm_adasyn_optuna.csv', index=False)
print(f"  Saved predictions to: {DATA_DIR}/predictions_lightgbm_adasyn_optuna.csv")
print()

# Save model
final_model.save_model(f'{DATA_DIR}/model_lightgbm_adasyn_optuna.txt')
print(f"  Saved model to: {DATA_DIR}/model_lightgbm_adasyn_optuna.txt")
print()

# Feature importance
print("  Top 20 most important features:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    print(f"    {row['feature']:<50} {row['importance']:>10.0f}")

feature_importance.to_csv(f'{DATA_DIR}/feature_importance_lightgbm_adasyn_optuna.csv', index=False)
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Approach: LightGBM + ADASYN + Optuna")
print(f"Cross-validation AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"GINI coefficient: {2*np.mean(cv_scores)-1:.4f}")
print(f"Best iteration: {final_model.best_iteration}")
print(f"Optimization trials: {len(study.trials)}")
print(f"Total runtime: {optimization_time/3600:.2f} hours")
print()
print("Comparison with baseline:")
print(f"  Baseline XGBoost:     AUC = 0.8047, GINI = 0.6094")
print(f"  This approach:        AUC = {np.mean(cv_scores):.4f}, GINI = {2*np.mean(cv_scores)-1:.4f}")
print(f"  Improvement:          AUC = {np.mean(cv_scores)-0.8047:+.4f} ({(np.mean(cv_scores)-0.8047)*100:+.2f}%)")
print()
print("Files saved:")
print(f"  - {DATA_DIR}/best_params_lightgbm_adasyn.json")
print(f"  - {DATA_DIR}/model_lightgbm_adasyn_optuna.txt")
print(f"  - {DATA_DIR}/predictions_lightgbm_adasyn_optuna.csv")
print(f"  - {DATA_DIR}/feature_importance_lightgbm_adasyn_optuna.csv")
print()
print("Next steps:")
print("  1. If AUC > 0.82: Congratulations! You've reached the target.")
print("  2. If AUC < 0.82: Try Approach #2 (Ensemble Stacking)")
print("  3. Optional: Run probability calibration for production deployment")
print()
print("="*80)
