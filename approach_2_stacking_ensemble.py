"""
Approach #2: Ensemble Stacking (LightGBM + XGBoost + CatBoost)

Expected AUC: 0.83-0.85
Implementation time: 4-5 hours
Target improvement: +0.025-0.045 AUC

This script implements a stacking ensemble combining three gradient boosting algorithms:
1. LightGBM (leaf-wise growth, fast training)
2. XGBoost (level-wise growth, strong regularization)
3. CatBoost (ordered boosting, symmetric trees)
4. LogisticRegression meta-learner to combine predictions

The ensemble leverages complementary strengths of each algorithm.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import ADASYN
import warnings
import time
import pickle

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
USE_ADASYN = True  # Set to False to skip ADASYN

# Paths
DATA_DIR = '/home/dr/cbu'
X_TRAIN_PATH = f'{DATA_DIR}/X_train_engineered.parquet'
Y_TRAIN_PATH = f'{DATA_DIR}/y_train.parquet'
X_TEST_PATH = f'{DATA_DIR}/X_test_engineered.parquet'

print("="*80)
print("APPROACH #2: ENSEMBLE STACKING")
print("="*80)
print()

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("[Step 1/6] Loading data...")
start_time = time.time()

X_train = pd.read_parquet(X_TRAIN_PATH)
y_train = pd.read_parquet(Y_TRAIN_PATH).values.ravel()
X_test = pd.read_parquet(X_TEST_PATH)

print(f"  Training set: {X_train.shape}")
print(f"  Test set: {X_test.shape}")

# Check class distribution
unique, counts = np.unique(y_train, dtype=int)
class_ratio = counts[0] / counts[1]
minority_pct = (counts[1] / len(y_train)) * 100

print(f"  Class distribution:")
print(f"    Class 0: {counts[0]:,} ({100-minority_pct:.2f}%)")
print(f"    Class 1: {counts[1]:,} ({minority_pct:.2f}%)")
print(f"    Ratio: {class_ratio:.1f}:1")
print(f"  Time: {time.time() - start_time:.1f}s")
print()

# ============================================================================
# STEP 2: Apply ADASYN (Optional)
# ============================================================================
if USE_ADASYN:
    print("[Step 2/6] Applying ADASYN over-sampling...")
    start_time = time.time()

    TARGET_MINORITY_PCT = 13.2
    target_sampling_ratio = (TARGET_MINORITY_PCT / 100) / (1 - TARGET_MINORITY_PCT / 100)

    print(f"  Target ratio: 6.6:1 (minority={TARGET_MINORITY_PCT:.1f}%)")

    adasyn = ADASYN(
        sampling_strategy=target_sampling_ratio,
        random_state=RANDOM_STATE,
        n_neighbors=5,
        n_jobs=-1
    )

    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

    unique_res, counts_res = np.unique(y_train_resampled, dtype=int)
    print(f"  Resampled: {len(y_train):,} -> {len(y_train_resampled):,} samples")
    print(f"  New minority: {counts_res[1]/len(y_train_resampled)*100:.2f}%")
    print(f"  Time: {time.time() - start_time:.1f}s")
    print()
else:
    print("[Step 2/6] Skipping ADASYN (USE_ADASYN=False)")
    X_train_resampled = X_train
    y_train_resampled = y_train
    print()

# ============================================================================
# STEP 3: Define Base Models
# ============================================================================
print("[Step 3/6] Configuring base models...")
print()

# Get class weight for imbalance handling
scale_pos_weight = np.sum(y_train_resampled == 0) / np.sum(y_train_resampled == 1)
print(f"  Scale_pos_weight: {scale_pos_weight:.2f}")
print()

# Model 1: LightGBM
print("  [1/3] LightGBM Configuration:")
lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 127,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'is_unbalance': True,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}
lgb_model = lgb.LGBMClassifier(**lgb_params)
print(f"    - Learning rate: {lgb_params['learning_rate']}")
print(f"    - Max depth: {lgb_params['max_depth']}")
print(f"    - Num leaves: {lgb_params['num_leaves']}")
print(f"    - Imbalance handling: is_unbalance=True")
print()

# Model 2: XGBoost
print("  [2/3] XGBoost Configuration:")
xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'auc'
}
xgb_model = xgb.XGBClassifier(**xgb_params)
print(f"    - Learning rate: {xgb_params['learning_rate']}")
print(f"    - Max depth: {xgb_params['max_depth']}")
print(f"    - Imbalance handling: scale_pos_weight={scale_pos_weight:.2f}")
print()

# Model 3: CatBoost
print("  [3/3] CatBoost Configuration:")
cat_params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 5,
    'auto_class_weights': 'Balanced',
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'thread_count': -1
}
cat_model = CatBoostClassifier(**cat_params)
print(f"    - Learning rate: {cat_params['learning_rate']}")
print(f"    - Depth: {cat_params['depth']}")
print(f"    - Imbalance handling: auto_class_weights='Balanced'")
print()

# ============================================================================
# STEP 4: Evaluate Individual Base Models
# ============================================================================
print("[Step 4/6] Evaluating individual base models...")
print("  Running 5-fold cross-validation for each model...")
print()

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Evaluate LightGBM
print("  [1/3] LightGBM:")
start_time = time.time()
lgb_scores = cross_val_score(
    lgb_model, X_train_resampled, y_train_resampled,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
print(f"    CV AUC: {lgb_scores.mean():.4f} +/- {lgb_scores.std():.4f}")
print(f"    GINI: {2*lgb_scores.mean()-1:.4f}")
print(f"    Time: {time.time() - start_time:.1f}s")
print()

# Evaluate XGBoost
print("  [2/3] XGBoost:")
start_time = time.time()
xgb_scores = cross_val_score(
    xgb_model, X_train_resampled, y_train_resampled,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
print(f"    CV AUC: {xgb_scores.mean():.4f} +/- {xgb_scores.std():.4f}")
print(f"    GINI: {2*xgb_scores.mean()-1:.4f}")
print(f"    Time: {time.time() - start_time:.1f}s")
print()

# Evaluate CatBoost
print("  [3/3] CatBoost:")
start_time = time.time()
cat_scores = cross_val_score(
    cat_model, X_train_resampled, y_train_resampled,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
print(f"    CV AUC: {cat_scores.mean():.4f} +/- {cat_scores.std():.4f}")
print(f"    GINI: {2*cat_scores.mean()-1:.4f}")
print(f"    Time: {time.time() - start_time:.1f}s")
print()

# Summary of base models
print("  Base Models Summary:")
print(f"    Best single model: ", end="")
best_single_auc = max(lgb_scores.mean(), xgb_scores.mean(), cat_scores.mean())
if best_single_auc == lgb_scores.mean():
    print(f"LightGBM (AUC={lgb_scores.mean():.4f})")
elif best_single_auc == xgb_scores.mean():
    print(f"XGBoost (AUC={xgb_scores.mean():.4f})")
else:
    print(f"CatBoost (AUC={cat_scores.mean():.4f})")
print()

# ============================================================================
# STEP 5: Build Stacking Ensemble
# ============================================================================
print("[Step 5/6] Building stacking ensemble...")
print()

# Define base estimators
base_estimators = [
    ('lightgbm', lgb_model),
    ('xgboost', xgb_model),
    ('catboost', cat_model)
]

# Define meta-learner
meta_learner = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced',
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_STATE
)

print("  Meta-learner: LogisticRegression")
print("    - Regularization: L2 (C=1.0)")
print("    - Class weight: balanced")
print()

# Build stacking classifier
print("  Creating StackingClassifier...")
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=cv,  # Use same CV splits for OOF predictions
    passthrough=False,  # Set to True to add original features
    n_jobs=-1,
    verbose=1
)

# Train stacking model
print("  Training stacking ensemble (this may take 10-20 minutes)...")
start_time = time.time()

stacking_clf.fit(X_train_resampled, y_train_resampled)

print(f"  Training complete! Time: {(time.time() - start_time)/60:.1f} minutes")
print()

# Evaluate stacking ensemble
print("  Evaluating stacking ensemble with 5-fold CV...")
start_time = time.time()

stacking_scores = cross_val_score(
    stacking_clf, X_train_resampled, y_train_resampled,
    cv=cv, scoring='roc_auc', n_jobs=-1
)

print(f"  Stacking CV AUC: {stacking_scores.mean():.4f} +/- {stacking_scores.std():.4f}")
print(f"  GINI: {2*stacking_scores.mean()-1:.4f}")
print(f"  Time: {(time.time() - start_time)/60:.1f} minutes")
print()

# ============================================================================
# STEP 6: Generate Predictions
# ============================================================================
print("[Step 6/6] Generating test predictions...")
start_time = time.time()

# Retrain on full dataset
stacking_clf.fit(X_train_resampled, y_train_resampled)

# Predict on test set
y_test_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]

# Save predictions
predictions_df = pd.DataFrame({
    'prediction': y_test_pred_proba
})
predictions_df.to_csv(f'{DATA_DIR}/predictions_stacking_ensemble.csv', index=False)
print(f"  Saved predictions to: {DATA_DIR}/predictions_stacking_ensemble.csv")

# Save model
with open(f'{DATA_DIR}/model_stacking_ensemble.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f)
print(f"  Saved model to: {DATA_DIR}/model_stacking_ensemble.pkl")

print(f"  Time: {time.time() - start_time:.1f}s")
print()

# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================
print("="*80)
print("RESULTS COMPARISON")
print("="*80)
print()

print("Individual Base Models:")
print(f"  LightGBM:  AUC = {lgb_scores.mean():.4f} +/- {lgb_scores.std():.4f}")
print(f"  XGBoost:   AUC = {xgb_scores.mean():.4f} +/- {xgb_scores.std():.4f}")
print(f"  CatBoost:  AUC = {cat_scores.mean():.4f} +/- {cat_scores.std():.4f}")
print()

print("Ensemble:")
print(f"  Stacking:  AUC = {stacking_scores.mean():.4f} +/- {stacking_scores.std():.4f}")
print()

print("Improvement:")
improvement = stacking_scores.mean() - best_single_auc
print(f"  vs Best Single Model: {improvement:+.4f} AUC ({improvement*100:+.2f}%)")
print()

print("Comparison with Baseline:")
baseline_auc = 0.8047
improvement_baseline = stacking_scores.mean() - baseline_auc
print(f"  Baseline XGBoost:      AUC = 0.8047, GINI = 0.6094")
print(f"  Stacking Ensemble:     AUC = {stacking_scores.mean():.4f}, GINI = {2*stacking_scores.mean()-1:.4f}")
print(f"  Improvement:           {improvement_baseline:+.4f} AUC ({improvement_baseline/baseline_auc*100:+.2f}%)")
print()

# ============================================================================
# META-LEARNER COEFFICIENTS
# ============================================================================
print("="*80)
print("META-LEARNER ANALYSIS")
print("="*80)
print()

# Get meta-learner coefficients (weights for each base model)
meta_coef = stacking_clf.final_estimator_.coef_[0]
meta_intercept = stacking_clf.final_estimator_.intercept_[0]

print("Logistic Regression Coefficients:")
print(f"  Intercept: {meta_intercept:.4f}")
print()
print("Base Model Weights:")
for (name, _), coef in zip(base_estimators, meta_coef):
    print(f"  {name:<15} {coef:>8.4f}")
print()

# Softmax to interpret as "importance"
weights = np.exp(meta_coef) / np.sum(np.exp(meta_coef))
print("Normalized Weights (softmax):")
for (name, _), weight in zip(base_estimators, weights):
    print(f"  {name:<15} {weight:>8.2%}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)
print()

print("Stacking Ensemble Configuration:")
print(f"  Base models: LightGBM, XGBoost, CatBoost")
print(f"  Meta-learner: LogisticRegression (L2, balanced)")
print(f"  Cross-validation: {N_FOLDS}-fold StratifiedKFold")
print(f"  ADASYN: {'Enabled (6.6:1 ratio)' if USE_ADASYN else 'Disabled'}")
print()

print("Performance:")
print(f"  CV AUC: {stacking_scores.mean():.4f} +/- {stacking_scores.std():.4f}")
print(f"  GINI: {2*stacking_scores.mean()-1:.4f}")
print(f"  Improvement vs baseline: {improvement_baseline:+.4f} AUC")
print()

print("Files saved:")
print(f"  - {DATA_DIR}/predictions_stacking_ensemble.csv")
print(f"  - {DATA_DIR}/model_stacking_ensemble.pkl")
print()

print("Next steps:")
print(f"  1. Current AUC: {stacking_scores.mean():.4f}")
if stacking_scores.mean() >= 0.85:
    print("     EXCELLENT! You've exceeded the 0.85 target!")
elif stacking_scores.mean() >= 0.83:
    print("     GREAT! You've reached the 0.83-0.85 range!")
elif stacking_scores.mean() >= 0.82:
    print("     GOOD! You've reached the 0.82 target!")
else:
    print("     Try advanced stacking with 9 models (3 variants each)")
    print("     Run: python approach_2_stacking_advanced.py")
print()
print("  2. Optional: Probability calibration (isotonic regression)")
print("  3. Optional: SHAP analysis for interpretability")
print()
print("="*80)
