"""
Focal Loss Implementation for LightGBM

Focal Loss is designed for imbalanced classification by down-weighting
easy examples and focusing on hard examples.

Formula: FL(p) = -(1-p)^gamma * log(p)

where:
- p: predicted probability
- gamma: focusing parameter (typically 2.0)
- Higher gamma = more focus on hard examples

Expected improvement: +0.01-0.02 AUC vs standard loss
Complexity: Medium (custom gradient/hessian)

Source: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
Implementation: Based on Max Halford's blog post
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import ADASYN
import warnings

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
USE_ADASYN = True
FOCAL_GAMMA = 2.0  # Focusing parameter (1.0-5.0)
FOCAL_ALPHA = 0.25  # Class weight (0.05-0.25 for 5% minority)

# Paths
DATA_DIR = '/home/dr/cbu'
X_TRAIN_PATH = f'{DATA_DIR}/X_train_engineered.parquet'
Y_TRAIN_PATH = f'{DATA_DIR}/y_train.parquet'
X_TEST_PATH = f'{DATA_DIR}/X_test_engineered.parquet'

print("="*80)
print("LightGBM with Focal Loss")
print("="*80)
print()


# ============================================================================
# Focal Loss Implementation
# ============================================================================

def focal_loss_lgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss for LightGBM

    Parameters:
    -----------
    y_pred : array-like
        Raw predictions (not probabilities!)
    dtrain : lgb.Dataset
        Training dataset containing true labels
    alpha : float
        Class weight for positive class (0.25 means minority gets 0.25 weight)
        For imbalanced data with 5% minority, use 0.05-0.25
    gamma : float
        Focusing parameter (0 = cross-entropy, 2 = standard focal)
        Higher gamma = more focus on hard examples
        Typical values: 1.0-5.0

    Returns:
    --------
    grad : array-like
        First derivative (gradient)
    hess : array-like
        Second derivative (hessian)
    """
    # Get true labels
    y_true = dtrain.get_label()

    # Convert raw predictions to probabilities using sigmoid
    # Important: LightGBM provides raw margin scores, not probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Compute gradient (first derivative)
    # For y=1: grad = alpha * [(gamma*(1-p)^(gamma-1)*p*log(p)) + (1-p)^gamma]
    # For y=0: grad = -(1-alpha) * [(gamma*p^(gamma-1)*(1-p)*log(1-p)) + p^gamma]
    grad = np.where(
        y_true == 1,
        alpha * (
            (gamma * (1 - p) ** (gamma - 1) * p * np.log(np.clip(p, 1e-15, 1 - 1e-15))) +
            ((1 - p) ** gamma)
        ),
        -(1 - alpha) * (
            (gamma * p ** (gamma - 1) * (1 - p) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15))) +
            (p ** gamma)
        )
    )

    # Compute hessian (second derivative)
    # For y=1: hess = alpha * complex_formula
    # For y=0: hess = (1-alpha) * complex_formula
    hess = np.where(
        y_true == 1,
        alpha * (
            (gamma * (gamma - 1) * (1 - p) ** (gamma - 2) * p * np.log(np.clip(p, 1e-15, 1 - 1e-15))) +
            (2 * gamma * (1 - p) ** (gamma - 1)) +
            ((1 - p) ** gamma * p)
        ),
        (1 - alpha) * (
            (gamma * (gamma - 1) * p ** (gamma - 2) * (1 - p) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15))) +
            (2 * gamma * p ** (gamma - 1)) +
            (p ** gamma * (1 - p))
        )
    )

    # Clip to avoid numerical issues
    hess = np.clip(hess, 1e-15, 1e15)

    return grad, hess


def focal_loss_eval(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss evaluation metric for LightGBM

    Returns:
    --------
    metric_name : str
    metric_value : float
    is_higher_better : bool
    """
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Compute focal loss value
    focal_loss_value = np.where(
        y_true == 1,
        -alpha * ((1 - p) ** gamma) * np.log(np.clip(p, 1e-15, 1 - 1e-15)),
        -(1 - alpha) * (p ** gamma) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15))
    )

    return 'focal_loss', np.mean(focal_loss_value), False


# ============================================================================
# Load Data
# ============================================================================
print("[1/4] Loading data...")
X_train = pd.read_parquet(X_TRAIN_PATH)
y_train = pd.read_parquet(Y_TRAIN_PATH).values.ravel()
X_test = pd.read_parquet(X_TEST_PATH)

print(f"  Training set: {X_train.shape}")
unique, counts = np.unique(y_train, dtype=int)
minority_pct = (counts[1] / len(y_train)) * 100
print(f"  Class distribution: {counts[0]:,} / {counts[1]:,} ({minority_pct:.2f}% minority)")
print()


# ============================================================================
# Apply ADASYN
# ============================================================================
if USE_ADASYN:
    print("[2/4] Applying ADASYN over-sampling...")
    TARGET_MINORITY_PCT = 13.2
    target_sampling_ratio = (TARGET_MINORITY_PCT / 100) / (1 - TARGET_MINORITY_PCT / 100)

    adasyn = ADASYN(
        sampling_strategy=target_sampling_ratio,
        random_state=RANDOM_STATE,
        n_neighbors=5,
        n_jobs=-1
    )

    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    print(f"  Resampled: {len(y_train):,} -> {len(y_train_resampled):,} samples")
    print()
else:
    X_train_resampled = X_train
    y_train_resampled = y_train
    print("[2/4] Skipping ADASYN")
    print()


# ============================================================================
# Train with Focal Loss
# ============================================================================
print("[3/4] Training LightGBM with Focal Loss...")
print(f"  Focal Loss parameters:")
print(f"    gamma (focusing): {FOCAL_GAMMA}")
print(f"    alpha (class weight): {FOCAL_ALPHA}")
print()

# Create custom objective function with fixed alpha and gamma
def objective(y_pred, dtrain):
    return focal_loss_lgb(y_pred, dtrain, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

def eval_metric(y_pred, dtrain):
    return focal_loss_eval(y_pred, dtrain, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

# LightGBM parameters
params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 127,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

# Cross-validation with focal loss
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_resampled, y_train_resampled)):
    print(f"  Fold {fold+1}/{N_FOLDS}:")

    X_tr, X_val = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
    y_tr, y_val = y_train_resampled[train_idx], y_train_resampled[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with custom focal loss objective
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        fobj=objective,  # Custom objective (focal loss)
        feval=eval_metric,  # Custom evaluation metric
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Predict (returns raw scores, need sigmoid for probabilities)
    y_pred_raw = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_proba = 1.0 / (1.0 + np.exp(-y_pred_raw))  # Convert to probabilities

    # Evaluate with AUC
    auc = roc_auc_score(y_val, y_pred_proba)
    cv_scores.append(auc)

    print(f"    AUC: {auc:.4f}, Best iteration: {model.best_iteration}")

print()
print(f"  Mean CV AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"  GINI: {2*np.mean(cv_scores)-1:.4f}")
print()


# ============================================================================
# Generate Predictions
# ============================================================================
print("[4/4] Generating test predictions...")

# Train final model on full dataset
train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)

final_model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    fobj=objective,
    callbacks=[lgb.log_evaluation(period=200)]
)

# Predict on test set
y_test_pred_raw = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_test_pred_proba = 1.0 / (1.0 + np.exp(-y_test_pred_raw))

# Save
predictions_df = pd.DataFrame({'prediction': y_test_pred_proba})
predictions_df.to_csv(f'{DATA_DIR}/predictions_focal_loss.csv', index=False)
final_model.save_model(f'{DATA_DIR}/model_focal_loss.txt')

print(f"  Saved predictions to: {DATA_DIR}/predictions_focal_loss.csv")
print(f"  Saved model to: {DATA_DIR}/model_focal_loss.txt")
print()


# ============================================================================
# Compare with Standard Loss
# ============================================================================
print("="*80)
print("COMPARISON: Focal Loss vs Standard Loss")
print("="*80)
print()

# Train with standard binary loss for comparison
print("Training LightGBM with standard binary loss...")
params_standard = params.copy()
params_standard['objective'] = 'binary'
params_standard['metric'] = 'auc'
params_standard['is_unbalance'] = True

cv_scores_standard = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_resampled, y_train_resampled)):
    X_tr, X_val = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
    y_tr, y_val = y_train_resampled[train_idx], y_train_resampled[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params_standard,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    cv_scores_standard.append(auc)

print()
print("Results:")
print(f"  Standard Loss: AUC = {np.mean(cv_scores_standard):.4f} +/- {np.std(cv_scores_standard):.4f}")
print(f"  Focal Loss:    AUC = {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"  Difference:    {np.mean(cv_scores) - np.mean(cv_scores_standard):+.4f} AUC")
print()

if np.mean(cv_scores) > np.mean(cv_scores_standard):
    print("Focal Loss WINS! Use this model.")
else:
    print("Standard Loss is better. Stick with standard approach.")
print()


# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)
print()
print("Focal Loss Configuration:")
print(f"  gamma: {FOCAL_GAMMA} (focusing parameter)")
print(f"  alpha: {FOCAL_ALPHA} (class weight)")
print(f"  ADASYN: {'Enabled' if USE_ADASYN else 'Disabled'}")
print()
print("Performance:")
print(f"  CV AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"  GINI: {2*np.mean(cv_scores)-1:.4f}")
print()
print("Comparison with baseline:")
print(f"  Baseline XGBoost:  AUC = 0.8047")
print(f"  Focal Loss:        AUC = {np.mean(cv_scores):.4f}")
print(f"  Improvement:       {np.mean(cv_scores) - 0.8047:+.4f} AUC")
print()
print("Tuning suggestions:")
print("  If AUC is lower than expected:")
print("    - Increase gamma (2.0 -> 3.0) for more focus on hard examples")
print("    - Decrease alpha (0.25 -> 0.10) for less minority class weight")
print("    - Try standard loss instead (focal may not always help)")
print()
print("="*80)
