#!/usr/bin/env python3
"""
XGBoost Leak-Free 90%+ AUC Pipeline v3
Target: Achieve 90%+ test AUC without overfitting
Author: ML Model Selection Expert
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.calibration import CalibratedClassifierCV
import optuna
import joblib
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data():
    """Load leak-free balanced datasets"""
    print("\n" + "="*70)
    print("XGBoost Leak-Free 90%+ AUC Pipeline v3")
    print("="*70)

    print("\n[1/7] Loading leak-free datasets...")

    # Load LEAK-FREE training data (v3 - no data leakage)
    X_train = pd.read_parquet('X_train_leak_free_v3.parquet')
    y_train = pd.read_parquet('y_train_leak_free_v3.parquet').values.ravel()

    # Load LEAK-FREE test data
    X_test = pd.read_parquet('X_test_leak_free_v3.parquet')
    y_test = pd.read_parquet('y_test_leak_free_v3.parquet').values.ravel()

    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Train default rate: {y_train.mean():.2%}")
    print(f"  Test default rate: {y_test.mean():.2%}")

    # Calculate class weight
    train_ratio = (1 - y_train.mean()) / y_train.mean()
    print(f"  Train imbalance ratio: {train_ratio:.1f}:1")

    return X_train, y_train, X_test, y_test, train_ratio


def create_holdout_split(X_train, y_train, val_size=0.2):
    """Create holdout validation set for early stopping"""
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    return X_tr, X_val, y_tr, y_val


def optimize_xgboost(X_train, y_train, scale_pos_weight, n_trials=100):
    """Bayesian hyperparameter optimization with Optuna"""
    print("\n[2/7] Bayesian hyperparameter optimization...")
    print(f"  Running {n_trials} trials (est. ~{n_trials//5}-{n_trials//3} minutes)")

    # Create holdout for early stopping
    X_tr, X_val, y_tr, y_val = create_holdout_split(X_train, y_train)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }

        # Train with validation monitoring (XGBoost 3.x)
        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate on validation set
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        return auc

    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['use_label_encoder'] = False
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1

    print(f"  Best validation AUC: {study.best_value:.4f}")
    print(f"  Best parameters found:")
    for key, value in best_params.items():
        if key not in ['objective', 'eval_metric', 'use_label_encoder', 'n_jobs']:
            print(f"    {key}: {value}")

    return best_params


def generate_oof_predictions(X_train, y_train, params, n_folds=5):
    """Generate out-of-fold predictions for ensemble"""
    print("\n[3/7] Generating OOF predictions...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(X_train))
    feature_importance = np.zeros(X_train.shape[1])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train model
        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict on validation fold
        oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
        feature_importance += model.feature_importances_ / n_folds

        # Report fold performance
        fold_auc = roc_auc_score(y_val, oof_predictions[val_idx])
        print(f"  Fold {fold+1}/{n_folds} AUC: {fold_auc:.4f}")

    # Overall OOF performance
    oof_auc = roc_auc_score(y_train, oof_predictions)
    print(f"  Overall OOF AUC: {oof_auc:.4f}")

    return oof_predictions, feature_importance


def train_ensemble_models(X_train, y_train, X_test, base_params, scale_pos_weight):
    """Train diverse ensemble of XGBoost models"""
    print("\n[4/7] Training ensemble models...")

    ensemble_params = [
        # Model 1: Conservative (prevent overfitting)
        {
            **base_params,
            'max_depth': min(base_params['max_depth'], 4),
            'learning_rate': min(base_params['learning_rate'], 0.01),
            'n_estimators': min(base_params['n_estimators'] + 500, 1500),
            'min_child_weight': max(base_params['min_child_weight'], 10),
            'reg_alpha': max(base_params['reg_alpha'], 1.0),
            'reg_lambda': max(base_params['reg_lambda'], 3.0)
        },
        # Model 2: Balanced (best params)
        base_params.copy(),
        # Model 3: Different random state
        {
            **base_params,
            'max_depth': base_params['max_depth'] - 1,
            'learning_rate': base_params['learning_rate'] * 0.8,
            'n_estimators': base_params['n_estimators'] + 200,
            'random_state': 123
        }
    ]

    models = []
    train_predictions = []
    test_predictions = []

    for i, params in enumerate(ensemble_params):
        print(f"  Training model {i+1}/3...")

        # Create holdout for early stopping
        X_tr, X_val, y_tr, y_val = create_holdout_split(X_train, y_train)

        # Train model
        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Generate predictions
        train_pred = model.predict_proba(X_train)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]

        train_predictions.append(train_pred)
        test_predictions.append(test_pred)
        models.append(model)

        # Report individual model performance
        train_auc = roc_auc_score(y_train, train_pred)
        print(f"    Train AUC: {train_auc:.4f}")

    return models, np.array(train_predictions).T, np.array(test_predictions).T


def optimize_ensemble_weights(train_predictions, y_train):
    """Find optimal ensemble weights"""
    print("\n[5/7] Optimizing ensemble weights...")

    from scipy.optimize import minimize

    def ensemble_score(weights):
        # Normalize weights
        weights = weights / weights.sum()
        # Weighted average
        pred = np.average(train_predictions, weights=weights, axis=1)
        # Negative AUC (for minimization)
        return -roc_auc_score(y_train, pred)

    # Initial equal weights
    n_models = train_predictions.shape[1]
    initial_weights = np.ones(n_models) / n_models

    # Constraints: weights sum to 1, all non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    bounds = [(0, 1)] * n_models

    # Optimize
    result = minimize(
        ensemble_score,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    print(f"  Optimal weights: {optimal_weights}")

    # Final ensemble prediction
    ensemble_train = np.average(train_predictions, weights=optimal_weights, axis=1)
    ensemble_auc = roc_auc_score(y_train, ensemble_train)
    print(f"  Ensemble train AUC: {ensemble_auc:.4f}")

    return optimal_weights


def calibrate_predictions(models, weights, X_train, y_train, X_test):
    """Apply calibration to improve probability estimates"""
    print("\n[6/7] Calibrating predictions...")

    # Create ensemble predictions for calibration
    train_pred = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))

    for model, weight in zip(models, weights):
        train_pred += model.predict_proba(X_train)[:, 1] * weight
        test_pred += model.predict_proba(X_test)[:, 1] * weight

    # Create a simple wrapper for calibration
    class EnsembleModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights

        def predict_proba(self, X):
            pred = np.zeros((len(X), 2))
            for model, weight in zip(self.models, self.weights):
                pred += model.predict_proba(X) * weight
            return pred

        def fit(self, X, y):
            return self

    ensemble_model = EnsembleModel(models, weights)

    # Apply isotonic calibration
    calibrated = CalibratedClassifierCV(
        ensemble_model,
        method='isotonic',
        cv=3  # Use 3-fold for calibration
    )

    calibrated.fit(X_train, y_train)

    # Generate calibrated predictions
    calib_train = calibrated.predict_proba(X_train)[:, 1]
    calib_test = calibrated.predict_proba(X_test)[:, 1]

    # Compare before/after calibration
    uncalib_auc = roc_auc_score(y_train, train_pred)
    calib_auc = roc_auc_score(y_train, calib_train)

    print(f"  Train AUC before calibration: {uncalib_auc:.4f}")
    print(f"  Train AUC after calibration: {calib_auc:.4f}")

    return calibrated, calib_train, calib_test


def evaluate_final_model(y_train, train_pred, y_test, test_pred, feature_importance, feature_names):
    """Comprehensive evaluation of final model"""
    print("\n[7/7] Final evaluation...")

    # Calculate metrics
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    gap = train_auc - test_auc

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    print(f"  Train-Test Gap: {gap:.4f}")

    # Success criteria check
    print("\n  Success Criteria:")
    print(f"    Test AUC >= 0.90: {'PASS' if test_auc >= 0.90 else 'FAIL'} ({test_auc:.4f})")
    print(f"    Train-Test Gap < 0.05: {'PASS' if gap < 0.05 else 'FAIL'} ({gap:.4f})")

    # Additional metrics at optimal threshold
    from sklearn.metrics import confusion_matrix, classification_report

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test, test_pred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n  Optimal threshold: {optimal_threshold:.4f}")

    # Classification report at optimal threshold
    y_pred_binary = (test_pred >= optimal_threshold).astype(int)

    print("\n  Classification Report (at optimal threshold):")
    print(classification_report(y_test, y_pred_binary, target_names=['Non-Default', 'Default']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    print("\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"    FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

    # Feature importance (top 15)
    print("\n  Top 15 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(15)

    for idx, row in importance_df.iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")

    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'gap': gap,
        'optimal_threshold': optimal_threshold,
        'feature_importance': importance_df
    }


def save_artifacts(models, weights, calibrated_model, results):
    """Save all model artifacts"""
    print("\n  Saving model artifacts...")

    # Save individual models
    for i, model in enumerate(models):
        joblib.dump(model, f'xgboost_model_{i+1}_v3.pkl')

    # Save ensemble weights
    np.save('ensemble_weights_v3.npy', weights)

    # Save calibrated model
    joblib.dump(calibrated_model, 'xgboost_calibrated_ensemble_v3.pkl')

    # Save results
    pd.DataFrame([results]).to_csv('model_results_v3.csv', index=False)

    # Save feature importance
    results['feature_importance'].to_csv('feature_importance_v3.csv', index=False)

    print("  All artifacts saved successfully!")


def main():
    """Main pipeline execution"""
    start_time = time.time()

    # Load data
    X_train, y_train, X_test, y_test, scale_pos_weight = load_data()

    # Optimize hyperparameters (50 trials = balance between quality and speed)
    best_params = optimize_xgboost(X_train, y_train, scale_pos_weight, n_trials=50)

    # Generate OOF predictions (for analysis)
    oof_predictions, feature_importance = generate_oof_predictions(
        X_train, y_train, best_params
    )

    # Train ensemble models
    models, train_preds, test_preds = train_ensemble_models(
        X_train, y_train, X_test, best_params, scale_pos_weight
    )

    # Optimize ensemble weights
    weights = optimize_ensemble_weights(train_preds, y_train)

    # Apply calibration
    calibrated, calib_train, calib_test = calibrate_predictions(
        models, weights, X_train, y_train, X_test
    )

    # Final evaluation
    results = evaluate_final_model(
        y_train, calib_train, y_test, calib_test,
        feature_importance, X_train.columns
    )

    # Save artifacts
    save_artifacts(models, weights, calibrated, results)

    # Report execution time
    elapsed_time = (time.time() - start_time) / 60
    print(f"\n  Total execution time: {elapsed_time:.1f} minutes")

    # Final message
    if results['test_auc'] >= 0.90 and results['gap'] < 0.05:
        print("\n" + "="*70)
        print("SUCCESS: Achieved 90%+ test AUC with minimal overfitting!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("Model trained successfully. Check results for performance details.")
        print("="*70)

    return results


if __name__ == "__main__":
    results = main()