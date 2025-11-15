"""
Credit Default Prediction - Model Training Script
==================================================

This script trains and evaluates machine learning models for credit default prediction.

Primary Metric: AUC (Area Under ROC Curve)
Data: Preprocessed train/test splits from data preparation step

Usage:
    python3 train_model.py

Author: Data Science Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDIT DEFAULT PREDICTION - MODEL TRAINING")
print("=" * 80)

# ============================================================================
# STEP 1: Load Prepared Data
# ============================================================================
print("\n[1/6] Loading prepared datasets...")

# Load unscaled data (for tree-based models)
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')

# Load scaled data (for linear models)
X_train_scaled = pd.read_parquet('/home/dr/cbu/X_train_scaled.parquet')
X_test_scaled = pd.read_parquet('/home/dr/cbu/X_test_scaled.parquet')

# Load target variables
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')['default']
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')['default']

# Load metadata
with open('/home/dr/cbu/preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

with open('/home/dr/cbu/class_balance_info.json', 'r') as f:
    class_info = json.load(f)

print(f"✓ Loaded training data: {X_train.shape}")
print(f"✓ Loaded test data: {X_test.shape}")
print(f"✓ Target variable shape: Train {y_train.shape}, Test {y_test.shape}")
print(f"✓ Number of features: {len(metadata['feature_names'])}")
print(f"✓ Class balance (train): {y_train.mean():.4f} ({y_train.mean()*100:.2f}% defaults)")

# ============================================================================
# STEP 2: Check Available ML Libraries
# ============================================================================
print("\n[2/6] Checking available machine learning libraries...")

available_libraries = {}

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    available_libraries['sklearn'] = True
    print("✓ scikit-learn available")
except ImportError:
    available_libraries['sklearn'] = False
    print("✗ scikit-learn NOT available - install with: pip install scikit-learn")

try:
    import xgboost as xgb
    available_libraries['xgboost'] = True
    print("✓ XGBoost available")
except ImportError:
    available_libraries['xgboost'] = False
    print("✗ XGBoost NOT available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    available_libraries['lightgbm'] = True
    print("✓ LightGBM available")
except ImportError:
    available_libraries['lightgbm'] = False
    print("✗ LightGBM NOT available - install with: pip install lightgbm")

try:
    import catboost as cb
    available_libraries['catboost'] = True
    print("✓ CatBoost available")
except ImportError:
    available_libraries['catboost'] = False
    print("✗ CatBoost NOT available - install with: pip install catboost")

if not any(available_libraries.values()):
    print("\n❌ ERROR: No ML libraries available!")
    print("Please install at least one of the following:")
    print("  pip install scikit-learn")
    print("  pip install xgboost")
    print("  pip install lightgbm")
    print("  pip install catboost")
    exit(1)

# ============================================================================
# STEP 3: Define Evaluation Function
# ============================================================================
print("\n[3/6] Defining evaluation functions...")

def evaluate_model(y_true, y_pred_proba, model_name):
    """
    Evaluate model performance using AUC and other metrics.

    Parameters:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Name of the model

    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_proba)

    # Get predictions with threshold 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'model': model_name,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

    return results

def print_evaluation(results):
    """Print evaluation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Model: {results['model']}")
    print(f"{'='*60}")
    print(f"AUC Score:       {results['auc']:.4f} ⭐")
    print(f"Precision:       {results['precision']:.4f}")
    print(f"Recall:          {results['recall']:.4f}")
    print(f"F1-Score:        {results['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {results['true_negatives']:,}")
    print(f"  False Positives: {results['false_positives']:,}")
    print(f"  False Negatives: {results['false_negatives']:,}")
    print(f"  True Positives:  {results['true_positives']:,}")
    print(f"{'='*60}")

print("✓ Evaluation functions defined")

# ============================================================================
# STEP 4: Train Baseline Models
# ============================================================================
print("\n[4/6] Training baseline models...")

# Get recommended class weights
class_weights = class_info['recommended_class_weights']
class_weight_dict = {0: class_weights['0'], 1: class_weights['1']}

print(f"Using class weights: {class_weight_dict}")

results_list = []

# --- Logistic Regression (with scaled data) ---
if available_libraries['sklearn']:
    print("\n--- Training Logistic Regression ---")
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(
        class_weight=class_weight_dict,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    lr_model.fit(X_train_scaled, y_train)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    lr_results = evaluate_model(y_test, y_pred_proba_lr, "Logistic Regression")
    print_evaluation(lr_results)
    results_list.append(lr_results)

# --- Random Forest (with unscaled data) ---
if available_libraries['sklearn']:
    print("\n--- Training Random Forest ---")
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20
    )

    rf_model.fit(X_train, y_train)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    rf_results = evaluate_model(y_test, y_pred_proba_rf, "Random Forest")
    print_evaluation(rf_results)
    results_list.append(rf_results)

# --- XGBoost (with unscaled data) ---
if available_libraries['xgboost']:
    print("\n--- Training XGBoost ---")
    import xgboost as xgb

    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )

    xgb_model.fit(X_train, y_train)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    xgb_results = evaluate_model(y_test, y_pred_proba_xgb, "XGBoost")
    print_evaluation(xgb_results)
    results_list.append(xgb_results)

# --- LightGBM (with unscaled data) ---
if available_libraries['lightgbm']:
    print("\n--- Training LightGBM ---")
    import lightgbm as lgb

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgb_model.fit(X_train, y_train)
    y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

    lgb_results = evaluate_model(y_test, y_pred_proba_lgb, "LightGBM")
    print_evaluation(lgb_results)
    results_list.append(lgb_results)

# --- CatBoost (with unscaled data) ---
if available_libraries['catboost']:
    print("\n--- Training CatBoost ---")
    import catboost as cb

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    cat_model = cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=False
    )

    cat_model.fit(X_train, y_train)
    y_pred_proba_cat = cat_model.predict_proba(X_test)[:, 1]

    cat_results = evaluate_model(y_test, y_pred_proba_cat, "CatBoost")
    print_evaluation(cat_results)
    results_list.append(cat_results)

# ============================================================================
# STEP 5: Compare Models
# ============================================================================
print("\n[5/6] Comparing model performance...")

if results_list:
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values('auc', ascending=False)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON (sorted by AUC)")
    print("=" * 80)
    print(comparison_df[['model', 'auc', 'precision', 'recall', 'f1_score']].to_string(index=False))
    print("=" * 80)

    # Save comparison results
    comparison_df.to_csv('/home/dr/cbu/model_comparison.csv', index=False)
    print("\n✓ Saved: model_comparison.csv")

    # Identify best model
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model']} (AUC = {best_model['auc']:.4f})")

# ============================================================================
# STEP 6: Next Steps Recommendations
# ============================================================================
print("\n[6/6] Next steps and recommendations...")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           NEXT STEPS FOR IMPROVEMENT                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. HYPERPARAMETER TUNING
   - Use GridSearchCV or RandomizedSearchCV
   - Tune based on AUC metric
   - Try different combinations of:
     * n_estimators, max_depth, learning_rate (for gradient boosting)
     * max_features, min_samples_split (for Random Forest)

2. FEATURE ENGINEERING
   - Analyze feature importance from tree-based models
   - Create additional domain-specific features
   - Remove low-importance features
   - Consider polynomial features or interactions

3. ADVANCED TECHNIQUES FOR IMBALANCED DATA
   - Try SMOTE oversampling
   - Experiment with different class weights
   - Optimize classification threshold on ROC curve
   - Use stratified k-fold cross-validation

4. ENSEMBLE METHODS
   - Combine predictions from multiple models (voting/averaging)
   - Stack models (use predictions as features)
   - Try different ensemble strategies

5. MODEL INTERPRETATION
   - Use SHAP values for feature importance
   - Analyze misclassified examples
   - Create partial dependence plots
   - Identify which features drive defaults

6. CROSS-VALIDATION
   - Perform stratified k-fold CV (k=5 or k=10)
   - Get more robust AUC estimates
   - Detect overfitting

7. PRODUCTION PIPELINE
   - Save best model with pickle/joblib
   - Create prediction pipeline
   - Add data validation
   - Document model versioning

Example code for next steps:
----------------------------

# Hyperparameter tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Cross-validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
""")

print("\n" + "=" * 80)
print("✓ Model training complete!")
print("=" * 80)
