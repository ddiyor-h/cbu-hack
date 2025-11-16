#!/usr/bin/env python3
"""
CREATE INFERENCE FEATURES
Создает KNN, Target Encoding, WOE признаки для evaluation set
используя обученные статистики из training set
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATING INFERENCE FEATURES FOR EVALUATION SET")
print("="*80)

# Load training data
print("\n[1/3] Loading training data...")
X_train = pd.read_parquet('X_train_leak_free_v3.parquet')
y_train = pd.read_parquet('y_train_leak_free_v3.parquet').values.ravel()

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Default rate: {y_train.mean():.2%}")

# Load evaluation data (already cleaned and merged)
print("\n[2/3] Loading evaluation data...")
X_eval = pd.read_parquet('X_eval_cleaned.parquet')
print(f"✓ Evaluation data: {X_eval.shape}")

# ============================================================================
# KNN FEATURES - Train on full training set, apply to evaluation
# ============================================================================
print("\n[3/3] Creating KNN features...")

def create_knn_features_for_inference(X_train, y_train, X_eval, n_neighbors=50):
    """Train KNN on FULL training set, apply to evaluation"""

    # Get numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target encoding/WOE columns if present
    numeric_cols = [c for c in numeric_cols if 'target_oof' not in c and 'woe_oof' not in c and 'knn_oof' not in c]

    print(f"  Training KNN-{n_neighbors} on {len(numeric_cols)} features...")

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols].fillna(0))
    X_eval_scaled = scaler.transform(X_eval[numeric_cols].fillna(0))

    # Train KNN on FULL training set
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    # Predict on evaluation set
    eval_probs = knn.predict_proba(X_eval_scaled)[:, 1]

    return eval_probs

# Create KNN features for different K values
for n_neighbors in [50, 100, 500]:
    print(f"\n  KNN-{n_neighbors}:")
    knn_probs = create_knn_features_for_inference(X_train, y_train, X_eval, n_neighbors)
    X_eval[f'knn_oof_{n_neighbors}'] = knn_probs
    print(f"    ✓ Mean: {knn_probs.mean():.4f}, Std: {knn_probs.std():.4f}")

# ============================================================================
# TARGET ENCODING - Use global statistics from training
# ============================================================================
print("\n  Target Encoding features...")

def create_target_encoding_for_inference(X_train, y_train, X_eval, column, smoothing=10):
    """Use global target encoding from training set"""

    global_mean = y_train.mean()

    # Calculate encoding from training data
    encoding_dict = {}
    for value in X_train[column].unique():
        mask = X_train[column] == value
        n = mask.sum()
        if n == 0:
            encoding_dict[value] = global_mean
        else:
            category_mean = y_train[mask].mean()
            # Bayesian smoothing
            encoding_dict[value] = (category_mean * n + global_mean * smoothing) / (n + smoothing)

    # Apply to evaluation set
    eval_encoded = X_eval[column].map(encoding_dict).fillna(global_mean)

    return eval_encoded

categorical_cols = ['state', 'marital_status', 'education', 'employment_type']
for col in categorical_cols:
    if col in X_train.columns and col in X_eval.columns:
        encoded = create_target_encoding_for_inference(X_train, y_train, X_eval, col)
        X_eval[f'{col}_target_oof'] = encoded
        print(f"    {col}_target_oof: ✓")

# ============================================================================
# WOE FEATURES - Use bins from training
# ============================================================================
print("\n  WOE features...")

def create_woe_for_inference(X_train, y_train, X_eval, column, n_bins=10):
    """Use WOE bins from training set"""

    try:
        # Create bins from training data
        _, bins = pd.qcut(X_train[column].fillna(0), q=n_bins, duplicates='drop', retbins=True)

        # Bin the training data
        train_binned = pd.cut(X_train[column].fillna(0), bins=bins, include_lowest=True)

        # Calculate WOE for each bin from training data
        woe_dict = {}
        total_good = (y_train == 0).sum()
        total_bad = y_train.sum()

        for bin_label in train_binned.cat.categories:
            mask = train_binned == bin_label
            n_good = (y_train[mask] == 0).sum()
            n_bad = y_train[mask].sum()

            # Smoothing
            n_good = max(n_good, 0.5)
            n_bad = max(n_bad, 0.5)

            # Calculate WOE
            pct_good = n_good / total_good
            pct_bad = n_bad / total_bad
            woe = np.log(pct_good / pct_bad)
            woe_dict[bin_label] = woe

        # Apply to evaluation set
        eval_binned = pd.cut(X_eval[column].fillna(0), bins=bins, include_lowest=True)
        eval_woe = eval_binned.map(woe_dict).fillna(0)

        return eval_woe
    except:
        return np.zeros(len(X_eval))

woe_features = ['debt_to_income_ratio', 'credit_utilization', 'credit_score', 'age']
for feature in woe_features:
    if feature in X_train.columns and feature in X_eval.columns:
        woe = create_woe_for_inference(X_train, y_train, X_eval, feature)
        X_eval[f'{feature}_woe_oof'] = woe
        print(f"    {feature}_woe_oof: ✓")

# ============================================================================
# SAVE
# ============================================================================
print("\n[4/4] Saving enhanced evaluation data...")
X_eval.to_parquet('X_eval_with_inference_features.parquet', index=False)
print(f"✓ Saved: X_eval_with_inference_features.parquet ({X_eval.shape})")

print("\n" + "="*80)
print("INFERENCE FEATURES CREATED")
print("="*80)
print(f"\nAdded features:")
print(f"  KNN: 3 features (knn_oof_50, knn_oof_100, knn_oof_500)")
print(f"  Target Encoding: {len([c for c in categorical_cols if c in X_train.columns])} features")
print(f"  WOE: {len([f for f in woe_features if f in X_train.columns])} features")
