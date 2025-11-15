"""
Credit Default Prediction - Data Preparation for Model Training (Pandas-Only Version)
====================================================================================

This script prepares the cleaned and imputed dataset for machine learning model training
using only pandas and numpy (no sklearn required).

Input: final_dataset_imputed.parquet (89,999 rows × 62 columns)
Output: Train/test splits with engineered features and proper encoding

Author: Data Science Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("CREDIT DEFAULT PREDICTION - DATA PREPARATION FOR ML TRAINING")
print("=" * 80)

# ============================================================================
# STEP 1: Load and Explore Data
# ============================================================================
print("\n[1/8] Loading imputed dataset...")

df = pd.read_parquet('/home/dr/cbu/final_dataset_imputed.parquet')

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Check data types
print(f"\nData types distribution:")
print(df.dtypes.value_counts())

# Target variable analysis
print(f"\n--- Target Variable Analysis ---")
print(f"Target column: 'default'")
print(f"Value counts:\n{df['default'].value_counts()}")
print(f"Default rate: {df['default'].mean():.4f} ({df['default'].mean()*100:.2f}%)")

# Verify no missing values
missing_count = df.isnull().sum().sum()
if missing_count == 0:
    print(f"\n✓ No missing values detected")
else:
    print(f"\n⚠ Warning: {missing_count} missing values found")

# ============================================================================
# STEP 2: Identify Feature Types
# ============================================================================
print("\n[2/8] Identifying feature types...")

# Separate target variable
target_col = 'default'
y = df[target_col].copy()
X = df.drop(columns=[target_col]).copy()

print(f"\nFeature set shape: {X.shape}")
print(f"Target set shape: {y.shape}")

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Identify binary features (0/1 or True/False encoded as integers)
binary_features = []
for col in numeric_features:
    unique_vals = X[col].nunique()
    if unique_vals == 2:
        vals = sorted(X[col].unique())
        if vals == [0, 1] or vals == [0.0, 1.0]:
            binary_features.append(col)

# Remove binary features from numeric features list
numeric_features = [f for f in numeric_features if f not in binary_features]

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Binary features: {len(binary_features)}")

if len(categorical_features) <= 20:
    print(f"\nCategorical features: {categorical_features}")

# ============================================================================
# STEP 3: Feature Engineering
# ============================================================================
print("\n[3/8] Engineering new features...")

X_engineered = X.copy()
features_created = []

# --- Financial Ratio Combinations ---
if 'annual_income' in X.columns and 'total_debt_amount' in X.columns:
    X_engineered['total_debt_to_income'] = X['total_debt_amount'] / (X['annual_income'] + 1)
    features_created.append('total_debt_to_income')

if 'revolving_balance' in X.columns and 'annual_income' in X.columns:
    X_engineered['revolving_to_income'] = X['revolving_balance'] / (X['annual_income'] + 1)
    features_created.append('revolving_to_income')

if 'monthly_payment' in X.columns and 'monthly_income' in X.columns:
    X_engineered['payment_burden'] = X['monthly_payment'] / (X['monthly_income'] + 1)
    features_created.append('payment_burden')

# --- Behavioral Metrics ---
if 'num_login_sessions' in X.columns and 'account_open_year' in X.columns:
    current_year = 2025
    account_age = (current_year - X['account_open_year']).clip(lower=1)
    X_engineered['logins_per_year'] = X['num_login_sessions'] / account_age
    features_created.append('logins_per_year')

if 'num_customer_service_calls' in X.columns and 'account_open_year' in X.columns:
    current_year = 2025
    account_age = (current_year - X['account_open_year']).clip(lower=1)
    X_engineered['service_calls_per_year'] = X['num_customer_service_calls'] / account_age
    features_created.append('service_calls_per_year')

# --- Account Age ---
if 'account_open_year' in X.columns:
    current_year = 2025
    X_engineered['account_age_years'] = current_year - X['account_open_year']
    features_created.append('account_age_years')

# --- Credit Utilization Categories ---
if 'credit_utilization' in X.columns:
    X_engineered['credit_util_category'] = pd.cut(
        X['credit_utilization'],
        bins=[-0.1, 0.3, 0.6, 1.1],
        labels=['low', 'medium', 'high']
    ).astype(str)
    features_created.append('credit_util_category')

# --- Age Groups ---
if 'age' in X.columns:
    X_engineered['age_group'] = pd.cut(
        X['age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '55+']
    ).astype(str)
    features_created.append('age_group')

# --- Employment Length Groups ---
if 'employment_length' in X.columns:
    X_engineered['employment_stability'] = pd.cut(
        X['employment_length'],
        bins=[-1, 1, 3, 5, 50],
        labels=['new', 'short', 'medium', 'long']
    ).astype(str)
    features_created.append('employment_stability')

# --- Temporal Features ---
if 'application_hour' in X.columns:
    X_engineered['application_time_of_day'] = pd.cut(
        X['application_hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    ).astype(str)
    features_created.append('application_time_of_day')

# --- Debt Service Coverage ---
if 'monthly_income' in X.columns and 'existing_monthly_debt' in X.columns and 'monthly_payment' in X.columns:
    total_monthly_obligation = X['existing_monthly_debt'] + X['monthly_payment']
    X_engineered['debt_service_coverage'] = X['monthly_income'] / (total_monthly_obligation + 1)
    features_created.append('debt_service_coverage')

# --- Available Credit After Loan ---
if 'available_credit' in X.columns and 'loan_amount' in X.columns:
    X_engineered['available_credit_after_loan'] = X['available_credit'] - X['loan_amount']
    features_created.append('available_credit_after_loan')

# --- Interaction: Regional Economics + Income ---
if 'regional_median_income' in X.columns and 'annual_income' in X.columns:
    X_engineered['income_vs_regional'] = X['annual_income'] / (X['regional_median_income'] + 1)
    features_created.append('income_vs_regional')

# --- Income per Dependent ---
if 'annual_income' in X.columns and 'num_dependents' in X.columns:
    X_engineered['income_per_dependent'] = X['annual_income'] / (X['num_dependents'] + 1)
    features_created.append('income_per_dependent')

# --- Combined Risk Score (simple heuristic) ---
if all(col in X.columns for col in ['debt_to_income_ratio', 'credit_utilization', 'num_customer_service_calls']):
    X_engineered['combined_risk_score'] = (
        X['debt_to_income_ratio'] * 0.4 +
        X['credit_utilization'] * 0.4 +
        (X['num_customer_service_calls'] / 10) * 0.2
    )
    features_created.append('combined_risk_score')

# --- Credit Age Features ---
if 'available_credit' in X.columns and 'revolving_balance' in X.columns:
    X_engineered['credit_capacity'] = X['available_credit'] + X['revolving_balance']
    features_created.append('credit_capacity')

print(f"\n✓ Created {len(features_created)} new features")
print(f"New dataset shape: {X_engineered.shape}")

# ============================================================================
# STEP 4: Identify Categorical Features (including new ones)
# ============================================================================
print("\n[4/8] Identifying categorical features for encoding...")

categorical_features_all = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nAll categorical features ({len(categorical_features_all)}):")
cat_info = {}
for cat_col in categorical_features_all:
    cardinality = X_engineered[cat_col].nunique()
    cat_info[cat_col] = cardinality
    print(f"  - {cat_col}: {cardinality} unique values")

# Separate by cardinality
low_cardinality_cats = [col for col, card in cat_info.items() if card < 10]
high_cardinality_cats = [col for col, card in cat_info.items() if card >= 10]

print(f"\nLow cardinality (one-hot): {len(low_cardinality_cats)} features")
print(f"High cardinality (frequency): {len(high_cardinality_cats)} features")

# ============================================================================
# STEP 5: Train/Test Split (BEFORE encoding to avoid data leakage)
# ============================================================================
print("\n[5/8] Creating stratified train/test split...")

# Manual stratified split
test_size = 0.2
n_samples = len(X_engineered)

# Shuffle indices for each class
default_indices = y[y == 1].index.tolist()
non_default_indices = y[y == 0].index.tolist()

np.random.shuffle(default_indices)
np.random.shuffle(non_default_indices)

# Split each class
n_test_default = int(len(default_indices) * test_size)
n_test_non_default = int(len(non_default_indices) * test_size)

test_indices = default_indices[:n_test_default] + non_default_indices[:n_test_non_default]
train_indices = default_indices[n_test_default:] + non_default_indices[n_test_non_default:]

# Create splits
X_train = X_engineered.loc[train_indices].copy()
X_test = X_engineered.loc[test_indices].copy()
y_train = y.loc[train_indices].copy()
y_test = y.loc[test_indices].copy()

# Reset indices
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"\nTrain set: {X_train.shape[0]:,} rows ({X_train.shape[0]/n_samples*100:.1f}%)")
print(f"Test set:  {X_test.shape[0]:,} rows ({X_test.shape[0]/n_samples*100:.1f}%)")
print(f"\nClass balance verification:")
print(f"  Train default rate: {y_train.mean():.4f} ({y_train.mean()*100:.2f}%)")
print(f"  Test default rate:  {y_test.mean():.4f} ({y_test.mean()*100:.2f}%)")
print(f"  Original:           {y.mean():.4f} ({y.mean()*100:.2f}%)")

# ============================================================================
# STEP 6: Categorical Encoding (fit on train, transform both)
# ============================================================================
print("\n[6/8] Encoding categorical features...")

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# Track encoding metadata
encoding_metadata = {
    'low_cardinality_features': low_cardinality_cats,
    'high_cardinality_features': high_cardinality_cats,
    'one_hot_columns': [],
    'frequency_encoders': {}
}

# --- One-Hot Encoding for Low Cardinality Features ---
if low_cardinality_cats:
    print(f"\nApplying one-hot encoding to {len(low_cardinality_cats)} features...")

    X_train_dummies = pd.get_dummies(
        X_train_encoded[low_cardinality_cats],
        prefix=low_cardinality_cats,
        drop_first=True,
        dtype=int
    )

    X_test_dummies = pd.get_dummies(
        X_test_encoded[low_cardinality_cats],
        prefix=low_cardinality_cats,
        drop_first=True,
        dtype=int
    )

    # Align columns
    missing_in_test = set(X_train_dummies.columns) - set(X_test_dummies.columns)
    for col in missing_in_test:
        X_test_dummies[col] = 0

    # Reorder to match train
    X_test_dummies = X_test_dummies[X_train_dummies.columns]

    # Save one-hot column names
    encoding_metadata['one_hot_columns'] = X_train_dummies.columns.tolist()

    # Drop original columns
    X_train_encoded = X_train_encoded.drop(columns=low_cardinality_cats)
    X_test_encoded = X_test_encoded.drop(columns=low_cardinality_cats)

    # Concat encoded features
    X_train_encoded = pd.concat([X_train_encoded, X_train_dummies], axis=1)
    X_test_encoded = pd.concat([X_test_encoded, X_test_dummies], axis=1)

    print(f"  ✓ Created {len(X_train_dummies.columns)} one-hot encoded features")

# --- Frequency Encoding for High Cardinality Features ---
if high_cardinality_cats:
    print(f"\nApplying frequency encoding to {len(high_cardinality_cats)} features...")

    for col in high_cardinality_cats:
        # Calculate frequency on training set
        freq_map = X_train_encoded[col].value_counts(normalize=True).to_dict()
        encoding_metadata['frequency_encoders'][col] = freq_map

        # Apply to train and test
        X_train_encoded[f'{col}_freq'] = X_train_encoded[col].map(freq_map).fillna(0)
        X_test_encoded[f'{col}_freq'] = X_test_encoded[col].map(freq_map).fillna(0)

        print(f"  ✓ Encoded: {col} -> {col}_freq")

    # Drop original high cardinality columns
    X_train_encoded = X_train_encoded.drop(columns=high_cardinality_cats)
    X_test_encoded = X_test_encoded.drop(columns=high_cardinality_cats)

print(f"\nEncoded dataset shapes:")
print(f"  Train: {X_train_encoded.shape}")
print(f"  Test:  {X_test_encoded.shape}")

# Verify no categorical features remain
remaining_cats = X_train_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
if len(remaining_cats) > 0:
    print(f"\n⚠ Warning: {len(remaining_cats)} categorical features remain: {remaining_cats}")
else:
    print(f"\n✓ All categorical features encoded successfully")

# ============================================================================
# STEP 7: Feature Scaling (for non-tree models)
# ============================================================================
print("\n[7/8] Creating scaled versions for linear models...")

# Identify numeric features
numeric_cols = X_train_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Calculate mean and std on training data
train_means = X_train_encoded[numeric_cols].mean()
train_stds = X_train_encoded[numeric_cols].std()

# Create scaled versions
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

# Standardize: (x - mean) / std
for col in numeric_cols:
    if train_stds[col] > 0:
        X_train_scaled[col] = (X_train_encoded[col] - train_means[col]) / train_stds[col]
        X_test_scaled[col] = (X_test_encoded[col] - train_means[col]) / train_stds[col]

# Save scaling parameters
scaling_params = {
    'numeric_cols': numeric_cols,
    'means': train_means.to_dict(),
    'stds': train_stds.to_dict()
}

print(f"  ✓ Scaled {len(numeric_cols)} numeric features")

# ============================================================================
# STEP 8: Save Prepared Datasets
# ============================================================================
print("\n[8/8] Saving prepared datasets...")

# Save unscaled versions (for tree-based models)
X_train_encoded.to_parquet('/home/dr/cbu/X_train.parquet', index=False)
X_test_encoded.to_parquet('/home/dr/cbu/X_test.parquet', index=False)
y_train.to_frame(name='default').to_parquet('/home/dr/cbu/y_train.parquet', index=False)
y_test.to_frame(name='default').to_parquet('/home/dr/cbu/y_test.parquet', index=False)

print(f"  ✓ X_train.parquet ({X_train_encoded.shape})")
print(f"  ✓ X_test.parquet ({X_test_encoded.shape})")
print(f"  ✓ y_train.parquet ({len(y_train)} rows)")
print(f"  ✓ y_test.parquet ({len(y_test)} rows)")

# Save scaled versions
X_train_scaled.to_parquet('/home/dr/cbu/X_train_scaled.parquet', index=False)
X_test_scaled.to_parquet('/home/dr/cbu/X_test_scaled.parquet', index=False)

print(f"  ✓ X_train_scaled.parquet (for logistic regression)")
print(f"  ✓ X_test_scaled.parquet")

# Save feature names
feature_names = X_train_encoded.columns.tolist()
with open('/home/dr/cbu/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))

print(f"  ✓ feature_names.txt ({len(feature_names)} features)")

# Save all preprocessing metadata
preprocessing_metadata = {
    'random_state': RANDOM_STATE,
    'test_size': test_size,
    'original_shape': df.shape,
    'engineered_shape': X_engineered.shape,
    'final_shape': X_train_encoded.shape,
    'features_created': features_created,
    'encoding': encoding_metadata,
    'scaling': scaling_params,
    'feature_names': feature_names,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'default_rate_train': float(y_train.mean()),
    'default_rate_test': float(y_test.mean()),
    'default_rate_overall': float(y.mean())
}

with open('/home/dr/cbu/preprocessing_metadata.json', 'w') as f:
    json.dump(preprocessing_metadata, f, indent=2)

print(f"  ✓ preprocessing_metadata.json")

# Save class balance information
class_balance_info = {
    'total_samples': len(y),
    'train_samples': len(y_train),
    'test_samples': len(y_test),
    'defaults_total': int(y.sum()),
    'defaults_train': int(y_train.sum()),
    'defaults_test': int(y_test.sum()),
    'default_rate': float(y.mean()),
    'class_ratio': float((1 - y.mean()) / y.mean()),
    'recommended_class_weights': {
        0: float(len(y) / (2 * (y == 0).sum())),
        1: float(len(y) / (2 * (y == 1).sum()))
    }
}

with open('/home/dr/cbu/class_balance_info.json', 'w') as f:
    json.dump(class_balance_info, f, indent=2)

print(f"  ✓ class_balance_info.json")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPARATION SUMMARY")
print("=" * 80)

print(f"""
Dataset Characteristics:
  - Original dataset: {df.shape[0]:,} rows × {df.shape[1]} columns
  - Features after engineering: {X_engineered.shape[1]} columns
  - Final features after encoding: {len(feature_names)} columns
  - Missing values: 0

Class Balance:
  - Default rate: {y.mean():.4f} ({y.mean()*100:.2f}%)
  - Non-defaults: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)
  - Defaults: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)
  - Class ratio (non-default:default): 1:{(1-y.mean())/y.mean():.1f}

Train/Test Split:
  - Training set: {len(X_train):,} rows ({len(X_train)/len(X_engineered)*100:.1f}%)
  - Test set: {len(X_test):,} rows ({len(X_test)/len(X_engineered)*100:.1f}%)
  - Stratified: Yes (preserves class balance)
  - Random state: {RANDOM_STATE}

Feature Engineering ({len(features_created)} new features):""")

for i, feat in enumerate(features_created, 1):
    print(f"  {i:2d}. {feat}")

print(f"""
Encoding Strategy:
  - One-hot encoding: {len(low_cardinality_cats)} features → {len(encoding_metadata['one_hot_columns'])} dummy variables
  - Frequency encoding: {len(high_cardinality_cats)} features

Feature Scaling:
  - Standardization applied: (x - mean) / std
  - Fitted on training data only
  - Applied to {len(numeric_cols)} numeric features

Files Created:
  ✓ X_train.parquet, X_test.parquet (unscaled - for tree models)
  ✓ X_train_scaled.parquet, X_test_scaled.parquet (scaled - for linear models)
  ✓ y_train.parquet, y_test.parquet (target variables)
  ✓ feature_names.txt (list of {len(feature_names)} features)
  ✓ preprocessing_metadata.json (all transformation details)
  ✓ class_balance_info.json (class weights and balance metrics)

Recommended Class Weights (for imbalanced data):
  - Class 0 (non-default): {class_balance_info['recommended_class_weights'][0]:.4f}
  - Class 1 (default): {class_balance_info['recommended_class_weights'][1]:.4f}

Next Steps:
  1. Train baseline models (Logistic Regression, Random Forest, XGBoost)
  2. Use class_weight parameter or SMOTE to handle imbalance
  3. Evaluate using AUC metric
  4. Perform cross-validation for robust estimates
  5. Tune hyperparameters for best model
  6. Analyze feature importance
""")

print("=" * 80)
print("✓ Data preparation complete! Ready for model training.")
print("=" * 80)
