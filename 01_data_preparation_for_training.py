"""
Credit Default Prediction - Data Preparation for Model Training
================================================================

This script prepares the cleaned and imputed dataset for machine learning model training.

Input: final_dataset_imputed.parquet (89,999 rows × 62 columns)
Output: Train/test splits with engineered features and proper encoding

Author: Data Science Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
print(f"\nColumn names:\n{df.columns.tolist()}")

# Check data types
print(f"\nData types:\n{df.dtypes.value_counts()}")

# Target variable analysis
print(f"\n--- Target Variable Analysis ---")
print(f"Target column: 'default'")
print(f"Value counts:\n{df['default'].value_counts()}")
print(f"Default rate: {df['default'].mean():.4f} ({df['default'].mean()*100:.2f}%)")

# Verify no missing values
print(f"\nMissing values per column:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values detected")
else:
    print(missing[missing > 0])

# ============================================================================
# STEP 2: Identify Feature Types
# ============================================================================
print("\n[2/8] Identifying feature types...")

# Separate target variable
target_col = 'default'
y = df[target_col]
X = df.drop(columns=[target_col])

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

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features[:10]}...")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"Binary features ({len(binary_features)}): {binary_features}")

# ============================================================================
# STEP 3: Feature Engineering
# ============================================================================
print("\n[3/8] Engineering new features...")

X_engineered = X.copy()

# --- Financial Ratio Combinations ---
if 'annual_income' in X.columns and 'total_debt_amount' in X.columns:
    # Total debt to annual income ratio
    X_engineered['total_debt_to_income'] = X['total_debt_amount'] / (X['annual_income'] + 1)
    print("  ✓ Created: total_debt_to_income")

if 'revolving_balance' in X.columns and 'annual_income' in X.columns:
    # Revolving balance to income ratio
    X_engineered['revolving_to_income'] = X['revolving_balance'] / (X['annual_income'] + 1)
    print("  ✓ Created: revolving_to_income")

if 'monthly_payment' in X.columns and 'monthly_income' in X.columns:
    # Payment burden ratio
    X_engineered['payment_burden'] = X['monthly_payment'] / (X['monthly_income'] + 1)
    print("  ✓ Created: payment_burden")

# --- Behavioral Metrics ---
if 'num_login_sessions' in X.columns and 'account_open_year' in X.columns:
    # Calculate account age (assuming current year is 2025)
    current_year = 2025
    account_age = current_year - X['account_open_year']
    account_age = account_age.clip(lower=1)  # Avoid division by zero
    X_engineered['logins_per_year'] = X['num_login_sessions'] / account_age
    print("  ✓ Created: logins_per_year")

if 'num_customer_service_calls' in X.columns and 'account_open_year' in X.columns:
    account_age = (2025 - X['account_open_year']).clip(lower=1)
    X_engineered['service_calls_per_year'] = X['num_customer_service_calls'] / account_age
    print("  ✓ Created: service_calls_per_year")

# --- Credit Utilization Categories ---
if 'credit_utilization' in X.columns:
    # Binned credit utilization (low/medium/high risk)
    X_engineered['credit_util_category'] = pd.cut(
        X['credit_utilization'],
        bins=[-0.1, 0.3, 0.6, 1.1],
        labels=['low', 'medium', 'high']
    ).astype(str)
    print("  ✓ Created: credit_util_category")

# --- Age Groups ---
if 'age' in X.columns:
    X_engineered['age_group'] = pd.cut(
        X['age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '55+']
    ).astype(str)
    print("  ✓ Created: age_group")

# --- Employment Length Groups ---
if 'employment_length' in X.columns:
    X_engineered['employment_stability'] = pd.cut(
        X['employment_length'],
        bins=[-1, 1, 3, 5, 50],
        labels=['new', 'short', 'medium', 'long']
    ).astype(str)
    print("  ✓ Created: employment_stability")

# --- Temporal Features ---
if 'application_hour' in X.columns:
    # Time of day categories
    X_engineered['application_time_of_day'] = pd.cut(
        X['application_hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    ).astype(str)
    print("  ✓ Created: application_time_of_day")

# --- Debt Service Coverage ---
if 'monthly_income' in X.columns and 'existing_monthly_debt' in X.columns and 'monthly_payment' in X.columns:
    total_monthly_obligation = X['existing_monthly_debt'] + X['monthly_payment']
    X_engineered['debt_service_coverage'] = X['monthly_income'] / (total_monthly_obligation + 1)
    print("  ✓ Created: debt_service_coverage")

# --- Available Credit After Loan ---
if 'available_credit' in X.columns and 'loan_amount' in X.columns:
    X_engineered['available_credit_after_loan'] = X['available_credit'] - X['loan_amount']
    print("  ✓ Created: available_credit_after_loan")

# --- Interaction: Regional Economics + Income ---
if 'regional_median_income' in X.columns and 'annual_income' in X.columns:
    X_engineered['income_vs_regional'] = X['annual_income'] / (X['regional_median_income'] + 1)
    print("  ✓ Created: income_vs_regional")

# --- Combined Risk Score (simple heuristic) ---
if all(col in X.columns for col in ['debt_to_income_ratio', 'credit_utilization', 'num_customer_service_calls']):
    X_engineered['combined_risk_score'] = (
        X['debt_to_income_ratio'] * 0.4 +
        X['credit_utilization'] * 0.4 +
        (X['num_customer_service_calls'] / 10) * 0.2
    )
    print("  ✓ Created: combined_risk_score")

print(f"\nEngineered features added: {X_engineered.shape[1] - X.shape[1]}")
print(f"New dataset shape: {X_engineered.shape}")

# ============================================================================
# STEP 4: Identify Categorical Features (including new ones)
# ============================================================================
print("\n[4/8] Identifying categorical features for encoding...")

# Re-identify categorical features after engineering
categorical_features_all = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nAll categorical features ({len(categorical_features_all)}):")
for cat_col in categorical_features_all:
    cardinality = X_engineered[cat_col].nunique()
    print(f"  - {cat_col}: {cardinality} unique values")

# Separate by cardinality for different encoding strategies
low_cardinality_cats = []  # One-hot encoding (< 10 categories)
high_cardinality_cats = []  # Frequency or target encoding (>= 10 categories)

for col in categorical_features_all:
    cardinality = X_engineered[col].nunique()
    if cardinality < 10:
        low_cardinality_cats.append(col)
    else:
        high_cardinality_cats.append(col)

print(f"\nLow cardinality (one-hot encoding): {low_cardinality_cats}")
print(f"High cardinality (frequency encoding): {high_cardinality_cats}")

# ============================================================================
# STEP 5: Train/Test Split (BEFORE encoding to avoid data leakage)
# ============================================================================
print("\n[5/8] Creating stratified train/test split...")

# Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTrain set: {X_train.shape[0]} rows ({X_train.shape[0]/len(X_engineered)*100:.1f}%)")
print(f"Test set:  {X_test.shape[0]} rows ({X_test.shape[0]/len(X_engineered)*100:.1f}%)")
print(f"\nClass balance in train: {y_train.mean():.4f}")
print(f"Class balance in test:  {y_test.mean():.4f}")

# ============================================================================
# STEP 6: Categorical Encoding (fit on train, transform both)
# ============================================================================
print("\n[6/8] Encoding categorical features...")

# Copy datasets to avoid modifying originals
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# --- One-Hot Encoding for Low Cardinality Features ---
if low_cardinality_cats:
    print(f"\nApplying one-hot encoding to {len(low_cardinality_cats)} features...")

    # Get dummies for train set
    X_train_dummies = pd.get_dummies(
        X_train_encoded[low_cardinality_cats],
        prefix=low_cardinality_cats,
        drop_first=True
    )

    # Get dummies for test set
    X_test_dummies = pd.get_dummies(
        X_test_encoded[low_cardinality_cats],
        prefix=low_cardinality_cats,
        drop_first=True
    )

    # Align columns (ensure test has same columns as train)
    X_test_dummies = X_test_dummies.reindex(columns=X_train_dummies.columns, fill_value=0)

    # Drop original categorical columns
    X_train_encoded = X_train_encoded.drop(columns=low_cardinality_cats)
    X_test_encoded = X_test_encoded.drop(columns=low_cardinality_cats)

    # Concatenate encoded features
    X_train_encoded = pd.concat([X_train_encoded, X_train_dummies], axis=1)
    X_test_encoded = pd.concat([X_test_encoded, X_test_dummies], axis=1)

    print(f"  ✓ Created {len(X_train_dummies.columns)} one-hot encoded features")

# --- Frequency Encoding for High Cardinality Features ---
frequency_encoders = {}
if high_cardinality_cats:
    print(f"\nApplying frequency encoding to {len(high_cardinality_cats)} features...")

    for col in high_cardinality_cats:
        # Calculate frequency on training set
        freq_map = X_train_encoded[col].value_counts(normalize=True).to_dict()
        frequency_encoders[col] = freq_map

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
remaining_cats = X_train_encoded.select_dtypes(include=['object', 'category']).columns
if len(remaining_cats) > 0:
    print(f"\n⚠ Warning: {len(remaining_cats)} categorical features remain: {remaining_cats.tolist()}")
else:
    print(f"\n  ✓ All categorical features encoded successfully")

# ============================================================================
# STEP 7: Feature Scaling (Optional - for non-tree models)
# ============================================================================
print("\n[7/8] Creating feature scaler (for non-tree models)...")

# Identify numeric features for scaling
numeric_cols = X_train_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create scaler fitted on training data only
scaler = StandardScaler()
scaler.fit(X_train_encoded[numeric_cols])

# Create scaled versions (save for logistic regression, neural nets, etc.)
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

X_train_scaled[numeric_cols] = scaler.transform(X_train_encoded[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test_encoded[numeric_cols])

print(f"  ✓ Scaler fitted on {len(numeric_cols)} numeric features")
print(f"  Note: Tree-based models don't require scaling - saved separately")

# ============================================================================
# STEP 8: Save Prepared Datasets
# ============================================================================
print("\n[8/8] Saving prepared datasets...")

# Save unscaled versions (for tree-based models: RF, XGBoost, LightGBM, CatBoost)
X_train_encoded.to_parquet('/home/dr/cbu/X_train.parquet', index=False)
X_test_encoded.to_parquet('/home/dr/cbu/X_test.parquet', index=False)
y_train.to_frame().to_parquet('/home/dr/cbu/y_train.parquet', index=False)
y_test.to_frame().to_parquet('/home/dr/cbu/y_test.parquet', index=False)

print(f"  ✓ Saved: X_train.parquet ({X_train_encoded.shape})")
print(f"  ✓ Saved: X_test.parquet ({X_test_encoded.shape})")
print(f"  ✓ Saved: y_train.parquet ({y_train.shape})")
print(f"  ✓ Saved: y_test.parquet ({y_test.shape})")

# Save scaled versions (for linear models, neural networks)
X_train_scaled.to_parquet('/home/dr/cbu/X_train_scaled.parquet', index=False)
X_test_scaled.to_parquet('/home/dr/cbu/X_test_scaled.parquet', index=False)

print(f"  ✓ Saved: X_train_scaled.parquet (for logistic regression, neural nets)")
print(f"  ✓ Saved: X_test_scaled.parquet")

# Save feature names
feature_names = X_train_encoded.columns.tolist()
with open('/home/dr/cbu/feature_names.txt', 'w') as f:
    for fname in feature_names:
        f.write(f"{fname}\n")

print(f"  ✓ Saved: feature_names.txt ({len(feature_names)} features)")

# Save preprocessing objects
preprocessing_objects = {
    'frequency_encoders': frequency_encoders,
    'scaler': scaler,
    'numeric_cols': numeric_cols,
    'feature_names': feature_names,
    'random_state': RANDOM_STATE,
    'test_size': 0.2,
    'one_hot_columns': X_train_dummies.columns.tolist() if low_cardinality_cats else []
}

with open('/home/dr/cbu/preprocessing_pipeline.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)

print(f"  ✓ Saved: preprocessing_pipeline.pkl")

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

Class Balance:
  - Default rate: {y.mean():.4f} ({y.mean()*100:.2f}%)
  - Class ratio (non-default:default): 1:{(1-y.mean())/y.mean():.1f}

Train/Test Split:
  - Training set: {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X_engineered)*100:.1f}%)
  - Test set: {X_test.shape[0]:,} rows ({X_test.shape[0]/len(X_engineered)*100:.1f}%)
  - Stratified: Yes (preserves class balance)

Feature Engineering:
  - New features created: {X_engineered.shape[1] - X.shape[1]}
  - Financial ratios: total_debt_to_income, revolving_to_income, payment_burden, etc.
  - Behavioral metrics: logins_per_year, service_calls_per_year
  - Categorical bins: credit_util_category, age_group, employment_stability
  - Temporal features: application_time_of_day
  - Interaction features: income_vs_regional, combined_risk_score

Encoding Strategy:
  - One-hot encoding: {len(low_cardinality_cats)} features → {len(X_train_dummies.columns) if low_cardinality_cats else 0} dummy variables
  - Frequency encoding: {len(high_cardinality_cats)} features

Saved Files:
  ✓ X_train.parquet, X_test.parquet (unscaled - for tree models)
  ✓ X_train_scaled.parquet, X_test_scaled.parquet (scaled - for linear models)
  ✓ y_train.parquet, y_test.parquet (target variables)
  ✓ feature_names.txt (list of all features)
  ✓ preprocessing_pipeline.pkl (encoders and scaler for reproducibility)

Recommendations for Model Training:
  1. Start with tree-based models (use unscaled data):
     - Random Forest
     - XGBoost
     - LightGBM
     - CatBoost

  2. Try linear models (use scaled data):
     - Logistic Regression
     - Ridge/Lasso Regression

  3. Handle class imbalance:
     - Use class_weight='balanced' parameter
     - Try SMOTE oversampling
     - Adjust classification threshold based on ROC curve

  4. Evaluation:
     - Primary metric: AUC (Area Under ROC Curve)
     - Use cross-validation for robust estimates
     - Track precision, recall, F1-score

  5. Feature importance:
     - Analyze feature importances from tree models
     - Consider feature selection based on importance
     - Check for multicollinearity in linear models

Next Steps:
  → Run train_model.py to train baseline models
  → Compare AUC scores across different algorithms
  → Tune hyperparameters for best performing model
  → Create final prediction pipeline
""")

print("=" * 80)
print("Data preparation complete! Ready for model training.")
print("=" * 80)
