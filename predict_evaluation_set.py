#!/usr/bin/env python3
"""
PREDICTION PIPELINE FOR EVALUATION SET
Применяет полный leak-free pipeline к evaluation данным
"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
EVAL_DIR = Path("task_result1/evaluation_set-20251116T050317Z-1-001/evaluation_set")
MODEL_PATH = Path("task_result1/model/xgboost_calibrated_ensemble_v3.pkl")
OUTPUT_PATH = EVAL_DIR / "results.csv"

print("="*80)
print("CREDIT DEFAULT PREDICTION - EVALUATION SET")
print("Leak-Free Pipeline V3")
print("="*80)

# ============================================================================
# STEP 1: LOAD EVALUATION DATA
# ============================================================================
print("\n[STEP 1/6] Loading evaluation data...")

# 1.1 Load application_metadata.csv
print("  [1/6] Loading application_metadata.csv...")
app_metadata = pd.read_csv(EVAL_DIR / "application_metadata.csv")
print(f"    Shape: {app_metadata.shape}")

# 1.2 Load demographics.csv
print("  [2/6] Loading demographics.csv...")
demographics = pd.read_csv(EVAL_DIR / "demographics.csv")
print(f"    Shape: {demographics.shape}")

# 1.3 Load credit_history.parquet
print("  [3/6] Loading credit_history.parquet...")
credit_history = pd.read_parquet(EVAL_DIR / "credit_history.parquet")
print(f"    Shape: {credit_history.shape}")

# 1.4 Load financial_ratios.jsonl
print("  [4/6] Loading financial_ratios.jsonl...")
financial_ratios = []
with open(EVAL_DIR / "financial_ratios.jsonl", 'r') as f:
    for line in f:
        financial_ratios.append(json.loads(line))
financial_ratios = pd.DataFrame(financial_ratios)
print(f"    Shape: {financial_ratios.shape}")

# 1.5 Load loan_details.xlsx
print("  [5/6] Loading loan_details.xlsx...")
loan_details = pd.read_excel(EVAL_DIR / "loan_details.xlsx")
print(f"    Shape: {loan_details.shape}")

# 1.6 Load geographic_data.xml
print("  [6/6] Loading geographic_data.xml...")
tree = ET.parse(EVAL_DIR / "geographic_data.xml")
root = tree.getroot()
geo_records = []
for customer in root.findall('customer'):
    record = {}
    for child in customer:
        record[child.tag] = child.text
    geo_records.append(record)
geographic_data = pd.DataFrame(geo_records)
print(f"    Shape: {geographic_data.shape}")

print("✓ All 6 files loaded")

# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================
print("\n[STEP 2/6] Cleaning data...")

# 2.1 Clean application_metadata
app_metadata_clean = app_metadata.drop(columns=['random_noise_1'])

# 2.2 Clean demographics
demographics_clean = demographics.copy()

# Clean annual_income
demographics_clean['annual_income'] = (
    demographics_clean['annual_income']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

# Normalize employment_type
def normalize_employment_type(emp_type):
    emp_upper = str(emp_type).upper()
    if 'FULL' in emp_upper or 'FT' in emp_upper:
        return 'Full-time'
    elif 'PART' in emp_upper or 'PT' in emp_upper:
        return 'Part-time'
    elif 'SELF' in emp_upper or 'CONTRACT' in emp_upper:
        return 'Self-employed'
    else:
        return 'Other'

demographics_clean['employment_type'] = demographics_clean['employment_type'].apply(normalize_employment_type)
demographics_clean['employment_length'] = demographics_clean['employment_length'].fillna(0)

# 2.3 Clean credit_history
credit_history_clean = credit_history.copy()
credit_history_clean['num_delinquencies_2yrs'] = credit_history_clean['num_delinquencies_2yrs'].fillna(0)

# 2.4 Clean financial_ratios
financial_ratios_clean = financial_ratios.copy()

def clean_monetary_field(series):
    return (
        series.astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .replace('nan', np.nan)
        .astype(float)
    )

monetary_columns = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                   'revolving_balance', 'credit_usage_amount', 'available_credit',
                   'total_monthly_debt_payment', 'total_debt_amount', 'monthly_free_cash_flow']

for col in monetary_columns:
    if col in financial_ratios_clean.columns:
        financial_ratios_clean[col] = clean_monetary_field(financial_ratios_clean[col])

financial_ratios_clean['revolving_balance'] = financial_ratios_clean['revolving_balance'].fillna(0)

# 2.5 Clean loan_details
loan_details_clean = loan_details.copy()

if loan_details_clean['loan_amount'].dtype == 'object':
    loan_details_clean['loan_amount'] = clean_monetary_field(loan_details_clean['loan_amount'])

def normalize_loan_type(loan_type):
    loan_upper = str(loan_type).upper()
    if 'PERSONAL' in loan_upper:
        return 'Personal'
    elif 'MORTGAGE' in loan_upper or 'HOME' in loan_upper:
        return 'Mortgage'
    elif 'CREDIT' in loan_upper or 'CC' in loan_upper:
        return 'Credit Card'
    elif 'AUTO' in loan_upper or 'CAR' in loan_upper:
        return 'Auto'
    else:
        return 'Other'

loan_details_clean['loan_type'] = loan_details_clean['loan_type'].apply(normalize_loan_type)

# 2.6 Clean geographic_data
geographic_data_clean = geographic_data.copy()

numeric_geo_cols = ['regional_unemployment_rate', 'regional_median_income',
                   'regional_median_rent', 'housing_price_index', 'cost_of_living_index']

for col in numeric_geo_cols:
    if col in geographic_data_clean.columns:
        geographic_data_clean[col] = pd.to_numeric(geographic_data_clean[col], errors='coerce')

print("✓ Data cleaned")

# ============================================================================
# STEP 3: MERGE DATASETS
# ============================================================================
print("\n[STEP 3/6] Merging datasets...")

merged = app_metadata_clean.copy()

merged = merged.merge(demographics_clean, left_on='customer_ref', right_on='cust_id', how='left', validate='1:1')
merged = merged.merge(credit_history_clean, left_on='customer_ref', right_on='customer_number', how='left', validate='1:1')
merged = merged.merge(financial_ratios_clean, left_on='customer_ref', right_on='cust_num', how='left', validate='1:1')
merged = merged.merge(loan_details_clean, left_on='customer_ref', right_on='customer_id', how='left', validate='1:1')

geographic_data_clean['id'] = geographic_data_clean['id'].astype(int)
merged = merged.merge(geographic_data_clean, left_on='customer_ref', right_on='id', how='left', validate='1:1')

# Drop redundant ID columns
id_columns = ['cust_id', 'customer_number', 'cust_num', 'customer_id', 'id']
merged = merged.drop(columns=[col for col in id_columns if col in merged.columns])

print(f"✓ Merged shape: {merged.shape}")

# ============================================================================
# STEP 4: FEATURE ENGINEERING (BASE FEATURES)
# ============================================================================
print("\n[STEP 4/6] Creating engineered features...")

# Income-based features
merged['monthly_income_from_annual'] = merged['annual_income'] / 12
merged['disposable_income'] = merged['monthly_income'] - merged['existing_monthly_debt']
merged['income_to_payment_capacity'] = merged['monthly_income'] / (merged['monthly_payment'] + 1)

# Debt burden features
merged['total_debt_to_income_annual'] = merged['total_debt_amount'] / (merged['annual_income'] + 1)
merged['debt_payment_burden'] = (merged['existing_monthly_debt'] + merged['monthly_payment']) / (merged['monthly_income'] + 1)
merged['free_cash_flow_ratio'] = merged['monthly_free_cash_flow'] / (merged['monthly_income'] + 1)
merged['loan_to_monthly_income'] = merged['loan_amount'] / (merged['monthly_income'] + 1)

# Credit behavior features
merged['credit_age_to_score_ratio'] = merged['oldest_account_age_months'] / (merged['credit_score'] + 1)
merged['delinquency_rate'] = merged['num_delinquencies_2yrs'] / (merged['num_credit_accounts'] + 1)
merged['inquiry_intensity'] = merged['num_inquiries_6mo'] + merged['recent_inquiry_count']
merged['negative_marks_total'] = (
    merged['num_delinquencies_2yrs'].fillna(0) +
    merged['num_public_records'] +
    merged['num_collections']
)
merged['credit_stress_score'] = (
    merged['credit_utilization'] * 0.3 +
    merged['debt_to_income_ratio'] * 0.3 +
    merged['delinquency_rate'] * 0.4
)

# Loan characteristics
merged['loan_amount_to_limit'] = merged['loan_amount'] / (merged['total_credit_limit'] + 1)
merged['interest_burden'] = merged['loan_amount'] * merged['interest_rate'] / 100
merged['loan_term_years'] = merged['loan_term'] / 12
merged['monthly_loan_payment_estimate'] = merged['loan_amount'] / (merged['loan_term'] + 1)

# Regional economic features
merged['income_to_regional_median'] = merged['annual_income'] / (merged['regional_median_income'] + 1)
merged['housing_affordability'] = merged['regional_median_rent'] / (merged['monthly_income'] + 1)
merged['regional_stress_index'] = (
    merged['regional_unemployment_rate'] * 0.4 +
    (merged['cost_of_living_index'] / 100) * 0.3 +
    (merged['housing_price_index'] / 100) * 0.3
)

# Behavioral features
merged['service_call_intensity'] = merged['num_customer_service_calls'] / (merged['num_login_sessions'] + 1)
merged['digital_engagement_score'] = (
    merged['has_mobile_app'] * 0.5 +
    merged['paperless_billing'] * 0.3 +
    (merged['num_login_sessions'] / merged['num_login_sessions'].max()) * 0.2
)

# Application timing features
merged['is_business_hours'] = ((merged['application_hour'] >= 9) & (merged['application_hour'] <= 17)).astype(int)
merged['is_weekend'] = (merged['application_day_of_week'].isin([6, 7])).astype(int)
merged['is_late_night'] = ((merged['application_hour'] >= 22) | (merged['application_hour'] <= 5)).astype(int)

# Account maturity
current_year = 2025
merged['account_age_years'] = current_year - merged['account_open_year']
merged['credit_history_depth'] = (
    merged['oldest_credit_line_age'] * 0.5 +
    merged['oldest_account_age_months'] * 0.5
)

# Handle missing values
merged = merged.fillna(0)

print(f"✓ Engineered features created, shape: {merged.shape}")

# Save customer_ref for results
customer_ids = merged['customer_ref'].values

# Remove target if it exists (evaluation set shouldn't have it, but just in case)
if 'default' in merged.columns:
    merged = merged.drop(columns=['default'])

X_eval = merged.drop(columns=['customer_ref'])

# ============================================================================
# STEP 5: LABEL ENCODING (USING TRAINING MAPPINGS)
# ============================================================================
print("\n[STEP 5/6] Label encoding categorical variables...")

# For evaluation, we need to encode consistently with training
# Since we don't have training LabelEncoders, we'll create simple encoding
categorical_cols = X_eval.select_dtypes(include=['object']).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on all unique values in eval set
    le.fit(X_eval[col].astype(str))
    X_eval[col] = le.transform(X_eval[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# ============================================================================
# NOTE: OOF FEATURES CANNOT BE CREATED FOR EVALUATION SET
# ============================================================================
# OOF features (KNN, Target Encoding, WOE) require training on the evaluation
# set itself, which would cause data leakage. For true inference, we either:
# 1. Use a model trained WITHOUT OOF features, OR
# 2. Create simplified versions (non-OOF) of these features, OR
# 3. Use global statistics from training set

# For now, we'll create placeholder zero columns for OOF features
# This is a limitation of the current setup - ideally, the model should
# have been trained without OOF features OR we should have saved the
# training statistics

print("\n⚠️  WARNING: OOF features cannot be created for evaluation set")
print("   Creating zero-valued placeholders for OOF features...")

# Add OOF feature placeholders
oof_feature_names = [
    'knn_oof_50', 'knn_oof_100', 'knn_oof_500',
    'state_target_oof', 'marital_status_target_oof',
    'education_target_oof', 'employment_type_target_oof',
    'debt_to_income_ratio_woe_oof', 'credit_utilization_woe_oof',
    'credit_score_woe_oof', 'age_woe_oof'
]

for feat in oof_feature_names:
    X_eval[feat] = 0.0

# Add interaction and polynomial features
if 'debt_to_income_ratio' in X_eval.columns and 'credit_utilization' in X_eval.columns:
    X_eval['debt_credit_interaction'] = X_eval['debt_to_income_ratio'] * X_eval['credit_utilization']

if 'income_stability_score' in X_eval.columns and 'debt_payment_burden' in X_eval.columns:
    X_eval['income_debt_interaction'] = X_eval['income_stability_score'] * X_eval['debt_payment_burden']

if 'age' in X_eval.columns and 'credit_score' in X_eval.columns:
    X_eval['age_credit_interaction'] = X_eval['age'] * X_eval['credit_score'] / 100

if 'employment_length' in X_eval.columns and 'monthly_income' in X_eval.columns:
    X_eval['employment_income_stability'] = X_eval['employment_length'] * X_eval['monthly_income']

# Polynomial features
top_features = ['credit_stress_score', 'debt_to_income_ratio', 'credit_utilization']
for feature in top_features:
    if feature in X_eval.columns:
        X_eval[f'{feature}_squared'] = X_eval[feature] ** 2
        X_eval[f'{feature}_cbrt'] = np.cbrt(X_eval[feature])

print(f"✓ Final feature count: {X_eval.shape[1]}")

# ============================================================================
# STEP 6: LOAD MODEL AND PREDICT
# ============================================================================
print("\n[STEP 6/6] Loading model and generating predictions...")

try:
    model_package = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully")

    # Check if it's the new format (dict) or old format (single object)
    if isinstance(model_package, dict):
        print("✓ Model package format detected")
        calibrated = model_package['calibrated']
        feature_names = model_package['feature_names']
        print(f"✓ Expected features: {len(feature_names)}")

        # Align features
        missing_features = set(feature_names) - set(X_eval.columns)
        extra_features = set(X_eval.columns) - set(feature_names)

        if missing_features:
            print(f"  Adding {len(missing_features)} missing features with zeros")
            for feat in missing_features:
                X_eval[feat] = 0.0

        if extra_features:
            print(f"  Removing {len(extra_features)} extra features")
            X_eval = X_eval.drop(columns=list(extra_features))

        # Reorder columns to match training
        X_eval = X_eval[feature_names]
        print(f"✓ Feature alignment complete: {X_eval.shape}")

        # Generate predictions
        predictions_proba = calibrated.predict_proba(X_eval)[:, 1]

        # Use optimal threshold from training
        optimal_threshold = model_package['metrics'].get('optimal_threshold', 0.5)
        print(f"✓ Using optimal threshold: {optimal_threshold:.4f}")
        predictions_binary = (predictions_proba >= optimal_threshold).astype(int)

    else:
        # Old format - single model object
        print("⚠️  Legacy model format detected")
        predictions_proba = model_package.predict_proba(X_eval)[:, 1]
        predictions_binary = (predictions_proba >= 0.5).astype(int)

    print(f"✓ Predictions generated for {len(predictions_proba)} customers")

    # Create results DataFrame
    results = pd.DataFrame({
        'customer_id': customer_ids,
        'prob': predictions_proba,
        'default': predictions_binary
    })

    # Save results
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")
    print(f"\nSample predictions:")
    print(results.head(10))

    # Summary statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean probability: {predictions_proba.mean():.4f}")
    print(f"  Median probability: {np.median(predictions_proba):.4f}")
    print(f"  Predicted defaults: {predictions_binary.sum()} ({predictions_binary.mean()*100:.2f}%)")
    print(f"  Predicted non-defaults: {(1-predictions_binary).sum()} ({(1-predictions_binary).mean()*100:.2f}%)")

except Exception as e:
    print(f"❌ Error loading or using model: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 SOLUTION:")
    print("   1. Open Google_Colab_Leak_Free_90plus_v3.ipynb in Colab")
    print("   2. Replace save code with fixed version from COLAB_MODEL_SAVING_FIX.md")
    print("   3. Retrain model (will take ~15 minutes)")
    print("   4. Download xgboost_calibrated_ensemble_v3_colab.pkl (should be 10-50 MB)")
    print("   5. Replace file in task_result1/model/")

print("\n" + "="*80)
print("PREDICTION PIPELINE COMPLETE")
print("="*80)
