#!/usr/bin/env python3
"""
EXACT COPY prediction pipeline - полностью соответствует обучению
"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import joblib
import warnings
import gc
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
np.random.seed(42)

# Paths
EVAL_DIR = Path("task_result1") / "evaluation_set-20251116T050317Z-1-001" / "evaluation_set"
MODEL_PATH = Path("task_result1") / "model" / "xgboost_calibrated_ensemble_v3.pkl"
OUTPUT_PATH = EVAL_DIR / "results.csv"

print("="*80)
print("PREDICTION WITH EXACT DATA CLEANING (matches training)")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

app = pd.read_csv(EVAL_DIR / "application_metadata.csv")
demo = pd.read_csv(EVAL_DIR / "demographics.csv")
credit = pd.read_parquet(EVAL_DIR / "credit_history.parquet")

with open(EVAL_DIR / "financial_ratios.jsonl", 'r') as f:
    fin = pd.DataFrame([json.loads(line) for line in f])

loan = pd.read_excel(EVAL_DIR / "loan_details.xlsx")

tree = ET.parse(EVAL_DIR / "geographic_data.xml")
geo = pd.DataFrame([{child.tag: child.text for child in customer}
                     for customer in tree.getroot().findall('customer')])

print(f"✓ Loaded all 6 files")

# ============================================================================
# CLEAN DATA - EXACT COPY FROM Data_Cleaning_and_Merging_Colab.ipynb
# ============================================================================
print("\n[2/6] Cleaning data (EXACT match to training)...")

# 1. application_metadata
app_clean = app.drop(columns=['random_noise_1'])

# 2. demographics - EXACT normalization
demo_clean = demo.copy()
demo_clean['annual_income'] = (
    demo_clean['annual_income']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

# EXACT function from training
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

demo_clean['employment_type'] = demo_clean['employment_type'].apply(normalize_employment_type)
demo_clean['employment_length'] = demo_clean['employment_length'].fillna(0)

# 3. credit_history
credit_clean = credit.copy()
credit_clean['num_delinquencies_2yrs'] = credit_clean['num_delinquencies_2yrs'].fillna(0)

# 4. financial_ratios
fin_clean = fin.copy()

# EXACT function from training
def clean_monetary_field(series):
    return (
        series.astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .replace('nan', np.nan)
        .astype(float)
    )

monetary_cols = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                 'revolving_balance', 'credit_usage_amount', 'available_credit',
                 'total_monthly_debt_payment', 'total_debt_amount', 'monthly_free_cash_flow']

for col in monetary_cols:
    if col in fin_clean.columns:
        fin_clean[col] = clean_monetary_field(fin_clean[col])

fin_clean['revolving_balance'] = fin_clean['revolving_balance'].fillna(0)

# 5. loan_details
loan_clean = loan.copy()

if loan_clean['loan_amount'].dtype == 'object':
    loan_clean['loan_amount'] = clean_monetary_field(loan_clean['loan_amount'])

# EXACT function from training
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

loan_clean['loan_type'] = loan_clean['loan_type'].apply(normalize_loan_type)

# 6. geographic_data
geo_clean = geo.copy()
numeric_geo_cols = ['regional_unemployment_rate', 'regional_median_income',
                   'regional_median_rent', 'housing_price_index', 'cost_of_living_index']

for col in numeric_geo_cols:
    if col in geo_clean.columns:
        geo_clean[col] = pd.to_numeric(geo_clean[col], errors='coerce')

print("✓ Data cleaned")

# ============================================================================
# MERGE - EXACT COPY FROM TRAINING
# ============================================================================
print("\n[3/6] Merging datasets...")

merged = app_clean.copy()
merged = merged.merge(demo_clean, left_on='customer_ref', right_on='cust_id', how='left', validate='1:1')
merged = merged.merge(credit_clean, left_on='customer_ref', right_on='customer_number', how='left', validate='1:1')
merged = merged.merge(fin_clean, left_on='customer_ref', right_on='cust_num', how='left', validate='1:1')
merged = merged.merge(loan_clean, left_on='customer_ref', right_on='customer_id', how='left', validate='1:1')

geo_clean['id'] = geo_clean['id'].astype(int)
merged = merged.merge(geo_clean, left_on='customer_ref', right_on='id', how='left', validate='1:1')

# Drop redundant IDs
id_cols = ['cust_id', 'customer_number', 'cust_num', 'customer_id', 'id']
merged = merged.drop(columns=[col for col in id_cols if col in merged.columns])

print(f"✓ Merged: {merged.shape}")

# Clean up memory
del app_clean, demo_clean, credit_clean, fin_clean, loan_clean, geo_clean
gc.collect()

# ============================================================================
# FEATURE ENGINEERING - EXACT COPY FROM TRAINING
# ============================================================================
print("\n[4/6] Feature engineering (EXACT match to training)...")

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
merged['negative_marks_total'] = (merged['num_delinquencies_2yrs'].fillna(0) + merged['num_public_records'] + merged['num_collections'])
merged['credit_stress_score'] = (merged['credit_utilization'] * 0.3 + merged['debt_to_income_ratio'] * 0.3 + merged['delinquency_rate'] * 0.4)

# Loan characteristics
merged['loan_amount_to_limit'] = merged['loan_amount'] / (merged['total_credit_limit'] + 1)
merged['interest_burden'] = merged['loan_amount'] * merged['interest_rate'] / 100
merged['loan_term_years'] = merged['loan_term'] / 12
merged['monthly_loan_payment_estimate'] = merged['loan_amount'] / (merged['loan_term'] + 1)

# Regional economic features
merged['income_to_regional_median'] = merged['annual_income'] / (merged['regional_median_income'] + 1)
merged['housing_affordability'] = merged['regional_median_rent'] / (merged['monthly_income'] + 1)
merged['regional_stress_index'] = (merged['regional_unemployment_rate'] * 0.4 + (merged['cost_of_living_index'] / 100) * 0.3 + (merged['housing_price_index'] / 100) * 0.3)

# Behavioral features
merged['service_call_intensity'] = merged['num_customer_service_calls'] / (merged['num_login_sessions'] + 1)
merged['digital_engagement_score'] = (merged['has_mobile_app'] * 0.5 + merged['paperless_billing'] * 0.3 + (merged['num_login_sessions'] / merged['num_login_sessions'].max()) * 0.2)

# Application timing features
merged['is_business_hours'] = ((merged['application_hour'] >= 9) & (merged['application_hour'] <= 17)).astype(int)
merged['is_weekend'] = (merged['application_day_of_week'].isin([6, 7])).astype(int)
merged['is_late_night'] = ((merged['application_hour'] >= 22) | (merged['application_hour'] <= 5)).astype(int)

# Account maturity
current_year = 2025
merged['account_age_years'] = current_year - merged['account_open_year']
merged['credit_history_depth'] = (merged['oldest_credit_line_age'] * 0.5 + merged['oldest_account_age_months'] * 0.5)

# Fill missing values
merged = merged.fillna(0)

print(f"✓ Features created: {merged.shape}")

# Save customer IDs
customer_ids = merged['customer_ref'].values
merged = merged.drop(columns=['customer_ref'])

# ============================================================================
# LABEL ENCODING - Similar to training
# ============================================================================
print("\n[5/6] Label encoding...")

categorical_cols = merged.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    merged[col] = le.fit_transform(merged[col].astype(str))

# Add OOF placeholders (these cannot be calculated for evaluation set)
oof_features = ['knn_oof_50', 'knn_oof_100', 'knn_oof_500', 'state_target_oof',
                'marital_status_target_oof', 'education_target_oof', 'employment_type_target_oof',
                'debt_to_income_ratio_woe_oof', 'credit_utilization_woe_oof', 'credit_score_woe_oof', 'age_woe_oof']

for feat in oof_features:
    merged[feat] = 0.0

# Add interaction features
if 'debt_to_income_ratio' in merged.columns and 'credit_utilization' in merged.columns:
    merged['debt_credit_interaction'] = merged['debt_to_income_ratio'] * merged['credit_utilization']

if 'income_stability_score' in merged.columns and 'debt_payment_burden' in merged.columns:
    merged['income_debt_interaction'] = merged['income_stability_score'] * merged['debt_payment_burden']

if 'age' in merged.columns and 'credit_score' in merged.columns:
    merged['age_credit_interaction'] = merged['age'] * merged['credit_score'] / 100

if 'employment_length' in merged.columns and 'monthly_income' in merged.columns:
    merged['employment_income_stability'] = merged['employment_length'] * merged['monthly_income']

# Polynomial features
for feat in ['credit_stress_score', 'debt_to_income_ratio', 'credit_utilization']:
    if feat in merged.columns:
        merged[f'{feat}_squared'] = merged[feat] ** 2
        merged[f'{feat}_cbrt'] = np.cbrt(merged[feat])

print(f"✓ Final shape: {merged.shape}")

# ============================================================================
# PREDICT
# ============================================================================
print("\n[6/6] Predicting...")

model_pkg = joblib.load(MODEL_PATH)
models = model_pkg['models']
weights = model_pkg['weights']
feature_names = model_pkg['feature_names']
optimal_threshold = model_pkg['metrics']['optimal_threshold']

print(f"✓ Model loaded: {len(models)} models, threshold={optimal_threshold:.4f}")

# Align features
missing = set(feature_names) - set(merged.columns)
if missing:
    print(f"  Adding {len(missing)} missing features")
    for f in missing:
        merged[f] = 0.0

extra = set(merged.columns) - set(feature_names)
if extra:
    print(f"  Removing {len(extra)} extra features")
    merged = merged.drop(columns=list(extra))

merged = merged[feature_names]
print(f"✓ Aligned: {merged.shape}")

# Predict with ensemble
probs = np.zeros(len(merged))
for i, (model, weight) in enumerate(zip(models, weights)):
    model_probs = model.predict_proba(merged)[:, 1]
    probs += model_probs * weight

preds = (probs >= optimal_threshold).astype(int)

# Save results
results = pd.DataFrame({'customer_id': customer_ids, 'prob': probs, 'default': preds})
results.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Results saved: {OUTPUT_PATH}")
print(f"\nSample:")
print(results.head(10))
print(f"\nStatistics:")
print(f"  Mean prob: {probs.mean():.4f}")
print(f"  Default predictions: {preds.sum()} ({preds.mean()*100:.2f}%)")
print("\n" + "="*80)
print("COMPLETE")
print("="*80)
