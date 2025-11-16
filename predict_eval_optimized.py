#!/usr/bin/env python3
"""Optimized prediction pipeline for evaluation set"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import joblib
import warnings
import gc
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# Paths
EVAL_DIR = Path("./evaluation_set")
MODEL_PATH = Path("model/xgboost_calibrated_ensemble_v3.pkl")
OUTPUT_PATH = EVAL_DIR / "results.csv"

print("="*80)
print("CREDIT DEFAULT PREDICTION - EVALUATION SET (OPTIMIZED)")
print("="*80)

# ============================================================================
# LOAD & CLEAN DATA
# ============================================================================
print("\n[1/4] Loading and cleaning data...")

# 1. Load application_metadata
app = pd.read_csv(EVAL_DIR / "application_metadata.csv").drop(columns=['random_noise_1'])

# 2. Load demographics
demo = pd.read_csv(EVAL_DIR / "demographics.csv")
demo['annual_income'] = demo['annual_income'].str.replace('$', '').str.replace(',', '').astype(float)
demo['employment_type'] = demo['employment_type'].str.upper().str.replace(' ', '-')
demo['employment_length'] = demo['employment_length'].fillna(0)

# 3. Load credit_history
credit = pd.read_parquet(EVAL_DIR / "credit_history.parquet")
credit['num_delinquencies_2yrs'] = credit['num_delinquencies_2yrs'].fillna(0)

# 4. Load financial_ratios
with open(EVAL_DIR / "financial_ratios.jsonl", 'r') as f:
    fin = pd.DataFrame([json.loads(line) for line in f])

for col in ['monthly_income', 'existing_monthly_debt', 'monthly_payment', 'revolving_balance',
            'credit_usage_amount', 'available_credit', 'total_monthly_debt_payment',
            'total_debt_amount', 'monthly_free_cash_flow']:
    if col in fin.columns:
        fin[col] = fin[col].astype(str).str.replace('$', '').str.replace(',', '').replace('nan', np.nan).astype(float)
fin['revolving_balance'] = fin['revolving_balance'].fillna(0)

# 5. Load loan_details
loan = pd.read_excel(EVAL_DIR / "loan_details.xlsx")
if loan['loan_amount'].dtype == 'object':
    loan['loan_amount'] = loan['loan_amount'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

# 6. Load geographic_data
tree = ET.parse(EVAL_DIR / "geographic_data.xml")
geo = pd.DataFrame([{child.tag: child.text for child in customer} for customer in tree.getroot().findall('customer')])
for col in ['regional_unemployment_rate', 'regional_median_income', 'regional_median_rent',
            'housing_price_index', 'cost_of_living_index']:
    if col in geo.columns:
        geo[col] = pd.to_numeric(geo[col], errors='coerce')

print(f"✓ Loaded: app={app.shape}, demo={demo.shape}, credit={credit.shape}")
print(f"          fin={fin.shape}, loan={loan.shape}, geo={geo.shape}")

# ============================================================================
# MERGE
# ============================================================================
print("\n[2/4] Merging datasets...")

df = app.merge(demo, left_on='customer_ref', right_on='cust_id', how='left', validate='1:1')
df = df.merge(credit, left_on='customer_ref', right_on='customer_number', how='left', validate='1:1')
df = df.merge(fin, left_on='customer_ref', right_on='cust_num', how='left', validate='1:1')
df = df.merge(loan, left_on='customer_ref', right_on='customer_id', how='left', validate='1:1')
geo['id'] = geo['id'].astype(int)
df = df.merge(geo, left_on='customer_ref', right_on='id', how='left', validate='1:1')

# Clean up
df = df.drop(columns=['cust_id', 'customer_number', 'cust_num', 'customer_id', 'id'])
del app, demo, credit, fin, loan, geo
gc.collect()

print(f"✓ Merged: {df.shape}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n[3/4] Engineering features...")

# Basic features
df['monthly_income_from_annual'] = df['annual_income'] / 12
df['disposable_income'] = df['monthly_income'] - df['existing_monthly_debt']
df['income_to_payment_capacity'] = df['monthly_income'] / (df['monthly_payment'] + 1)
df['total_debt_to_income_annual'] = df['total_debt_amount'] / (df['annual_income'] + 1)
df['debt_payment_burden'] = (df['existing_monthly_debt'] + df['monthly_payment']) / (df['monthly_income'] + 1)
df['free_cash_flow_ratio'] = df['monthly_free_cash_flow'] / (df['monthly_income'] + 1)
df['loan_to_monthly_income'] = df['loan_amount'] / (df['monthly_income'] + 1)
df['credit_age_to_score_ratio'] = df['oldest_account_age_months'] / (df['credit_score'] + 1)
df['delinquency_rate'] = df['num_delinquencies_2yrs'] / (df['num_credit_accounts'] + 1)
df['inquiry_intensity'] = df['num_inquiries_6mo'] + df['recent_inquiry_count']
df['negative_marks_total'] = df['num_delinquencies_2yrs'].fillna(0) + df['num_public_records'] + df['num_collections']
df['credit_stress_score'] = df['credit_utilization'] * 0.3 + df['debt_to_income_ratio'] * 0.3 + df['delinquency_rate'] * 0.4
df['loan_amount_to_limit'] = df['loan_amount'] / (df['total_credit_limit'] + 1)
df['interest_burden'] = df['loan_amount'] * df['interest_rate'] / 100
df['loan_term_years'] = df['loan_term'] / 12
df['monthly_loan_payment_estimate'] = df['loan_amount'] / (df['loan_term'] + 1)
df['income_to_regional_median'] = df['annual_income'] / (df['regional_median_income'] + 1)
df['housing_affordability'] = df['regional_median_rent'] / (df['monthly_income'] + 1)
df['regional_stress_index'] = df['regional_unemployment_rate'] * 0.4 + (df['cost_of_living_index'] / 100) * 0.3 + (df['housing_price_index'] / 100) * 0.3
df['service_call_intensity'] = df['num_customer_service_calls'] / (df['num_login_sessions'] + 1)
df['digital_engagement_score'] = df['has_mobile_app'] * 0.5 + df['paperless_billing'] * 0.3 + (df['num_login_sessions'] / df['num_login_sessions'].max()) * 0.2
df['is_business_hours'] = ((df['application_hour'] >= 9) & (df['application_hour'] <= 17)).astype(int)
df['is_weekend'] = df['application_day_of_week'].isin([6, 7]).astype(int)
df['is_late_night'] = ((df['application_hour'] >= 22) | (df['application_hour'] <= 5)).astype(int)
df['account_age_years'] = 2025 - df['account_open_year']
df['credit_history_depth'] = df['oldest_credit_line_age'] * 0.5 + df['oldest_account_age_months'] * 0.5

# Interaction features
df['debt_credit_interaction'] = df['debt_to_income_ratio'] * df['credit_utilization']
if 'income_stability_score' in df.columns and 'debt_payment_burden' in df.columns:
    df['income_debt_interaction'] = df['income_stability_score'] * df['debt_payment_burden']
df['age_credit_interaction'] = df['age'] * df['credit_score'] / 100
df['employment_income_stability'] = df['employment_length'] * df['monthly_income']

# Polynomial features
for feat in ['credit_stress_score', 'debt_to_income_ratio', 'credit_utilization']:
    if feat in df.columns:
        df[f'{feat}_squared'] = df[feat] ** 2
        df[f'{feat}_cbrt'] = np.cbrt(df[feat])

# Fill missing
df = df.fillna(0)

print(f"✓ Features created: {df.shape}")

# Save customer IDs
customer_ids = df['customer_ref'].values
df = df.drop(columns=['customer_ref'])

# Label encode categoricals
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# OOF placeholder features
for feat in ['knn_oof_50', 'knn_oof_100', 'knn_oof_500', 'state_target_oof',
             'marital_status_target_oof', 'education_target_oof', 'employment_type_target_oof',
             'debt_to_income_ratio_woe_oof', 'credit_utilization_woe_oof', 'credit_score_woe_oof', 'age_woe_oof']:
    df[feat] = 0.0

print(f"✓ Final shape: {df.shape}")

# ============================================================================
# PREDICT
# ============================================================================
print("\n[4/4] Loading model and predicting...")

model_pkg = joblib.load(MODEL_PATH)
feature_names = model_pkg['feature_names']
models = model_pkg['models']
weights = model_pkg['weights']
optimal_threshold = model_pkg['metrics']['optimal_threshold']

print(f"✓ Model loaded: {len(models)} models, {len(feature_names)} features, threshold={optimal_threshold:.4f}")

# Align features
missing = set(feature_names) - set(df.columns)
if missing:
    print(f"  Adding {len(missing)} missing features")
    for f in missing:
        df[f] = 0.0

extra = set(df.columns) - set(feature_names)
if extra:
    print(f"  Removing {len(extra)} extra features")
    df = df.drop(columns=list(extra))

df = df[feature_names]
print(f"✓ Aligned: {df.shape}")

# Predict using ensemble (weighted average)
print("  Generating predictions from ensemble...")
probs = np.zeros(len(df))
for i, (model, weight) in enumerate(zip(models, weights)):
    model_probs = model.predict_proba(df)[:, 1]
    probs += model_probs * weight
    print(f"    Model {i+1}: weight={weight:.4f}")

preds = (probs >= optimal_threshold).astype(int)

# Save results
results = pd.DataFrame({'customer_id': customer_ids, 'prob': probs, 'default': preds})
results.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Results saved to: {OUTPUT_PATH}")
print(f"\nSample:")
print(results.head(10))
print(f"\nStatistics:")
print(f"  Mean prob: {probs.mean():.4f}")
print(f"  Predicted defaults: {preds.sum()} ({preds.mean()*100:.2f}%)")
print("\n" + "="*80)
print("COMPLETE")
print("="*80)
