#!/usr/bin/env python3
"""
FULL PREDICTION PIPELINE WITH INFERENCE FEATURES
Точная копия обучения + inference признаки (KNN, Target Encoding, WOE)
"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import joblib
import warnings
import gc
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# Paths
EVAL_DIR = Path("evaluation_set")
MODEL_DIR = Path("model")
OUTPUT_PATH = EVAL_DIR / "results.csv"

print("="*80)
print("PREDICTION WITH FULL INFERENCE FEATURES")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND CLEAN EVALUATION DATA (exact copy from training)
# ============================================================================
print("\n[STEP 1/5] Loading and cleaning evaluation data...")

# Load 6 files
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

# Clean data - EXACT COPY
app_clean = app.drop(columns=['random_noise_1'])

demo_clean = demo.copy()
demo_clean['annual_income'] = (demo_clean['annual_income']
    .str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float))

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

credit_clean = credit.copy()
credit_clean['num_delinquencies_2yrs'] = credit_clean['num_delinquencies_2yrs'].fillna(0)

fin_clean = fin.copy()

def clean_monetary_field(series):
    return (series.astype(str).str.replace('$', '', regex=False)
           .str.replace(',', '', regex=False).replace('nan', np.nan).astype(float))

monetary_cols = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                 'revolving_balance', 'credit_usage_amount', 'available_credit',
                 'total_monthly_debt_payment', 'total_debt_amount', 'monthly_free_cash_flow']

for col in monetary_cols:
    if col in fin_clean.columns:
        fin_clean[col] = clean_monetary_field(fin_clean[col])

fin_clean['revolving_balance'] = fin_clean['revolving_balance'].fillna(0)

loan_clean = loan.copy()
if loan_clean['loan_amount'].dtype == 'object':
    loan_clean['loan_amount'] = clean_monetary_field(loan_clean['loan_amount'])

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

geo_clean = geo.copy()
for col in ['regional_unemployment_rate', 'regional_median_income', 'regional_median_rent',
            'housing_price_index', 'cost_of_living_index']:
    if col in geo_clean.columns:
        geo_clean[col] = pd.to_numeric(geo_clean[col], errors='coerce')

print(f"✓ Data cleaned")

# Merge
merged = app_clean.copy()
merged = merged.merge(demo_clean, left_on='customer_ref', right_on='cust_id', how='left')
merged = merged.merge(credit_clean, left_on='customer_ref', right_on='customer_number', how='left')
merged = merged.merge(fin_clean, left_on='customer_ref', right_on='cust_num', how='left')
merged = merged.merge(loan_clean, left_on='customer_ref', right_on='customer_id', how='left')
geo_clean['id'] = geo_clean['id'].astype(int)
merged = merged.merge(geo_clean, left_on='customer_ref', right_on='id', how='left')

merged = merged.drop(columns=['cust_id', 'customer_number', 'cust_num', 'customer_id', 'id'])

print(f"✓ Merged: {merged.shape}")

# Feature engineering - EXACT COPY
merged['monthly_income_from_annual'] = merged['annual_income'] / 12
merged['disposable_income'] = merged['monthly_income'] - merged['existing_monthly_debt']
merged['income_to_payment_capacity'] = merged['monthly_income'] / (merged['monthly_payment'] + 1)
merged['total_debt_to_income_annual'] = merged['total_debt_amount'] / (merged['annual_income'] + 1)
merged['debt_payment_burden'] = (merged['existing_monthly_debt'] + merged['monthly_payment']) / (merged['monthly_income'] + 1)
merged['free_cash_flow_ratio'] = merged['monthly_free_cash_flow'] / (merged['monthly_income'] + 1)
merged['loan_to_monthly_income'] = merged['loan_amount'] / (merged['monthly_income'] + 1)
merged['credit_age_to_score_ratio'] = merged['oldest_account_age_months'] / (merged['credit_score'] + 1)
merged['delinquency_rate'] = merged['num_delinquencies_2yrs'] / (merged['num_credit_accounts'] + 1)
merged['inquiry_intensity'] = merged['num_inquiries_6mo'] + merged['recent_inquiry_count']
merged['negative_marks_total'] = (merged['num_delinquencies_2yrs'].fillna(0) + merged['num_public_records'] + merged['num_collections'])
merged['credit_stress_score'] = (merged['credit_utilization'] * 0.3 + merged['debt_to_income_ratio'] * 0.3 + merged['delinquency_rate'] * 0.4)
merged['loan_amount_to_limit'] = merged['loan_amount'] / (merged['total_credit_limit'] + 1)
merged['interest_burden'] = merged['loan_amount'] * merged['interest_rate'] / 100
merged['loan_term_years'] = merged['loan_term'] / 12
merged['monthly_loan_payment_estimate'] = merged['loan_amount'] / (merged['loan_term'] + 1)
merged['income_to_regional_median'] = merged['annual_income'] / (merged['regional_median_income'] + 1)
merged['housing_affordability'] = merged['regional_median_rent'] / (merged['monthly_income'] + 1)
merged['regional_stress_index'] = (merged['regional_unemployment_rate'] * 0.4 + (merged['cost_of_living_index'] / 100) * 0.3 + (merged['housing_price_index'] / 100) * 0.3)
merged['service_call_intensity'] = merged['num_customer_service_calls'] / (merged['num_login_sessions'] + 1)
merged['digital_engagement_score'] = (merged['has_mobile_app'] * 0.5 + merged['paperless_billing'] * 0.3 + (merged['num_login_sessions'] / merged['num_login_sessions'].max()) * 0.2)
merged['is_business_hours'] = ((merged['application_hour'] >= 9) & (merged['application_hour'] <= 17)).astype(int)
merged['is_weekend'] = (merged['application_day_of_week'].isin([6, 7])).astype(int)
merged['is_late_night'] = ((merged['application_hour'] >= 22) | (merged['application_hour'] <= 5)).astype(int)
merged['account_age_years'] = 2025 - merged['account_open_year']
merged['credit_history_depth'] = (merged['oldest_credit_line_age'] * 0.5 + merged['oldest_account_age_months'] * 0.5)

merged = merged.fillna(0)

customer_ids = merged['customer_ref'].values
X_eval = merged.drop(columns=['customer_ref'])

# Label encoding
for col in X_eval.select_dtypes(include=['object']).columns:
    X_eval[col] = LabelEncoder().fit_transform(X_eval[col].astype(str))

# Interaction and polynomial features
if 'debt_to_income_ratio' in X_eval.columns and 'credit_utilization' in X_eval.columns:
    X_eval['debt_credit_interaction'] = X_eval['debt_to_income_ratio'] * X_eval['credit_utilization']
if 'age' in X_eval.columns and 'credit_score' in X_eval.columns:
    X_eval['age_credit_interaction'] = X_eval['age'] * X_eval['credit_score'] / 100

for feat in ['credit_stress_score', 'debt_to_income_ratio', 'credit_utilization']:
    if feat in X_eval.columns:
        X_eval[f'{feat}_squared'] = X_eval[feat] ** 2
        X_eval[f'{feat}_cbrt'] = np.cbrt(X_eval[feat])

print(f"✓ Base features created: {X_eval.shape}")

# ============================================================================
# STEP 2: LOAD TRAINING DATA
# ============================================================================
print("\n[STEP 2/5] Loading training data for inference features...")

X_train = pd.read_parquet('X_train_leak_free_v3.parquet')
y_train = pd.read_parquet('y_train_leak_free_v3.parquet').values.ravel()

print(f"✓ Training: {X_train.shape}, default rate: {y_train.mean():.2%}")

# ============================================================================
# STEP 3: CREATE INFERENCE FEATURES
# ============================================================================
print("\n[STEP 3/5] Creating inference features...")

# KNN features
print("  Creating KNN features...")
def create_knn_inference(X_train, y_train, X_eval, n_neighbors):
    # Use только общие числовые колонки
    train_numeric = [c for c in X_train.select_dtypes(include=[np.number]).columns
                    if 'oof' not in c.lower()]
    eval_numeric = [c for c in X_eval.select_dtypes(include=[np.number]).columns
                   if 'oof' not in c.lower()]
    common_cols = list(set(train_numeric) & set(eval_numeric))
    common_cols.sort()  # Для консистентности

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[common_cols].fillna(0))
    X_eval_scaled = scaler.transform(X_eval[common_cols].fillna(0))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    return knn.predict_proba(X_eval_scaled)[:, 1]

for k in [50, 100, 500]:
    X_eval[f'knn_oof_{k}'] = create_knn_inference(X_train, y_train, X_eval, k)
    print(f"    knn_oof_{k}: ✓")

# Target encoding
print("  Creating Target Encoding features...")
def target_encode_inference(X_train, y_train, X_eval, col, smoothing=10):
    global_mean = y_train.mean()
    encoding = {}
    for val in X_train[col].unique():
        mask = X_train[col] == val
        n = mask.sum()
        if n > 0:
            cat_mean = y_train[mask].mean()
            encoding[val] = (cat_mean * n + global_mean * smoothing) / (n + smoothing)
        else:
            encoding[val] = global_mean
    return X_eval[col].map(encoding).fillna(global_mean)

for col in ['state', 'marital_status', 'education', 'employment_type']:
    if col in X_train.columns and col in X_eval.columns:
        X_eval[f'{col}_target_oof'] = target_encode_inference(X_train, y_train, X_eval, col)
        print(f"    {col}_target_oof: ✓")

# WOE features
print("  Creating WOE features...")
def woe_inference(X_train, y_train, X_eval, col, n_bins=10):
    try:
        _, bins = pd.qcut(X_train[col].fillna(0), q=n_bins, duplicates='drop', retbins=True)
        train_binned = pd.cut(X_train[col].fillna(0), bins=bins, include_lowest=True)
        woe_dict = {}
        total_good = (y_train == 0).sum()
        total_bad = y_train.sum()
        for bin_label in train_binned.cat.categories:
            mask = train_binned == bin_label
            n_good = max((y_train[mask] == 0).sum(), 0.5)
            n_bad = max(y_train[mask].sum(), 0.5)
            woe = np.log((n_good / total_good) / (n_bad / total_bad))
            woe_dict[bin_label] = woe
        eval_binned = pd.cut(X_eval[col].fillna(0), bins=bins, include_lowest=True)
        return eval_binned.map(woe_dict).fillna(0)
    except:
        return np.zeros(len(X_eval))

for col in ['debt_to_income_ratio', 'credit_utilization', 'credit_score', 'age']:
    if col in X_train.columns and col in X_eval.columns:
        X_eval[f'{col}_woe_oof'] = woe_inference(X_train, y_train, X_eval, col)
        print(f"    {col}_woe_oof: ✓")

print(f"✓ All inference features created: {X_eval.shape}")

# ============================================================================
# STEP 4: PREDICT
# ============================================================================
print("\n[STEP 4/5] Loading model and predicting...")

model_pkg = joblib.load(MODEL_DIR / 'xgboost_calibrated_ensemble_v3.pkl')
models = model_pkg['models']
weights = model_pkg['weights']
feature_names = model_pkg['feature_names']
optimal_threshold = model_pkg['metrics']['optimal_threshold']

print(f"✓ Model: {len(models)} models, {len(feature_names)} features, threshold={optimal_threshold:.4f}")

# Align features
missing = set(feature_names) - set(X_eval.columns)
if missing:
    print(f"  WARNING: Adding {len(missing)} missing features with zeros")
    for f in missing:
        X_eval[f] = 0.0

extra = set(X_eval.columns) - set(feature_names)
if extra:
    print(f"  Removing {len(extra)} extra features")
    X_eval = X_eval.drop(columns=list(extra))

X_eval = X_eval[feature_names]
print(f"✓ Aligned: {X_eval.shape}")

# Predict
probs = np.zeros(len(X_eval))
for model, weight in zip(models, weights):
    probs += model.predict_proba(X_eval)[:, 1] * weight

preds = (probs >= optimal_threshold).astype(int)

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================
print("\n[STEP 5/5] Saving results...")

results = pd.DataFrame({'customer_id': customer_ids, 'prob': probs, 'default': preds})
results.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ RESULTS SAVED: {OUTPUT_PATH}")
print(f"\nSample:")
print(results.head(10))
print(f"\nStatistics:")
print(f"  Mean probability: {probs.mean():.4f}")
print(f"  Median probability: {np.median(probs):.4f}")
print(f"  Default predictions: {preds.sum()} ({preds.mean()*100:.2f}%)")
print(f"  Non-default predictions: {(1-preds).sum()} ({(1-preds).mean()*100:.2f}%)")

print("\n" + "="*80)
print("PREDICTION COMPLETE WITH FULL INFERENCE FEATURES")
print("="*80)
