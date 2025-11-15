#!/usr/bin/env python3
"""
Credit Default Prediction - Comprehensive Data Analysis Report
Author: Data Science Specialist
Date: November 15, 2025
Purpose: Analyze all 6 data files to understand structure, quality, and merge strategy
"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Define data directory
DATA_DIR = Path('/home/dr/cbu/drive-download-20251115T045945Z-1-001')

print("=" * 100)
print("CREDIT DEFAULT PREDICTION - COMPREHENSIVE DATA ANALYSIS REPORT")
print("=" * 100)
print()

# Initialize report dictionary to store findings
report = {
    'file_summaries': {},
    'data_quality_issues': {},
    'merge_strategy': {},
    'recommendations': []
}

# ============================================================================
# 1. APPLICATION_METADATA.CSV ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("1. APPLICATION_METADATA.CSV ANALYSIS")
print("=" * 100)

try:
    app_metadata = pd.read_csv(DATA_DIR / 'application_metadata.csv')

    print("\n### Dataset Overview ###")
    print(f"Shape: {app_metadata.shape[0]} rows × {app_metadata.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(app_metadata.dtypes)

    print(f"\nFirst 5 rows:")
    print(app_metadata.head())

    print("\n### Data Quality Analysis ###")

    # Check for duplicates
    print(f"Duplicate rows: {app_metadata.duplicated().sum()}")
    print(f"Duplicate customer_ref: {app_metadata['customer_ref'].duplicated().sum()}")

    # Missing values
    print("\nMissing Values:")
    missing_df = pd.DataFrame({
        'Column': app_metadata.columns,
        'Missing_Count': app_metadata.isnull().sum(),
        'Missing_Percentage': (app_metadata.isnull().sum() / len(app_metadata) * 100).round(2)
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    if missing_df['Missing_Count'].sum() == 0:
        print("No missing values found!")

    # Target variable analysis
    print("\n### Target Variable Analysis ###")
    print("Default distribution:")
    print(app_metadata['default'].value_counts())
    print(f"Default rate: {(app_metadata['default'].mean() * 100):.2f}%")

    # Check for noise columns
    if 'random_noise_1' in app_metadata.columns:
        print("\n⚠️ NOISE COLUMN DETECTED: 'random_noise_1' should be removed during cleaning")
        print(f"Sample values from random_noise_1: {app_metadata['random_noise_1'].head().tolist()}")

    # Store summary
    report['file_summaries']['application_metadata'] = {
        'rows': app_metadata.shape[0],
        'columns': app_metadata.shape[1],
        'primary_key': 'customer_ref',
        'target_variable': 'default',
        'duplicates': app_metadata.duplicated().sum(),
        'missing_values': missing_df['Missing_Count'].sum()
    }

    report['data_quality_issues']['application_metadata'] = [
        'Contains noise column: random_noise_1'
    ]

except Exception as e:
    print(f"Error loading application_metadata.csv: {e}")

# ============================================================================
# 2. DEMOGRAPHICS.CSV ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("2. DEMOGRAPHICS.CSV ANALYSIS")
print("=" * 100)

try:
    demographics = pd.read_csv(DATA_DIR / 'demographics.csv')

    print("\n### Dataset Overview ###")
    print(f"Shape: {demographics.shape[0]} rows × {demographics.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(demographics.dtypes)

    print(f"\nFirst 5 rows:")
    print(demographics.head())

    print("\n### Data Quality Analysis ###")

    # Check for duplicates
    print(f"Duplicate rows: {demographics.duplicated().sum()}")
    print(f"Duplicate cust_id: {demographics['cust_id'].duplicated().sum()}")

    # Missing values
    print("\nMissing Values:")
    missing_df = pd.DataFrame({
        'Column': demographics.columns,
        'Missing_Count': demographics.isnull().sum(),
        'Missing_Percentage': (demographics.isnull().sum() / len(demographics) * 100).round(2)
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    if missing_df['Missing_Count'].sum() == 0:
        print("No missing values found!")

    # Check annual_income formatting issues
    print("\n### Annual Income Formatting Issues ###")
    if 'annual_income' in demographics.columns:
        # Sample values to check formatting
        sample_incomes = demographics['annual_income'].dropna().head(10)
        print("Sample annual_income values:")
        print(sample_incomes.tolist())

        # Check for different formats
        has_dollar = demographics['annual_income'].astype(str).str.contains('\\$', na=False).sum()
        has_comma = demographics['annual_income'].astype(str).str.contains(',', na=False).sum()
        print(f"\nRows with $ symbol: {has_dollar}")
        print(f"Rows with comma separator: {has_comma}")

    # Check employment_type consistency
    print("\n### Employment Type Consistency Issues ###")
    if 'employment_type' in demographics.columns:
        print("Unique employment_type values (showing first 20):")
        unique_employment = demographics['employment_type'].unique()[:20]
        print(unique_employment)
        print(f"Total unique values: {demographics['employment_type'].nunique()}")

    # Store summary
    report['file_summaries']['demographics'] = {
        'rows': demographics.shape[0],
        'columns': demographics.shape[1],
        'primary_key': 'cust_id',
        'duplicates': demographics.duplicated().sum(),
        'missing_values': missing_df['Missing_Count'].sum()
    }

    report['data_quality_issues']['demographics'] = [
        'Inconsistent annual_income formatting (mixed "$" and "," usage)',
        'Inconsistent employment_type casing (Full-time vs FULL_TIME vs Full Time)'
    ]

except Exception as e:
    print(f"Error loading demographics.csv: {e}")

# ============================================================================
# 3. CREDIT_HISTORY.PARQUET ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("3. CREDIT_HISTORY.PARQUET ANALYSIS")
print("=" * 100)

try:
    credit_history = pd.read_parquet(DATA_DIR / 'credit_history.parquet')

    print("\n### Dataset Overview ###")
    print(f"Shape: {credit_history.shape[0]} rows × {credit_history.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(credit_history.dtypes)

    print(f"\nFirst 5 rows:")
    print(credit_history.head())

    print("\n### Data Quality Analysis ###")

    # Identify potential primary key
    print("\nChecking for potential primary keys:")
    for col in credit_history.columns:
        if credit_history[col].nunique() == len(credit_history):
            print(f"  ✓ '{col}' has all unique values (potential primary key)")
        elif 'id' in col.lower() or 'ref' in col.lower() or 'cust' in col.lower():
            unique_ratio = credit_history[col].nunique() / len(credit_history)
            print(f"  - '{col}' uniqueness ratio: {unique_ratio:.2%}")

    # Check for duplicates
    print(f"\nDuplicate rows: {credit_history.duplicated().sum()}")

    # Missing values
    print("\nMissing Values:")
    missing_df = pd.DataFrame({
        'Column': credit_history.columns,
        'Missing_Count': credit_history.isnull().sum(),
        'Missing_Percentage': (credit_history.isnull().sum() / len(credit_history) * 100).round(2)
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    if missing_df['Missing_Count'].sum() == 0:
        print("No missing values found!")

    # Store summary
    report['file_summaries']['credit_history'] = {
        'rows': credit_history.shape[0],
        'columns': credit_history.shape[1],
        'duplicates': credit_history.duplicated().sum(),
        'missing_values': missing_df['Missing_Count'].sum()
    }

except Exception as e:
    print(f"Error loading credit_history.parquet: {e}")

# ============================================================================
# 4. FINANCIAL_RATIOS.JSONL ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("4. FINANCIAL_RATIOS.JSONL ANALYSIS")
print("=" * 100)

try:
    # Read JSONL file
    financial_data = []
    with open(DATA_DIR / 'financial_ratios.jsonl', 'r') as f:
        for line in f:
            financial_data.append(json.loads(line))

    financial_ratios = pd.DataFrame(financial_data)

    print("\n### Dataset Overview ###")
    print(f"Shape: {financial_ratios.shape[0]} rows × {financial_ratios.shape[1]} columns")
    print(f"\n⚠️ NOTE: This file has {financial_ratios.shape[0]} rows (1 less than others)")

    print(f"\nColumn Names and Types:")
    print(financial_ratios.dtypes)

    print(f"\nFirst 5 rows:")
    print(financial_ratios.head())

    print("\n### Data Quality Analysis ###")

    # Check for duplicates
    print(f"Duplicate rows: {financial_ratios.duplicated().sum()}")
    if 'cust_num' in financial_ratios.columns:
        print(f"Duplicate cust_num: {financial_ratios['cust_num'].duplicated().sum()}")

    # Check for string formatting issues in numeric columns
    print("\n### Numeric Field Formatting Issues ###")
    numeric_cols = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                   'revolving_balance', 'available_credit', 'total_debt_amount']

    for col in numeric_cols:
        if col in financial_ratios.columns:
            # Check if column contains strings
            if financial_ratios[col].dtype == 'object':
                sample_values = financial_ratios[col].dropna().head(5)
                print(f"\n{col} (stored as string):")
                print(f"  Sample values: {sample_values.tolist()}")

                # Check for $ and comma
                has_dollar = financial_ratios[col].astype(str).str.contains('\\$', na=False).sum()
                has_comma = financial_ratios[col].astype(str).str.contains(',', na=False).sum()
                print(f"  Rows with $: {has_dollar}")
                print(f"  Rows with comma: {has_comma}")

    # Missing values
    print("\n\nMissing Values:")
    missing_df = pd.DataFrame({
        'Column': financial_ratios.columns,
        'Missing_Count': financial_ratios.isnull().sum(),
        'Missing_Percentage': (financial_ratios.isnull().sum() / len(financial_ratios) * 100).round(2)
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    if missing_df['Missing_Count'].sum() == 0:
        print("No missing values found!")

    # Store summary
    report['file_summaries']['financial_ratios'] = {
        'rows': financial_ratios.shape[0],
        'columns': financial_ratios.shape[1],
        'primary_key': 'cust_num',
        'duplicates': financial_ratios.duplicated().sum(),
        'missing_values': missing_df['Missing_Count'].sum()
    }

    report['data_quality_issues']['financial_ratios'] = [
        'Has 89,999 rows (1 less than other files)',
        'Numeric fields stored as strings with "$" and "," formatting'
    ]

except Exception as e:
    print(f"Error loading financial_ratios.jsonl: {e}")

# ============================================================================
# 5. LOAN_DETAILS.XLSX ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("5. LOAN_DETAILS.XLSX ANALYSIS")
print("=" * 100)

try:
    loan_details = pd.read_excel(DATA_DIR / 'loan_details.xlsx', engine='openpyxl')

    print("\n### Dataset Overview ###")
    print(f"Shape: {loan_details.shape[0]} rows × {loan_details.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(loan_details.dtypes)

    print(f"\nFirst 5 rows:")
    print(loan_details.head())

    print("\n### Data Quality Analysis ###")

    # Identify potential primary key
    print("\nChecking for potential primary keys:")
    for col in loan_details.columns:
        if loan_details[col].nunique() == len(loan_details):
            print(f"  ✓ '{col}' has all unique values (potential primary key)")
        elif 'id' in col.lower() or 'ref' in col.lower() or 'cust' in col.lower() or 'loan' in col.lower():
            unique_ratio = loan_details[col].nunique() / len(loan_details)
            print(f"  - '{col}' uniqueness ratio: {unique_ratio:.2%}")

    # Check for duplicates
    print(f"\nDuplicate rows: {loan_details.duplicated().sum()}")

    # Missing values
    print("\nMissing Values:")
    missing_df = pd.DataFrame({
        'Column': loan_details.columns,
        'Missing_Count': loan_details.isnull().sum(),
        'Missing_Percentage': (loan_details.isnull().sum() / len(loan_details) * 100).round(2)
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    if missing_df['Missing_Count'].sum() == 0:
        print("No missing values found!")

    # Store summary
    report['file_summaries']['loan_details'] = {
        'rows': loan_details.shape[0],
        'columns': loan_details.shape[1],
        'duplicates': loan_details.duplicated().sum(),
        'missing_values': missing_df['Missing_Count'].sum()
    }

except Exception as e:
    print(f"Error loading loan_details.xlsx: {e}")

# ============================================================================
# 6. GEOGRAPHIC_DATA.XML ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("6. GEOGRAPHIC_DATA.XML ANALYSIS")
print("=" * 100)

try:
    # Parse XML file
    tree = ET.parse(DATA_DIR / 'geographic_data.xml')
    root = tree.getroot()

    # Extract all customer records
    geographic_data = []
    for customer in root.findall('.//customer'):
        record = {}
        for child in customer:
            record[child.tag] = child.text
        geographic_data.append(record)

    geographic_df = pd.DataFrame(geographic_data)

    print("\n### Dataset Overview ###")
    print(f"Shape: {geographic_df.shape[0]} rows × {geographic_df.shape[1]} columns")
    print(f"\n⚠️ NOTE: This file has {geographic_df.shape[0]} rows (1 less than main file)")

    print(f"\nColumn Names and Types:")
    print(geographic_df.dtypes)

    print(f"\nFirst 5 rows:")
    print(geographic_df.head())

    print("\n### Data Quality Analysis ###")

    # Check for duplicates
    print(f"Duplicate rows: {geographic_df.duplicated().sum()}")
    if 'id' in geographic_df.columns:
        print(f"Duplicate id: {geographic_df['id'].duplicated().sum()}")

    # Missing values
    print("\nMissing Values:")
    missing_df = pd.DataFrame({
        'Column': geographic_df.columns,
        'Missing_Count': geographic_df.isnull().sum(),
        'Missing_Percentage': (geographic_df.isnull().sum() / len(geographic_df) * 100).round(2)
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    if missing_df['Missing_Count'].sum() == 0:
        print("No missing values found!")

    # Check numeric columns stored as strings
    print("\n### Numeric Columns Type Check ###")
    numeric_cols = ['regional_unemployment_rate', 'regional_median_income',
                   'regional_median_rent', 'housing_price_index', 'cost_of_living_index']
    for col in numeric_cols:
        if col in geographic_df.columns:
            print(f"{col}: dtype = {geographic_df[col].dtype}")

    # Store summary
    report['file_summaries']['geographic_data'] = {
        'rows': geographic_df.shape[0],
        'columns': geographic_df.shape[1],
        'primary_key': 'id',
        'duplicates': geographic_df.duplicated().sum(),
        'missing_values': missing_df['Missing_Count'].sum()
    }

    report['data_quality_issues']['geographic_data'] = [
        'Has 89,999 rows (1 less than main file)',
        'Numeric fields may be stored as strings (from XML parsing)'
    ]

except Exception as e:
    print(f"Error loading geographic_data.xml: {e}")

# ============================================================================
# MERGE STRATEGY ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("MERGE STRATEGY ANALYSIS")
print("=" * 100)

print("\n### Primary Keys Mapping ###")
print("1. application_metadata.csv: 'customer_ref' (90,000 rows)")
print("2. demographics.csv: 'cust_id' (90,000 rows)")
print("3. credit_history.parquet: [Need to identify] (check output above)")
print("4. financial_ratios.jsonl: 'cust_num' (89,999 rows)")
print("5. loan_details.xlsx: [Need to identify] (check output above)")
print("6. geographic_data.xml: 'id' (89,999 rows)")

print("\n### Record Count Discrepancy ###")
print("Files with 90,000 records:")
print("  - application_metadata.csv")
print("  - demographics.csv")
print("  - credit_history.parquet (needs verification)")
print("  - loan_details.xlsx (needs verification)")
print("\nFiles with 89,999 records:")
print("  - financial_ratios.jsonl")
print("  - geographic_data.xml")
print("\n⚠️ Need to identify which customer(s) are missing from the 89,999-row files")

# Try to identify missing records
print("\n### Identifying Missing Records ###")
try:
    # Check which customer_ref from application_metadata is missing in financial_ratios
    if 'customer_ref' in app_metadata.columns and 'cust_num' in financial_ratios.columns:
        app_refs = set(app_metadata['customer_ref'].astype(str))
        fin_refs = set(financial_ratios['cust_num'].astype(str))
        missing_in_financial = app_refs - fin_refs
        if missing_in_financial:
            print(f"Customer(s) missing in financial_ratios: {list(missing_in_financial)[:10]}")

    # Check which customer_ref is missing in geographic_data
    if 'customer_ref' in app_metadata.columns and 'id' in geographic_df.columns:
        geo_refs = set(geographic_df['id'].astype(str))
        missing_in_geographic = app_refs - geo_refs
        if missing_in_geographic:
            print(f"Customer(s) missing in geographic_data: {list(missing_in_geographic)[:10]}")
except:
    print("Could not identify missing records - will need to check after data type conversion")

# ============================================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 100)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("=" * 100)

print("\n### Dataset Summary ###")
print("-" * 50)
for file_name, summary in report['file_summaries'].items():
    print(f"\n{file_name.upper()}:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

print("\n### Critical Data Quality Issues ###")
print("-" * 50)
for file_name, issues in report['data_quality_issues'].items():
    print(f"\n{file_name.upper()}:")
    for issue in issues:
        print(f"  • {issue}")

print("\n### Recommended Data Cleaning Steps ###")
print("-" * 50)
print("\n1. STANDARDIZE PRIMARY KEYS:")
print("   • Map all key columns to unified 'customer_id'")
print("   • Ensure all keys are same data type (preferably integer)")

print("\n2. CLEAN NUMERIC FIELDS:")
print("   • Remove '$' and ',' from annual_income in demographics")
print("   • Remove '$' and ',' from all numeric fields in financial_ratios")
print("   • Convert XML numeric fields from string to float")

print("\n3. NORMALIZE CATEGORICAL VALUES:")
print("   • Standardize employment_type casing in demographics")
print("   • Create mapping for consistent categorical encoding")

print("\n4. REMOVE NOISE:")
print("   • Drop 'random_noise_1' column from application_metadata")
print("   • Check for other potential noise columns")

print("\n5. HANDLE MISSING RECORD:")
print("   • Identify which customer is missing from 89,999-row files")
print("   • Decide on imputation or exclusion strategy")

print("\n### Recommended Merge Strategy ###")
print("-" * 50)
print("\n1. Start with application_metadata as base (contains target)")
print("2. Left join demographics on customer_ref = cust_id")
print("3. Left join financial_ratios on customer_ref = cust_num")
print("4. Left join geographic_data on customer_ref = id")
print("5. Left join credit_history (after identifying key)")
print("6. Left join loan_details (after identifying key)")
print("7. Validate no unexpected nulls after joins")
print("8. Handle the 1 missing record appropriately")

print("\n### Next Steps ###")
print("-" * 50)
print("1. Execute data cleaning based on identified issues")
print("2. Create unified dataset through systematic joins")
print("3. Engineer features for credit risk prediction")
print("4. Handle class imbalance if present")
print("5. Build and evaluate models optimizing for AUC")

print("\n" + "=" * 100)
print("END OF REPORT")
print("=" * 100)