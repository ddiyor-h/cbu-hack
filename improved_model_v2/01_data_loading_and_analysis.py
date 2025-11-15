"""
Data Loading and Exploratory Analysis
Based on best practices for credit default prediction
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_application_metadata(data_dir):
    """Load application metadata CSV"""
    df = pd.read_csv(f"{data_dir}/application_metadata.csv")
    print(f"Application Metadata: {df.shape}")
    return df

def load_demographics(data_dir):
    """Load demographics CSV with data cleaning"""
    df = pd.read_csv(f"{data_dir}/demographics.csv")
    print(f"Demographics: {df.shape}")

    # Clean annual_income: remove $ and , characters
    if df['annual_income'].dtype == 'object':
        df['annual_income'] = df['annual_income'].str.replace('$', '', regex=False)
        df['annual_income'] = df['annual_income'].str.replace(',', '', regex=False)
        df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')

    # Normalize employment_type to consistent format
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].str.strip().str.lower().str.replace(' ', '_')

    return df

def load_credit_history(data_dir):
    """Load credit history Parquet file"""
    df = pd.read_parquet(f"{data_dir}/credit_history.parquet")
    print(f"Credit History: {df.shape}")
    return df

def load_financial_ratios(data_dir):
    """Load financial ratios JSONL file with cleaning"""
    records = []
    with open(f"{data_dir}/financial_ratios.jsonl", 'r') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"Financial Ratios: {df.shape}")

    # Clean numeric columns with $ and , characters
    numeric_cols = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                    'revolving_balance', 'available_credit', 'total_debt_amount']

    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace('$', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def load_loan_details(data_dir):
    """Load loan details Excel file"""
    df = pd.read_excel(f"{data_dir}/loan_details.xlsx")
    print(f"Loan Details: {df.shape}")
    return df

def load_geographic_data(data_dir):
    """Load geographic data XML file"""
    tree = ET.parse(f"{data_dir}/geographic_data.xml")
    root = tree.getroot()

    records = []
    for customer in root.findall('.//customer'):
        record = {}
        record['id'] = customer.find('id').text if customer.find('id') is not None else None
        record['state'] = customer.find('state').text if customer.find('state') is not None else None
        record['previous_zip_code'] = customer.find('previous_zip_code').text if customer.find('previous_zip_code') is not None else None

        # Numeric fields
        for field in ['regional_unemployment_rate', 'regional_median_income',
                      'regional_median_rent', 'housing_price_index', 'cost_of_living_index']:
            elem = customer.find(field)
            record[field] = float(elem.text) if elem is not None and elem.text else None

        records.append(record)

    df = pd.DataFrame(records)
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    print(f"Geographic Data: {df.shape}")
    return df


# ============================================================================
# DATA QUALITY ANALYSIS FUNCTIONS
# ============================================================================

def analyze_data_quality(df, name):
    """Comprehensive data quality analysis"""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ANALYSIS: {name}")
    print(f"{'='*60}")

    print(f"\nShape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes.value_counts()}")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)

    if len(missing_df) > 0:
        print(f"\nMissing Values:")
        print(missing_df)
    else:
        print(f"\nNo missing values")

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")

    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Columns Summary:")
        print(df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']])

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical Columns (Unique Values):")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count < 20:  # Show value counts for low cardinality
                print(f"    {df[col].value_counts().head().to_dict()}")

    return missing_df


def calculate_correlation_with_target(df, target_col='default'):
    """Calculate correlation of all numeric features with target"""
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]

    correlations = {}
    for col in numeric_cols:
        if df[col].notna().sum() > 0:  # Only if there's non-null data
            corr = df[[col, target_col]].corr().iloc[0, 1]
            correlations[col] = corr

    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)

    print(f"\n{'='*60}")
    print(f"CORRELATION WITH TARGET (absolute values)")
    print(f"{'='*60}")
    print(corr_df)

    return corr_df


def identify_noise_columns(df, threshold=0.01):
    """Identify potential noise columns based on correlation and variance"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    noise_candidates = []

    for col in numeric_cols:
        # Check for columns with 'noise' or 'random' in name
        if 'noise' in col.lower() or 'random' in col.lower():
            noise_candidates.append((col, 'name_pattern'))
        # Check for very low variance (constant or near-constant)
        elif df[col].std() < 0.0001:
            noise_candidates.append((col, 'low_variance'))

    if noise_candidates:
        print(f"\n{'='*60}")
        print(f"POTENTIAL NOISE COLUMNS")
        print(f"{'='*60}")
        for col, reason in noise_candidates:
            print(f"  {col}: {reason}")

    return [col for col, _ in noise_candidates]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    data_dir = "drive-download-20251115T045945Z-1-001"

    print("="*60)
    print("CREDIT DEFAULT PREDICTION - DATA LOADING AND ANALYSIS")
    print("Based on Best Practices from Recent Research")
    print("="*60)

    # Load all datasets
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)

    app_meta = load_application_metadata(data_dir)
    demographics = load_demographics(data_dir)
    credit_history = load_credit_history(data_dir)
    financial_ratios = load_financial_ratios(data_dir)
    loan_details = load_loan_details(data_dir)
    geographic_data = load_geographic_data(data_dir)

    # Analyze each dataset
    print("\n" + "="*60)
    print("STEP 2: DATA QUALITY ANALYSIS")
    print("="*60)

    analyze_data_quality(app_meta, "Application Metadata")
    analyze_data_quality(demographics, "Demographics")
    analyze_data_quality(credit_history, "Credit History")
    analyze_data_quality(financial_ratios, "Financial Ratios")
    analyze_data_quality(loan_details, "Loan Details")
    analyze_data_quality(geographic_data, "Geographic Data")

    # Identify join keys
    print("\n" + "="*60)
    print("STEP 3: IDENTIFYING JOIN KEYS")
    print("="*60)

    print("\nApplication Metadata key columns:")
    print([col for col in app_meta.columns if 'id' in col.lower() or 'ref' in col.lower() or 'customer' in col.lower()])

    print("\nDemographics key columns:")
    print([col for col in demographics.columns if 'id' in col.lower() or 'ref' in col.lower() or 'customer' in col.lower()])

    print("\nCredit History key columns:")
    print([col for col in credit_history.columns if 'id' in col.lower() or 'ref' in col.lower() or 'customer' in col.lower()])

    print("\nFinancial Ratios key columns:")
    print([col for col in financial_ratios.columns if 'id' in col.lower() or 'num' in col.lower() or 'customer' in col.lower()])

    print("\nLoan Details key columns:")
    print([col for col in loan_details.columns if 'id' in col.lower() or 'ref' in col.lower() or 'customer' in col.lower()])

    print("\nGeographic Data key columns:")
    print([col for col in geographic_data.columns if 'id' in col.lower()])

    # Identify noise columns
    print("\n" + "="*60)
    print("STEP 4: IDENTIFYING NOISE COLUMNS")
    print("="*60)

    noise_cols_app = identify_noise_columns(app_meta)
    noise_cols_demo = identify_noise_columns(demographics)
    noise_cols_credit = identify_noise_columns(credit_history)
    noise_cols_fin = identify_noise_columns(financial_ratios)
    noise_cols_loan = identify_noise_columns(loan_details)
    noise_cols_geo = identify_noise_columns(geographic_data)

    print(f"\nTotal noise columns identified: {len(noise_cols_app + noise_cols_demo + noise_cols_credit + noise_cols_fin + noise_cols_loan + noise_cols_geo)}")

    # Target distribution
    if 'default' in app_meta.columns:
        print("\n" + "="*60)
        print("STEP 5: TARGET VARIABLE ANALYSIS")
        print("="*60)
        print(f"\nTarget Distribution:")
        print(app_meta['default'].value_counts())
        print(f"\nClass Balance:")
        print(app_meta['default'].value_counts(normalize=True))

    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)
