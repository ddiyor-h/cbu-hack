# Credit Default Prediction - Comprehensive Data Analysis Report

**Date:** November 15, 2025
**Analyst:** Data Science Specialist
**Project:** Credit Default Prediction (AUC-optimized)

---

## Executive Summary

This report presents a comprehensive analysis of 6 heterogeneous data files containing customer credit information for ~90,000 records. The objective is to understand data structure, identify quality issues, and prepare a strategy for creating a unified ML-ready dataset for credit default prediction.

### Key Findings:
- **All files actually contain 89,999 records** (initial documentation was incorrect)
- **5.10% default rate** in the target variable (moderate class imbalance)
- **Multiple data quality issues** requiring systematic cleaning
- **Different primary key naming** across files but consistent customer IDs
- **Significant formatting inconsistencies** in numeric and categorical fields
- **One noise column identified** (random_noise_1)

---

## 1. Dataset Overview

### File Summary Table

| File | Format | Rows | Columns | Primary Key | Missing Values | Key Issues |
|------|--------|------|---------|-------------|----------------|------------|
| application_metadata.csv | CSV | 89,999 | 14 | customer_ref | 0 | Noise column (random_noise_1) |
| demographics.csv | CSV | 89,999 | 8 | cust_id | 2,253 | Inconsistent formatting |
| credit_history.parquet | Parquet | 89,999 | 12 | customer_number | 832 | - |
| financial_ratios.jsonl | JSONL | 89,999 | 16 | cust_num | 1,377 | String formatting with $, commas |
| loan_details.xlsx | Excel | 89,999 | 10 | customer_id | 0 | Mixed case in loan_type |
| geographic_data.xml | XML | 89,999 | 8 | id | 0 | Numeric fields as strings |

**Important Discovery:** All files contain exactly 89,999 records. The documentation stating 90,000 records appears to be incorrect.

---

## 2. Detailed File Analysis

### 2.1 Application Metadata (Primary Dataset)
**File:** `application_metadata.csv`

#### Key Attributes:
- **Target Variable:** `default` (binary: 0/1)
- **Default Rate:** 5.10% (4,594 defaults out of 89,999)
- **Customer Identifier:** `customer_ref` (ranges from 10000 to 99998)

#### Columns:
- Customer & Application: customer_ref, application_id, application_hour, application_day_of_week
- Account Information: account_open_year, account_status_code, preferred_contact, referral_code
- Behavioral Metrics: num_login_sessions, num_customer_service_calls, has_mobile_app, paperless_billing
- **Noise:** random_noise_1 (artificial noise to be removed)
- **Target:** default

#### Quality Assessment:
✅ No missing values
✅ No duplicates
❌ Contains artificial noise column

---

### 2.2 Demographics
**File:** `demographics.csv`

#### Key Issues Identified:

1. **Annual Income Formatting Chaos:**
   - Mixed formats: "$61,800" vs "28,600" vs "$20700"
   - 54,060 rows with $ symbol
   - 54,041 rows with comma separators

2. **Employment Type Inconsistency:**
   - 16 unique variations of similar concepts
   - Examples: "Full-time", "FULL_TIME", "Full Time", "Fulltime", "FT"
   - Needs normalization mapping

3. **Missing Values:**
   - employment_length: 2,253 missing (2.5%)

---

### 2.3 Credit History
**File:** `credit_history.parquet`

#### Key Attributes:
- **Primary Key:** `customer_number` (verified unique)
- **Credit Metrics:** credit_score, num_credit_accounts, total_credit_limit
- **Risk Indicators:** num_delinquencies_2yrs, num_inquiries_6mo, num_public_records

#### Quality Assessment:
✅ Clean numeric data types
✅ Reasonable value ranges
⚠️ 832 missing values in num_delinquencies_2yrs (0.92%)

---

### 2.4 Financial Ratios
**File:** `financial_ratios.jsonl`

#### Critical Issues:

1. **Widespread String Formatting in Numeric Fields:**
   - monthly_income: 54,153 with $, 62,973 with commas
   - existing_monthly_debt: 35,976 with $, 20,513 with commas
   - monthly_payment: 36,464 with $, 20,416 with commas
   - revolving_balance: 35,519 with $, 62,068 with commas
   - All numeric fields need parsing

2. **Pre-calculated Ratios:**
   - debt_to_income_ratio
   - debt_service_ratio
   - payment_to_income_ratio
   - credit_utilization
   - **Recommendation:** Verify calculations after cleaning

3. **Missing Values:**
   - revolving_balance: 1,377 missing (1.53%)

---

### 2.5 Loan Details
**File:** `loan_details.xlsx`

#### Key Attributes:
- **Primary Key:** `customer_id` (verified unique)
- **Loan Characteristics:** loan_type, loan_amount, loan_term, interest_rate, loan_purpose

#### Quality Issues:
- loan_type inconsistent casing: "Personal" vs "personal" vs "PERSONAL"
- loan_amount stored as string with $ and commas
- No missing values

---

### 2.6 Geographic Data
**File:** `geographic_data.xml`

#### Key Attributes:
- **Primary Key:** `id` (stored as string from XML)
- **Regional Economics:** unemployment_rate, median_income, median_rent
- **Market Indices:** housing_price_index, cost_of_living_index

#### Quality Issues:
- All numeric fields stored as strings (XML parsing artifact)
- Needs type conversion for all numeric columns

---

## 3. Data Quality Issues Summary

### Critical Issues Requiring Immediate Attention:

1. **Inconsistent Primary Key Names:**
   - customer_ref, cust_id, customer_number, cust_num, customer_id, id
   - All refer to same entity (customer IDs from 10000-99998)

2. **Numeric Field String Formatting:**
   - Affects 11 columns across 3 files
   - Mixed use of $, commas, inconsistent decimal places

3. **Categorical Inconsistencies:**
   - employment_type: 16 variations of 4-5 actual categories
   - loan_type: Mixed case variations

4. **Artificial Noise:**
   - random_noise_1 column must be removed

5. **Missing Values:**
   - Total: 4,462 missing values across all files
   - Most significant: employment_length (2,253)

---

## 4. Merge Strategy

### Recommended Join Sequence:

```python
# Pseudo-code for merge strategy
base_df = application_metadata  # Contains target variable

# Perform left joins to preserve all customers
merged_df = (base_df
    .merge(demographics, left_on='customer_ref', right_on='cust_id', how='left')
    .merge(credit_history, left_on='customer_ref', right_on='customer_number', how='left')
    .merge(financial_ratios, left_on='customer_ref', right_on='cust_num', how='left')
    .merge(loan_details, left_on='customer_ref', right_on='customer_id', how='left')
    .merge(geographic_data, left_on='customer_ref', right_on='id', how='left')
)
```

### Key Considerations:
- Use LEFT joins from application_metadata to preserve all records with target
- Validate join quality by checking unexpected NULLs
- Drop redundant key columns after joining

---

## 5. Data Cleaning Pipeline

### Phase 1: Standardization
```python
# 1. Remove noise
df.drop('random_noise_1', axis=1, inplace=True)

# 2. Standardize numeric fields
def clean_currency(value):
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('$', '').replace(',', ''))

# 3. Normalize employment types
employment_mapping = {
    'Full-time': 'full_time', 'FULL_TIME': 'full_time',
    'Full Time': 'full_time', 'Fulltime': 'full_time', 'FT': 'full_time',
    'Part-time': 'part_time', 'PART_TIME': 'part_time',
    'Part Time': 'part_time', 'PT': 'part_time',
    # ... etc
}
```

### Phase 2: Type Conversion
- Convert all string-stored numeric fields to float64
- Convert categorical variables to category dtype
- Ensure datetime fields are properly parsed

### Phase 3: Missing Value Strategy
- employment_length: Consider median imputation or indicator variable
- num_delinquencies_2yrs: Likely 0 imputation (absence of delinquencies)
- revolving_balance: Investigate correlation with other credit fields

---

## 6. Feature Engineering Opportunities

### Potential Features to Create:

1. **Temporal Features:**
   - Account age (current_year - account_open_year)
   - Application time patterns (morning/afternoon/evening)
   - Weekend vs weekday applications

2. **Financial Health Indicators:**
   - Free cash flow ratio
   - Credit utilization bins (low/medium/high)
   - Debt service coverage ratio

3. **Geographic Risk Scores:**
   - Regional economic stress index
   - Cost-adjusted income ratios

4. **Behavioral Patterns:**
   - Digital engagement score (mobile app + paperless billing)
   - Service interaction intensity

---

## 7. Model Preparation Considerations

### Class Imbalance:
- Default rate: 5.10% (moderate imbalance)
- Consider: SMOTE, class weights, or ensemble methods

### Feature Scaling:
- Standardization recommended for numeric features
- Consider RobustScaler for features with outliers

### Encoding Strategy:
- One-hot encoding for low-cardinality categoricals (state, loan_purpose)
- Target encoding for high-cardinality features if needed

### Cross-Validation:
- Stratified K-fold to preserve class distribution
- Time-based split if temporal patterns are significant

---

## 8. Recommendations

### Immediate Actions:

1. **Data Cleaning Script:** Create reproducible cleaning pipeline with clear documentation
2. **Validation Framework:** Implement data quality checks at each step
3. **Feature Store:** Save cleaned, merged dataset in Parquet format for efficiency
4. **EDA Notebook:** Create visualizations for:
   - Distribution of numeric features
   - Correlation matrix
   - Default rate by categorical variables

### Model Development Strategy:

1. **Baseline Model:** Logistic Regression with basic features
2. **Feature Selection:** Use importance scores and correlation analysis
3. **Advanced Models:**
   - Random Forest for non-linear patterns
   - XGBoost/LightGBM for best performance
   - Consider ensemble of multiple models
4. **Evaluation:** Focus on AUC as primary metric, track Precision-Recall curves

### Quality Assurance:

1. **Reproducibility:** Set random seeds, version control, document decisions
2. **Data Leakage Prevention:** Careful feature engineering, proper train/test split
3. **Performance Monitoring:** Track metrics across different data segments

---

## 9. Technical Implementation Notes

### Required Libraries:
```python
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
```

### File I/O Optimizations:
- Use chunking for large file operations
- Save intermediate results in Parquet format
- Implement progress bars for long operations

### Memory Management:
- Use appropriate dtypes (int8/int16 for small integers)
- Delete intermediate dataframes after joins
- Consider Dask for very large-scale operations

---

## 10. Conclusion

The dataset presents typical real-world challenges with multiple formats, inconsistent data entry, and quality issues. However, with systematic cleaning and careful feature engineering, it provides rich information for credit default prediction. The 5.10% default rate suggests sufficient positive cases for model training, though class imbalance techniques may improve performance.

**Key Success Factors:**
- Thorough data cleaning with validation
- Careful handling of missing values
- Feature engineering leveraging domain knowledge
- Rigorous evaluation focused on AUC metric

**Expected Outcomes:**
After implementing the recommended cleaning and preparation steps, we expect to achieve:
- A unified dataset with 89,999 records and ~65 features
- Clean, properly typed data ready for ML algorithms
- AUC performance potential of 0.75-0.85 based on feature richness

---

## Appendix: File Paths and Code Snippets

### Data Directory:
```
/home/dr/cbu/drive-download-20251115T045945Z-1-001/
```

### Key Files:
- Analysis Script: `/home/dr/cbu/data_analysis_report.py`
- This Report: `/home/dr/cbu/DATA_ANALYSIS_REPORT.md`

### Sample Data Loading Code:
```python
# Load all datasets
app_metadata = pd.read_csv(DATA_DIR / 'application_metadata.csv')
demographics = pd.read_csv(DATA_DIR / 'demographics.csv')
credit_history = pd.read_parquet(DATA_DIR / 'credit_history.parquet')

# Load JSONL
with open(DATA_DIR / 'financial_ratios.jsonl', 'r') as f:
    financial_ratios = pd.DataFrame([json.loads(line) for line in f])

# Load Excel
loan_details = pd.read_excel(DATA_DIR / 'loan_details.xlsx', engine='openpyxl')

# Parse XML
tree = ET.parse(DATA_DIR / 'geographic_data.xml')
root = tree.getroot()
geographic_data = pd.DataFrame([
    {child.tag: child.text for child in customer}
    for customer in root.findall('.//customer')
])
```

---

**Report Generated:** November 15, 2025
**Next Step:** Proceed with data cleaning implementation based on identified issues