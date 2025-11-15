# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a credit default prediction machine learning project. The goal is to build an optimal and accurate model to predict loan defaults based on customer data from multiple sources. The project is evaluated on a 5-point scale focusing on:
- Data quality and cleaning (thoroughness and correctness)
- Prediction quality (AUC metric)
- Technical execution (reproducibility, code cleanliness and readability)

## Data Architecture

The project contains **~90,000 customer records** spread across 6 heterogeneous data files that need to be merged into a unified dataset:

### Data Files (in `drive-download-20251115T045945Z-1-001/`)

1. **application_metadata.csv** (90,000 rows)
   - Primary key: `customer_ref`
   - Target variable: `default` (0/1)
   - Application details: `application_id`, `application_hour`, `application_day_of_week`
   - Account info: `account_open_year`, `account_status_code`, `preferred_contact`, `referral_code`
   - Behavioral metrics: `num_login_sessions`, `num_customer_service_calls`, `has_mobile_app`, `paperless_billing`
   - Contains noise column: `random_noise_1` (should be removed during cleaning)

2. **demographics.csv** (90,000 rows)
   - Primary key: `cust_id`
   - Personal info: `age`, `marital_status`, `num_dependents`, `education`
   - Employment: `employment_type`, `employment_length`, `annual_income`
   - **Data quality issues**: Inconsistent formatting in `annual_income` (mixed "$61800", "28,600" formats) and `employment_type` (mixed "Full-time", "FULL_TIME", "Full Time")

3. **credit_history.parquet** (~90,000 rows)
   - Binary format - requires pandas/pyarrow to read
   - Contains historical credit information

4. **financial_ratios.jsonl** (89,999 rows, NDJSON format)
   - Primary key: `cust_num`
   - Financial metrics: `monthly_income`, `existing_monthly_debt`, `monthly_payment`
   - Calculated ratios: `debt_to_income_ratio`, `debt_service_ratio`, `payment_to_income_ratio`, `credit_utilization`
   - Credit details: `revolving_balance`, `available_credit`, `total_debt_amount`
   - **Data quality issues**: Inconsistent string formatting with "$" and "," characters in numeric fields

5. **loan_details.xlsx** (Excel format)
   - Requires openpyxl or xlrd to read
   - Contains loan-specific information

6. **geographic_data.xml** (89,999 records)
   - Primary key: `<id>` element
   - Location: `<state>`, `<previous_zip_code>`
   - Regional economics: `<regional_unemployment_rate>`, `<regional_median_income>`, `<regional_median_rent>`
   - Market indices: `<housing_price_index>`, `<cost_of_living_index>`

### Key Data Challenges

1. **Multi-format integration**: CSV, Parquet, JSONL, Excel, XML files must be unified
2. **Inconsistent keys**: Different primary key column names (`customer_ref`, `cust_id`, `cust_num`, `<id>`)
3. **String formatting noise**: Currency symbols, commas, and inconsistent case in categorical variables
4. **Missing data**: One file has 89,999 rows vs 90,000 in others
5. **Noise columns**: `random_noise_1` and potentially other artificially added noise features

## Development Workflow

### Setting Up Environment

```bash
# Install required data processing libraries
pip install pandas pyarrow openpyxl scikit-learn

# For model development
pip install numpy scipy matplotlib seaborn jupyter

# For advanced ML (if needed)
pip install xgboost lightgbm catboost
```

### Running Analysis

Since this is a data science project without existing code:
- Work should be done in Jupyter notebooks for exploratory data analysis
- Final model training scripts should be reproducible Python files
- Use `jupyter notebook` or `jupyter lab` to start interactive analysis

### Testing Approach

For reproducibility:
- Create a requirements.txt with pinned versions
- Document random seeds for model training
- Save preprocessed data to avoid re-running expensive transformations
- Track model performance metrics (especially AUC)

## Data Processing Strategy

### Phase 1: Load and Explore
- Load each file using appropriate library (pandas for CSV/Parquet/Excel, xml.etree for XML, json for JSONL)
- Check for duplicates, missing values, and data type issues
- Understand the join keys and relationships between datasets

### Phase 2: Clean Individual Datasets
- Standardize `annual_income` formatting (remove "$", ",", convert to float)
- Normalize `employment_type` values to consistent case
- Clean financial_ratios numeric fields (strip "$", ",")
- Remove `random_noise_1` column
- Parse XML properly to extract customer records

### Phase 3: Merge Strategy
- Primary dataset: `application_metadata.csv` (contains target variable `default`)
- Join demographics on `customer_ref = cust_id`
- Join financial_ratios on `customer_ref = cust_num`
- Join geographic_data on `customer_ref = id`
- Join credit_history (determine key after loading)
- Join loan_details (determine key after loading)
- Handle the 1-record discrepancy (investigate which customers are missing)

### Phase 4: Feature Engineering
- The financial ratios are pre-calculated but verify their correctness
- Consider interaction features between geographic and financial data
- Handle categorical encoding for state, employment_type, education, marital_status

### Phase 5: Model Training
- Target: `default` column (binary classification)
- Evaluation metric: **AUC (Area Under ROC Curve)**
- Consider class imbalance if present
- Try multiple algorithms (Logistic Regression baseline, Random Forest, Gradient Boosting)
- Cross-validation for robust performance estimation

## File Locations

All data files are in: `drive-download-20251115T045945Z-1-001/`
- Note: The directory name includes a timestamp - may need to adjust if re-downloaded

## Important Notes

- **Language**: README.md is in Russian, indicating project may be from Russian-speaking context
- **Target metric**: AUC is the primary evaluation metric for model quality
- **Deliverable**: A complete, reproducible solution with clean code
- No existing code structure - this is a greenfield data science project
