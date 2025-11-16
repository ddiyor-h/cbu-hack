# LEAK-FREE PIPELINE V3 REPORT
## Complete Data Leakage Analysis and Fix

**Date:** November 16, 2025
**Author:** Data Science Specialist
**Objective:** Fix severe data leakage causing 0.9603 train AUC vs 0.7732 test AUC

---

## EXECUTIVE SUMMARY

The v2 pipeline suffered from severe data leakage, causing an overfitting gap of **0.1871** (18.71%). This report documents all leakage sources found and how they were fixed using strict out-of-fold (OOF) methodology.

---

## PART 1: DATA LEAKAGE SOURCES IDENTIFIED IN V2

### 1. KNN Meta-Features (CRITICAL LEAKAGE)
**File:** `advanced_feature_engineering_v2.py`, Lines 54-67

**Problem:**
```python
# LEAKY CODE FROM V2:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_cols])  # Fit on ENTIRE train
knn.fit(X_train_scaled, y_train['default'])  # Train on ENTIRE train
X_train_v2[f'knn_target_prob_{n_neighbors}'] = knn.predict_proba(X_train_scaled)[:, 1]  # Predict on SAME data!
```

**Impact:** The model "memorizes" the training data, creating features with correlation >0.40 to target (vs 0.27 without leakage).

### 2. StandardScaler (MODERATE LEAKAGE)
**Problem:** Scaler fitted on entire X_train, then used for all folds in CV.

**Impact:** Validation folds see statistics from their own data, inflating CV scores.

### 3. Target Encoding (PARTIAL LEAKAGE)
**File:** Lines 87-107

**Problem:** While k-fold CV was used for train, test encoding used FULL training statistics:
```python
# LEAKY CODE:
state_means_full = pd.DataFrame({...}).groupby('state')['target'].mean()  # Uses ALL training data
X_test_v2['state_target_encoded'] = X_test['state'].map(state_means_full)
```

### 4. WOE Binning (CRITICAL LEAKAGE)
**File:** Lines 149-200

**Problem:**
```python
# LEAKY CODE:
bins = pd.qcut(X_train_col, q=n_bins, duplicates='drop', retbins=True)[1]  # Bins from ENTIRE train
train_woe = X_train_binned.map(woe_dict)  # Applied to SAME data
```

### 5. SMOTE-Tomek (CV LEAKAGE)
**File:** `smote_tomek_pipeline_v2.py`

**Problem:** Applied BEFORE cross-validation, causing:
- Synthetic samples in validation folds that are derived from training folds
- Validation scores become meaningless

---

## PART 2: LEAK-FREE SOLUTIONS IMPLEMENTED

### Solution 1: Strict OOF KNN Meta-Features

```python
def create_oof_knn_features(X, y, X_test, n_neighbors=50, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(len(X))
    oof_test_folds = []

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # FIT SCALER ONLY ON TRAINING FOLD
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X[numeric_cols].iloc[train_idx])
        X_val_fold = scaler.transform(X[numeric_cols].iloc[val_idx])

        # TRAIN KNN ONLY ON TRAINING FOLD
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_fold, y[train_idx])

        # PREDICT ON VALIDATION (OOF)
        oof_train[val_idx] = knn.predict_proba(X_val_fold)[:, 1]

        # Test predictions averaged across folds
        oof_test_folds.append(knn.predict_proba(X_test_fold)[:, 1])

    return oof_train, np.mean(oof_test_folds, axis=0)
```

**Result:** KNN correlation dropped from 0.40+ to 0.27-0.29 (GOOD - no leakage!)

### Solution 2: OOF Target Encoding with Smoothing

```python
def create_oof_target_encoding(X, y, X_test, column, n_splits=5, smoothing=10):
    # Each fold calculates encoding ONLY from its training data
    # Smoothing prevents overfitting on rare categories
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        global_mean = y[train_idx].mean()
        # Bayesian smoothing
        encoding = (category_mean * n + global_mean * smoothing) / (n + smoothing)
```

### Solution 3: NO SMOTE - Handle Imbalance in Model

**Decision:** Removed SMOTE entirely. Will use `scale_pos_weight` in XGBoost:
```python
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()  # ~18.6
```

### Solution 4: OOF WOE Binning

Each fold creates its own bins and WOE values from only its training data.

---

## PART 3: RESULTS AND VALIDATION

### Feature Engineering Results

| Metric | V2 (With Leakage) | V3 (Leak-Free) | Change |
|--------|-------------------|----------------|---------|
| KNN_50 correlation | 0.40+ | 0.2736 | -31.6% |
| KNN_100 correlation | 0.42+ | 0.2865 | -31.8% |
| KNN_500 correlation | 0.43+ | 0.2965 | -31.0% |
| Target encoding (state) | 0.02 | 0.0107 | -46.5% |
| Target encoding (education) | 0.12 | 0.0960 | -20.0% |

### Class Distribution

| Dataset | Defaults | Total | Rate | Status |
|---------|----------|-------|------|---------|
| Train | 3,675 | 71,999 | 5.10% | Original |
| Test | 919 | 18,000 | 5.11% | Original |

**NO balancing applied** - correct approach for honest evaluation.

### Expected Model Performance

| Metric | V2 (Leaky) | V3 Expected | Explanation |
|--------|------------|-------------|-------------|
| Train AUC | 0.9603 | ~0.78-0.82 | No more memorization |
| Test AUC | 0.7732 | ~0.76-0.80 | Better generalization |
| Gap | 0.1871 | <0.05 | Proper OOF prevents overfitting |

---

## PART 4: FILES CREATED

### Input Files (Clean, No Leakage)
- `X_train_optimized.parquet` - Base features before v2 engineering
- `X_test_optimized.parquet` - Base test features

### Output Files (Leak-Free)
- `X_train_leak_free_v3.parquet` - 71,999 × 89 features (all OOF)
- `X_test_leak_free_v3.parquet` - 18,000 × 89 features
- `y_train_leak_free_v3.parquet` - Original labels (5.10% positive)
- `y_test_leak_free_v3.parquet` - Original labels (5.11% positive)

---

## PART 5: KEY TAKEAWAYS

### What Went Wrong in V2
1. **Feature leakage:** KNN/WOE fitted on data they predicted on
2. **Validation leakage:** Scaler/encoders leaked info between CV folds
3. **SMOTE before CV:** Synthetic samples leaked across folds
4. **Over-engineering:** Too many complex features without proper isolation

### Best Practices Applied in V3
1. **Strict OOF:** Every feature computed out-of-fold
2. **No early balancing:** Handle imbalance in model parameters
3. **Proper test isolation:** Test never touches training process
4. **Realistic expectations:** Accept lower but honest metrics

### Impact on Business Metrics
- **False sense of performance:** V2 suggested 96% train AUC (impossible in credit risk)
- **Production failure risk:** Model would fail catastrophically in production
- **Honest evaluation:** V3 provides realistic performance estimates

---

## PART 6: NEXT STEPS

### Immediate Actions
1. Train XGBoost on leak-free data with proper class weights
2. Use 5-fold CV with same OOF methodology for evaluation
3. Track train-test gap (should be <5%)

### Model Training Recommendations
```python
import xgboost as xgb

# Calculate scale_pos_weight for imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # ~18.6 for 5% default rate
    max_depth=6,
    learning_rate=0.02,
    n_estimators=1000,
    early_stopping_rounds=50,
    eval_metric='auc',
    random_state=42
)

# Proper CV evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc'
)

print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Expected Realistic Performance
- **CV AUC:** 0.78-0.82
- **Test AUC:** 0.76-0.80
- **Train-Test Gap:** < 0.05

---

## CONCLUSION

The v3 pipeline successfully eliminates all data leakage through strict out-of-fold methodology. While the apparent performance metrics are lower, they now represent **honest, reproducible results** that will generalize to production.

The key insight: **A 0.78 AUC without leakage is infinitely more valuable than 0.96 AUC with leakage.**

---

## APPENDIX: VALIDATION CHECKLIST

- [x] All features created using OOF methodology
- [x] No class balancing before model training
- [x] Test set never used in any transformation fitting
- [x] Each CV fold isolated from others
- [x] Realistic correlation values (0.20-0.30 range)
- [x] Original class distribution preserved
- [x] No synthetic samples in training data
- [x] All transformations reproducible with random seeds
- [x] Clear documentation of all changes

---

**Report Generated:** November 16, 2024
**Pipeline Version:** V3 (Leak-Free)
**Status:** READY FOR MODEL TRAINING