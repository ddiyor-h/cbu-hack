# Class Imbalance Handling Recommendations
## Credit Default Prediction (5.11% default rate, 1:19 imbalance)

---

## Current Situation

**Dataset Characteristics:**
- Default rate: 5.11% (3,676 defaults / 72,000 total)
- Imbalance ratio: 1:18.6
- Classification type: Severe class imbalance

**Current Approach:**
- Using `scale_pos_weight=18.59` in XGBoost
- Stratified K-fold cross-validation (5 folds)
- AUC as primary metric (appropriate for imbalanced data)

---

## Problem Analysis

### Why Class Imbalance Matters

1. **Model Bias:** Models tend to predict majority class (non-default)
2. **Evaluation Challenges:** Accuracy is misleading (95% accuracy = always predict non-default)
3. **Business Impact:** Missing defaults (false negatives) is costly
4. **Learning Difficulty:** Model sees 18× more non-default examples

### Current Performance Indicators

**Symptoms of imbalance affecting performance:**
- High training AUC but lower CV AUC suggests model struggles to generalize on minority class
- Feature importance may be skewed toward features that help identify majority class

---

## Recommended Strategies (Ranked by Effectiveness)

### Strategy 1: Weighted Loss Functions (CURRENTLY USING)

**Status:** Implemented with `scale_pos_weight=18.59`

**How it works:**
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# Increases penalty for misclassifying defaults by 18.59×
```

**Pros:**
- No data modification required
- Works well with tree-based models
- Computationally efficient

**Cons:**
- May not fully solve severe imbalance
- Still limited by small number of minority examples

**Optimization:**
```python
# Try different weights empirically
for weight in [10, 15, 18.59, 20, 25, 30]:
    model = XGBClassifier(scale_pos_weight=weight, ...)
    cv_score = cross_val_score(model, X, y, scoring='roc_auc')
    # Plot weight vs AUC to find optimal
```

---

### Strategy 2: SMOTE (Synthetic Minority Over-sampling)

**Recommendation:** **HIGH PRIORITY - Try next**

**How it works:**
- Creates synthetic minority class examples
- Interpolates between existing minority examples
- Balances dataset to 30-50% minority class

**Implementation:**
```python
from imblearn.over_sampling import SMOTE

# Create synthetic examples to achieve 30% minority class
smote = SMOTE(
    sampling_strategy=0.3,  # Target 30% defaults
    random_state=42,
    k_neighbors=5
)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# New distribution: ~30% defaults instead of 5.11%
print(f"New default rate: {y_resampled.mean():.2%}")
# Output: New default rate: 30.00%
```

**Pros:**
- Dramatically increases minority class examples
- Model sees more default patterns
- Often improves AUC by 2-5 points

**Cons:**
- Increases training time (more samples)
- Synthetic samples may not represent real data
- Risk of overfitting to synthetic patterns

**Best Practices:**
```python
# 1. Apply SMOTE ONLY to training set (never test set)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
# X_test unchanged!

# 2. Use moderate oversampling (30-40%, not 50%)
smote = SMOTE(sampling_strategy=0.35)

# 3. Apply SMOTE after train/test split
# WRONG: smote on full data, then split
# RIGHT: split first, then smote on train only

# 4. Combine with weighted loss
model = XGBClassifier(scale_pos_weight=2.0, ...)  # Lower weight since data is more balanced
```

**Expected Impact:** AUC improvement of 0.02-0.04 (0.78 → 0.80-0.82)

---

### Strategy 3: Threshold Optimization

**Recommendation:** **Apply after model training**

**How it works:**
- Default classification threshold is 0.5
- Optimize threshold for business objective
- Trade precision vs recall

**Implementation:**
```python
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve

# Get predicted probabilities
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Method 1: Optimize F1 score
thresholds = np.arange(0.05, 0.5, 0.01)
f1_scores = []

for thresh in thresholds:
    y_pred = (y_pred_proba >= thresh).astype(int)
    f1 = f1_score(y_val, y_pred)
    f1_scores.append(f1)

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold for F1: {optimal_threshold:.3f}")

# Method 2: Business-driven threshold
# If cost of missing default is 10x cost of false alarm:
# Find threshold that maximizes: 10*TPR - FPR

# Method 3: Youden's J statistic (max sensitivity + specificity - 1)
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
```

**Pros:**
- Simple to implement
- No retraining needed
- Directly optimizes business metric

**Cons:**
- Doesn't improve AUC (AUC is threshold-independent)
- Must choose business objective carefully

**Expected Impact:** Improves precision/recall trade-off, but AUC unchanged

---

### Strategy 4: Cost-Sensitive Learning

**Recommendation:** **Combine with Strategy 1**

**How it works:**
- Assign custom sample weights
- Can vary by more than just class label

**Advanced Implementation:**
```python
# Simple version (similar to scale_pos_weight)
sample_weights = np.where(y_train == 1, 19, 1)

# Advanced version (weight by prediction difficulty)
# Step 1: Train initial model
model_initial = XGBClassifier(...).fit(X_train, y_train)
initial_probs = model_initial.predict_proba(X_train)[:, 1]

# Step 2: Weight hard-to-classify examples more
# Default examples close to 0.5 are hard → higher weight
weights = np.ones(len(y_train))
weights[y_train == 1] = 19  # Base weight for defaults
hard_defaults = (y_train == 1) & (initial_probs < 0.7)
weights[hard_defaults] = 30  # Extra weight for hard defaults

# Step 3: Retrain with custom weights
model_final = XGBClassifier(...).fit(
    X_train, y_train,
    sample_weight=weights
)
```

**Expected Impact:** Marginal improvement (0.01-0.02 AUC)

---

### Strategy 5: Ensemble with Balanced Subsampling

**Recommendation:** **Advanced technique if others insufficient**

**How it works:**
- Create multiple balanced subsets
- Train separate models on each
- Average predictions

**Implementation:**
```python
from sklearn.ensemble import BaggingClassifier

# Undersample majority class in each bootstrap sample
model = BaggingClassifier(
    base_estimator=XGBClassifier(...),
    n_estimators=10,
    max_samples=0.5,  # Use 50% of data per model
    bootstrap=True,
    random_state=42
)

# Alternative: BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

model = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='all',  # Balance each tree's bootstrap
    random_state=42
)
```

**Expected Impact:** AUC improvement of 0.02-0.03

---

### Strategy 6: Focal Loss (Advanced)

**Recommendation:** **Research-level approach**

**How it works:**
- Downweights loss for easy examples
- Focuses learning on hard examples
- Used in computer vision for extreme imbalance

**Implementation (requires custom XGBoost objective):**
```python
def focal_loss(y_pred, y_true):
    gamma = 2.0
    alpha = 0.25

    y_true = y_true.get_label()
    p = 1 / (1 + np.exp(-y_pred))  # sigmoid

    # Focal loss formula
    loss = -alpha * y_true * ((1 - p) ** gamma) * np.log(p + 1e-7) - \
           (1 - alpha) * (1 - y_true) * (p ** gamma) * np.log(1 - p + 1e-7)

    # Gradient and hessian for XGBoost
    grad = ...  # Complex derivative
    hess = ...  # Second derivative

    return grad, hess

model = XGBClassifier(objective=focal_loss, ...)
```

**Complexity:** High - requires deep understanding of gradient boosting

---

## Recommended Implementation Plan

### Phase 1: Optimize Current Approach (1 hour)

```python
# Tune scale_pos_weight empirically
weights_to_test = [10, 15, 18.59, 20, 25, 30]
results = []

for weight in weights_to_test:
    model = XGBClassifier(scale_pos_weight=weight, ...)
    cv_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    results.append({'weight': weight, 'auc': cv_auc})

best_weight = max(results, key=lambda x: x['auc'])['weight']
```

**Expected improvement:** +0.005-0.01 AUC

---

### Phase 2: Apply SMOTE (1-2 hours)

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

# Cross-validation with SMOTE
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in cv.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Apply SMOTE only to training fold
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_fold, y_train_fold)

    # Train model
    model = XGBClassifier(scale_pos_weight=2.0, ...)  # Lower weight
    model.fit(X_train_sm, y_train_sm)

    # Evaluate on original validation fold
    y_pred = model.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, y_pred)
    cv_scores.append(auc)

print(f"SMOTE CV AUC: {np.mean(cv_scores):.4f}")
```

**Expected improvement:** +0.02-0.04 AUC

---

### Phase 3: Ensemble Methods (2 hours)

```python
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier

# Three models with SMOTE
model1 = XGBClassifier(scale_pos_weight=2.0, max_depth=4, ...)
model2 = LGBMClassifier(scale_pos_weight=2.0, max_depth=5, ...)
model3 = RandomForestClassifier(class_weight='balanced', ...)

ensemble = VotingClassifier(
    estimators=[('xgb', model1), ('lgbm', model2), ('rf', model3)],
    voting='soft',  # Average probabilities
    weights=[0.4, 0.3, 0.3]
)

# Train on SMOTE-balanced data
X_sm, y_sm = smote.fit_resample(X_train, y_train)
ensemble.fit(X_sm, y_sm)
```

**Expected improvement:** +0.01-0.03 AUC

---

### Phase 4: Threshold Optimization (30 min)

```python
# Find optimal threshold for validation set
from sklearn.metrics import precision_recall_curve

y_pred_proba = model.predict_proba(X_val)[:, 1]

# Calculate precision and recall at different thresholds
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

# Find threshold where precision ≈ recall (balanced)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Precision: {precisions[optimal_idx]:.3f}")
print(f"Recall: {recalls[optimal_idx]:.3f}")
```

**Expected improvement:** Better precision/recall, AUC unchanged

---

## Cumulative Expected Impact

| Phase | Technique | Time | AUC Gain | Cumulative AUC |
|-------|-----------|------|----------|----------------|
| Baseline | Current approach | - | - | 0.7818 |
| Phase 1 | Tune scale_pos_weight | 1h | +0.008 | 0.7898 |
| Phase 2 | SMOTE | 1-2h | +0.025 | 0.8148 |
| Phase 3 | Ensemble | 2h | +0.015 | 0.8298 |
| Phase 4 | Threshold tuning | 0.5h | +0.000 | 0.8298 |

**Total expected AUC: 0.82-0.83** (exceeds 0.80 target)

---

## Evaluation Metrics for Imbalanced Data

### Metrics to Track

1. **AUC-ROC** (primary metric)
   - Threshold-independent
   - Measures ranking quality
   - Current: 0.7818, Target: > 0.80

2. **GINI Coefficient**
   - GINI = 2 × AUC - 1
   - Credit scoring industry standard
   - Target: > 0.60

3. **Precision-Recall AUC**
   - Better for severe imbalance than ROC-AUC
   - Focuses on minority class performance

4. **F1 Score**
   - Harmonic mean of precision and recall
   - Use to optimize threshold

5. **Confusion Matrix**
   ```
               Predicted
               No    Yes
   Actual No   TN    FP
   Actual Yes  FN    TP
   ```
   - Track FN rate (missed defaults) - most costly

### Metrics to AVOID

1. **Accuracy** - Misleading with imbalance
   - 95% accuracy = always predict "no default"

2. **Macro-average F1** - Treats classes equally
   - Use weighted-average F1 instead

---

## Business Considerations

### Cost Matrix

| Actual/Predicted | Predict No Default | Predict Default |
|------------------|-------------------|-----------------|
| **No Default** | 0 (correct) | C1 (investigation cost) |
| **Default** | C2 (loan loss) | 0 (prevented) |

**Typical credit scoring:**
- C1 = $100 (cost of additional review)
- C2 = $10,000 (average loan loss)
- Cost ratio: C2/C1 = 100:1

**Optimal threshold:**
```python
# Threshold that minimizes expected cost
cost_ratio = 100  # C2/C1
optimal_threshold = 1 / (1 + cost_ratio)  # ≈ 0.01

# Very low threshold = flag many for review
# Acceptable if review cost << default cost
```

---

## Validation Checklist

- [ ] Apply SMOTE only to training set, never test/validation
- [ ] Use stratified K-fold to preserve class balance in folds
- [ ] Evaluate on original (non-SMOTE) validation set
- [ ] Track both AUC and precision-recall curves
- [ ] Monitor overfitting gap (train AUC vs CV AUC)
- [ ] Test threshold optimization on separate holdout set
- [ ] Document all hyperparameters and random seeds
- [ ] Save model artifacts for reproducibility

---

## Code Template: Complete Pipeline

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# Define pipeline (SMOTE + model)
pipeline = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.3, random_state=42)),
    ('model', XGBClassifier(
        scale_pos_weight=2.0,
        max_depth=5,
        learning_rate=0.04,
        n_estimators=400,
        random_state=42
    ))
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print(f"SMOTE + XGBoost CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train final model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {test_auc:.4f}")
```

---

## References and Resources

1. **SMOTE Paper:** Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
2. **Imbalanced-learn:** https://imbalanced-learn.org/stable/
3. **Cost-sensitive Learning:** Elkan (2001) "The Foundations of Cost-Sensitive Learning"
4. **Focal Loss:** Lin et al. (2017) "Focal Loss for Dense Object Detection"

---

**Document Version:** 1.0
**Last Updated:** 2025-11-15
**Author:** Data Science Specialist
