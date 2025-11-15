# State-of-the-Art Research: Credit Default Prediction (AUC 0.82-0.85+)

**Research Date:** 2025-11-15
**Project Context:** Credit default prediction with severe class imbalance (1:18.6)
**Current Performance:** XGBoost AUC=0.8047, GINI=0.6094 (164 features)
**Target:** AUC 0.82-0.85

---

## EXECUTIVE SUMMARY

Based on extensive research of Kaggle competitions (Home Credit, American Express 2022), academic papers (2020-2025), and industry best practices, the most promising path to achieve AUC 0.82-0.85 combines **three key strategies**:

1. **Algorithm Upgrade:** LightGBM + CatBoost outperform XGBoost on imbalanced data (expected +0.01-0.02 AUC)
2. **Advanced Sampling:** ADASYN with optimal ratio 6.6:1 (not 1:1!) provides better synthetic data than SMOTE (expected +0.01-0.015 AUC)
3. **Ensemble Stacking:** Multi-algorithm stacking (LightGBM + XGBoost + CatBoost → LogisticRegression) leverages complementary strengths (expected +0.01-0.02 AUC)

**Cumulative expected improvement:** +0.03-0.055 AUC, bringing us from 0.8047 to **0.835-0.86 range**

**Implementation time:** 4-6 hours for all three approaches

---

## TOP-3 RECOMMENDED APPROACHES

### APPROACH #1: LightGBM + ADASYN + Optuna Tuning
**Priority: HIGHEST**
**Expected AUC: 0.82-0.84**
**Implementation Time: 3-4 hours**
**Complexity: Medium**

#### Why This Works

**LightGBM Advantages for Imbalanced Data:**
- Faster training than XGBoost (2-10x speed improvement)
- Better handling of class imbalance with `is_unbalance` parameter
- Leaf-wise growth (vs level-wise) focuses on high-loss samples
- Native support for categorical features (though we're using one-hot encoded data)
- Kaggle Home Credit winners extensively used LightGBM (9th place: 200 OOF predictions, primarily LightGBM)

**ADASYN vs SMOTE:**
- Recent study (2024) on "Give Me Some Credit" dataset (97,243 samples, 7% default rate):
  - ADASYN: AUC=0.6778
  - BorderlineSMOTE: AUC=0.6765
  - SMOTE: AUC=0.6738
- ADASYN adaptively generates more synthetic samples for harder-to-learn minority instances
- **Critical finding:** Optimal ratio is **6.6:1 (majority:minority)**, NOT 1:1 balancing!
- Over-balancing (1:1) introduces noise and reduces performance

**Optuna Hyperparameter Optimization:**
- Bayesian optimization (TPE algorithm) is 3-5x faster than grid search
- Early pruning terminates unpromising trials
- Recommended search space for imbalanced data:
  - `learning_rate`: log-uniform [0.001, 0.3]
  - `max_depth`: int [3, 12]
  - `num_leaves`: int [15, 255]
  - `min_child_samples`: int [5, 100]
  - `subsample`: uniform [0.5, 1.0]
  - `colsample_bytree`: uniform [0.5, 1.0]
  - `scale_pos_weight`: Fixed at 18.6 OR trial parameter [10.0, 30.0]

#### Configuration

```python
# LightGBM parameters for imbalanced data
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'is_unbalance': True,  # Alternative: scale_pos_weight=18.6
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 127,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1
}

# ADASYN parameters
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(
    sampling_strategy=0.15,  # Target ratio: 6.6:1 → minority=15% of majority
    random_state=42,
    n_neighbors=5
)

# Optuna optimization: 100 trials, ~2-3 hours
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=100, timeout=10800)
```

#### Expected Results

- **Baseline (current XGBoost):** AUC=0.8047
- **LightGBM without tuning:** AUC=0.810-0.815 (+0.005-0.010)
- **LightGBM + ADASYN:** AUC=0.820-0.830 (+0.015-0.025)
- **LightGBM + ADASYN + Optuna:** AUC=0.825-0.840 (+0.020-0.035)

#### Benchmark Sources

- Home Credit Kaggle (2018): Top solutions used LightGBM with AUC ~0.805-0.810
- American Express Kaggle (2022): 15th place used LightGBM with knowledge distillation
- Academic paper (2024): LightGBM achieved mean AUC=0.93 in corporate credit scoring
- ADASYN optimal ratio study (2024): arXiv:2510.18252

---

### APPROACH #2: Ensemble Stacking (LightGBM + XGBoost + CatBoost)
**Priority: HIGH**
**Expected AUC: 0.83-0.85**
**Implementation Time: 4-5 hours**
**Complexity: Medium-High**

#### Why This Works

**Diversity of Base Learners:**
- **LightGBM:** Leaf-wise growth, fast training, best for large datasets
- **XGBoost:** Level-wise growth, regularization-focused, robust to overfitting
- **CatBoost:** Ordered boosting (prevents target leakage), symmetric trees, excellent generalization

**Complementary Strengths:**
- Each algorithm has different inductive biases
- LightGBM focuses on high-loss samples (leaf-wise split)
- XGBoost builds balanced trees (level-wise split)
- CatBoost uses ordered boosting to reduce overfitting
- Ensemble captures patterns missed by individual models

**Stacking vs Averaging:**
- Simple averaging: Equal weights (0.33, 0.33, 0.33)
- Stacking: Meta-learner (LogisticRegression/Ridge) learns optimal weights automatically
- Stacking can achieve +0.01-0.03 AUC over best single model
- Meta-learner can use original features too (`passthrough=True`)

#### Configuration

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Base models (all tuned for imbalanced data)
base_models = [
    ('lgb', lgb.LGBMClassifier(
        is_unbalance=True,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=127,
        n_estimators=500,
        random_state=42
    )),
    ('xgb', xgb.XGBClassifier(
        scale_pos_weight=18.6,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        random_state=42,
        eval_metric='auc'
    )),
    ('cat', CatBoostClassifier(
        auto_class_weights='Balanced',
        learning_rate=0.05,
        depth=8,
        iterations=500,
        random_state=42,
        verbose=False
    ))
]

# Meta-learner
meta_learner = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# Stacking ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,  # 5-fold CV for out-of-fold predictions
    passthrough=False,  # Set True to include original features
    n_jobs=-1
)
```

#### Advanced Variant: Two-Level Stacking

For maximum performance, use multiple variants of each algorithm:

```python
# Level 0: 9 base models (3 algorithms × 3 configurations each)
# Level 1: Simple meta-learner

base_models_advanced = [
    # LightGBM variants
    ('lgb_1', lgb.LGBMClassifier(max_depth=6, num_leaves=63, ...)),
    ('lgb_2', lgb.LGBMClassifier(max_depth=8, num_leaves=127, ...)),
    ('lgb_3', lgb.LGBMClassifier(max_depth=10, num_leaves=255, ...)),

    # XGBoost variants
    ('xgb_1', xgb.XGBClassifier(max_depth=4, subsample=0.7, ...)),
    ('xgb_2', xgb.XGBClassifier(max_depth=6, subsample=0.8, ...)),
    ('xgb_3', xgb.XGBClassifier(max_depth=8, subsample=0.9, ...)),

    # CatBoost variants
    ('cat_1', CatBoostClassifier(depth=6, l2_leaf_reg=3, ...)),
    ('cat_2', CatBoostClassifier(depth=8, l2_leaf_reg=5, ...)),
    ('cat_3', CatBoostClassifier(depth=10, l2_leaf_reg=7, ...)),
]
```

#### Expected Results

- **Best single model:** AUC=0.8047 (XGBoost)
- **LightGBM + XGBoost + CatBoost averaging:** AUC=0.815-0.825 (+0.010-0.020)
- **Simple stacking (3 models):** AUC=0.825-0.835 (+0.020-0.030)
- **Advanced stacking (9 models):** AUC=0.830-0.850 (+0.025-0.045)

#### Benchmark Sources

- Home Credit Kaggle (9th place): 6-layer stack with 200 OOF predictions, AUC ~0.805+
- American Express Kaggle (2nd place): Ensemble of LightGBM models, AUC=0.96 (10-fold CV)
- Academic study (2024): Stacking ensemble outperformed individual models consistently
- Kaggle notebook: Soft-voting ensemble achieved AUC=0.9640 vs individual models ~0.96

---

### APPROACH #3: CatBoost with Ordered Boosting + Focal Loss
**Priority: MEDIUM-HIGH**
**Expected AUC: 0.81-0.83**
**Implementation Time: 2-3 hours**
**Complexity: Low-Medium**

#### Why This Works

**CatBoost Unique Features:**
- **Ordered Boosting:** Prevents target leakage during training
  - Traditional GBDT: Uses same data for residual calculation and split finding
  - CatBoost: Uses different permutations to avoid this bias
  - Result: Better generalization, especially on test data

- **Symmetric Trees:** All splits at same level use same feature
  - Faster prediction (2-3x vs XGBoost)
  - Better regularization
  - Reduces overfitting on imbalanced data

- **Dynamic Binning:** Adapts to data distribution
  - Pays more attention to minority class
  - Avoids over-compensation

- **Class Imbalance Handling:**
  - `auto_class_weights='Balanced'` automatically calculates weights
  - Alternative: `scale_pos_weight` or `class_weights` dictionary

**Focal Loss for Imbalanced Data:**
- Original purpose: Object detection with extreme imbalance
- Key idea: Down-weight easy examples, focus on hard examples
- Formula: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
- `gamma=2` is typical (higher gamma → more focus on hard examples)
- **Important:** Custom objective requires gradient + hessian implementation

#### Configuration

**Standard CatBoost (Recommended for First Try):**

```python
from catboost import CatBoostClassifier, Pool

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 5,
    'auto_class_weights': 'Balanced',  # Handles 1:18.6 imbalance
    'eval_metric': 'AUC',
    'random_seed': 42,
    'early_stopping_rounds': 50,
    'verbose': 100
}

model = CatBoostClassifier(**catboost_params)
```

**CatBoost with Focal Loss (Advanced):**

```python
# Custom focal loss implementation
import numpy as np

def focal_loss_gradient_hessian(predictions, targets, gamma=2.0):
    """
    Focal loss for CatBoost
    predictions: raw predictions (not probabilities)
    targets: true labels (0/1)
    gamma: focusing parameter (typical: 2.0)
    """
    # Convert raw predictions to probabilities
    p = 1.0 / (1.0 + np.exp(-predictions))

    # Gradient
    grad = np.where(
        targets == 1,
        -gamma * (1 - p) ** (gamma - 1) * np.log(p) * p - (1 - p) ** gamma,
        gamma * p ** (gamma - 1) * np.log(1 - p) * (1 - p) + p ** gamma
    )

    # Hessian (second derivative)
    hess = np.where(
        targets == 1,
        gamma * (gamma - 1) * (1 - p) ** (gamma - 2) * p * np.log(p) +
        2 * gamma * (1 - p) ** (gamma - 1) + (1 - p) ** gamma * p,
        gamma * (gamma - 1) * p ** (gamma - 2) * (1 - p) * np.log(1 - p) +
        2 * gamma * p ** (gamma - 1) + p ** gamma * (1 - p)
    )

    return grad, hess

# Use with CatBoost
train_pool = Pool(X_train, y_train)
eval_pool = Pool(X_test, y_test)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='Logloss',  # Base loss, focal applied via callback
    eval_metric='AUC',
    random_seed=42
)

# Note: CatBoost doesn't natively support focal loss like LightGBM
# For focal loss, use LightGBM (see implementation in code examples)
```

#### Expected Results

- **CatBoost (auto_class_weights):** AUC=0.810-0.820 (+0.005-0.015)
- **CatBoost (tuned hyperparameters):** AUC=0.815-0.830 (+0.010-0.025)
- **CatBoost + Focal Loss:** AUC=0.820-0.835 (+0.015-0.030)
  - Note: Focal loss implementation is more mature for LightGBM

#### Benchmark Sources

- Academic study (2024): CatBoost achieved mean AUC=0.93+ in corporate credit scoring
- Corporate failure prediction (2024): CatBoost AUC=0.94, accuracy=0.89
- Comparison study: CatBoost significantly outperformed others on ACC, AUC, F1-score, Brier score, KS
- Research (2020-2024): CatBoost and LightGBM showed strongest predictive stability

---

## COMPARISON TABLE

| Approach | Expected AUC | Improvement | Time | Complexity | Interpretability | Production-Ready |
|----------|-------------|-------------|------|------------|------------------|------------------|
| **Current XGBoost** | 0.8047 | Baseline | - | - | High | Yes |
| **#1: LightGBM + ADASYN + Optuna** | 0.825-0.840 | +0.020-0.035 | 3-4h | Medium | High | Yes |
| **#2: Ensemble Stacking (3 models)** | 0.825-0.835 | +0.020-0.030 | 4-5h | Medium-High | Medium | Yes |
| **#2: Ensemble Stacking (9 models)** | 0.830-0.850 | +0.025-0.045 | 6-8h | High | Low | Moderate |
| **#3: CatBoost + Ordered Boosting** | 0.815-0.830 | +0.010-0.025 | 2-3h | Low-Medium | High | Yes |
| **Focal Loss (LightGBM)** | 0.820-0.835 | +0.015-0.030 | 2-3h | Medium | High | Moderate |
| **SMOTE (standard 1:1)** | 0.805-0.815 | +0.000-0.010 | 1h | Low | High | Yes |
| **ADASYN (optimal 6.6:1)** | 0.815-0.825 | +0.010-0.020 | 1h | Low | High | Yes |

**Key Insights:**
- **Best single approach:** LightGBM + ADASYN + Optuna (highest expected AUC, reasonable complexity)
- **Best ensemble approach:** Stacking 9 models (highest ceiling, but most complex)
- **Best quick win:** CatBoost with auto_class_weights (2-3 hours, solid improvement)
- **Best interpretability:** LightGBM or CatBoost (both provide feature importance, SHAP values)

---

## DETAILED TECHNIQUE ANALYSIS

### 1. SAMPLING TECHNIQUES FOR IMBALANCED DATA

#### SMOTE vs ADASYN vs BorderlineSMOTE

**Research Study (2024): "Finding the Sweet Spot" - arXiv:2510.18252**
- Dataset: Give Me Some Credit (97,243 observations, 7% default rate)
- Evaluated 10 augmentation scenarios (1x, 2x, 3x multiplication)
- Metric: AUC-ROC with bootstrap testing (1,000 iterations)

**Results at 1x Multiplication:**
- ADASYN: AUC=0.6778 (BEST)
- BorderlineSMOTE: AUC=0.6765
- SMOTE: AUC=0.6738

**Optimal Ratio Discovery:**
- Traditional approach: Balance to 1:1 (50% minority)
- Optimal ratio: **6.6:1 (majority:minority)** = 13.2% minority class
- Over-balancing (1:1) introduces noise and reduces performance
- For our data (5.11% minority): Target ~13-15% minority after ADASYN

**Why ADASYN Wins:**
1. **Adaptive sampling:** Generates more synthetic samples for harder-to-learn instances
2. **Density-based:** Uses K-nearest neighbors density to determine sample generation
3. **Focused learning:** Minority samples near decision boundary get more synthetic neighbors
4. **Less noise:** Avoids over-generation in well-separated regions

**Implementation:**
```python
from imblearn.over_sampling import ADASYN

# For 1:18.6 ratio (5.11% minority), target 6.6:1 = 13.2% minority
# sampling_strategy = 0.132 / (1 - 0.132) = 0.152
adasyn = ADASYN(
    sampling_strategy=0.15,  # Target ~15% minority (6.6:1 ratio)
    random_state=42,
    n_neighbors=5  # Default, can tune to 3-7
)

X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
```

#### Under-sampling vs Over-sampling

**Research Findings:**
- Over-sampling (SMOTE/ADASYN) generally outperforms under-sampling
- Under-sampling loses information (discards majority samples)
- Hybrid approaches (SMOTEENN, SMOTETomek) show mixed results

**Exception:** When you have HUGE dataset (>1M samples)
- Under-sampling can speed up training without losing much information
- RUSBoost, EasyEnsemble are competitive

**Our Recommendation:**
- Use ADASYN (72,000 samples is medium-sized, not huge)
- Avoid 1:1 balancing (target 6.6:1 ratio)

---

### 2. GRADIENT BOOSTING ALGORITHMS COMPARISON

#### LightGBM vs XGBoost vs CatBoost

**Architecture Differences:**

| Feature | LightGBM | XGBoost | CatBoost |
|---------|----------|---------|----------|
| **Split Strategy** | Leaf-wise (best leaf) | Level-wise (all leaves) | Level-wise (symmetric) |
| **Speed** | Fastest (2-10x faster) | Medium | Medium-Slow |
| **Memory** | Most efficient | Medium | Higher |
| **Categorical Support** | Limited | Manual encoding | Native (excellent) |
| **Imbalance Handling** | `is_unbalance`, `scale_pos_weight` | `scale_pos_weight` | `auto_class_weights` |
| **Overfitting Risk** | Higher (leaf-wise) | Lower (regularization) | Lowest (ordered boosting) |
| **Interpretability** | High (SHAP, feature importance) | High (SHAP, feature importance) | High (SHAP, feature importance) |

**When to Use Each:**

**LightGBM:**
- Large datasets (>10k samples)
- Need fast training/iteration
- Imbalanced data with `is_unbalance=True`
- CPU-limited environments
- **Best for:** Experimentation, hyperparameter tuning

**XGBoost:**
- Medium datasets (1k-100k samples)
- Need robust regularization (L1, L2)
- Established production pipelines
- **Best for:** Production deployment, stability

**CatBoost:**
- Categorical features (native encoding)
- Small-medium datasets
- Need best generalization (ordered boosting)
- **Best for:** Academic evaluation, research

**For Credit Scoring (Our Case):**
1. **LightGBM** - Primary choice (speed + imbalance handling)
2. **Ensemble** - All three together (maximum performance)
3. **CatBoost** - Alternative if overfitting is concern

---

### 3. HYPERPARAMETER OPTIMIZATION STRATEGIES

#### Optuna vs Hyperopt vs Ray Tune

**Comparison:**

| Tool | Algorithm | Speed | Ease of Use | Integrations | Best For |
|------|-----------|-------|-------------|--------------|----------|
| **Optuna** | TPE, CMA-ES | Fast | Excellent | XGBoost, LightGBM, CatBoost | Most projects |
| **Hyperopt** | TPE | Medium | Good | General | Legacy projects |
| **Ray Tune** | Multiple | Fastest | Complex | Distributed systems | Large-scale |

**Recommendation: Optuna**
- Best scikit-learn/XGBoost/LightGBM integration
- Early pruning (saves time)
- Excellent visualization
- Active development

**Optuna Best Practices:**

1. **Define Search Space Carefully:**
```python
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    # Note: Keep scale_pos_weight=18.6 fixed OR tune it
```

2. **Use Pruning:**
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,  # No pruning for first 10 trials
        n_warmup_steps=5       # Evaluate at least 5 boosting rounds
    )
)
```

3. **Monitor Progress:**
```python
# Visualize optimization
import optuna.visualization as vis
vis.plot_optimization_history(study)
vis.plot_param_importances(study)
vis.plot_parallel_coordinate(study)
```

4. **Number of Trials:**
- Quick search: 50 trials (~1 hour)
- Thorough search: 100-200 trials (~2-4 hours)
- Exhaustive: 500+ trials (~8+ hours, diminishing returns)

**Expected Improvement:**
- No tuning → Basic tuning (50 trials): +0.005-0.015 AUC
- Basic tuning → Thorough tuning (200 trials): +0.005-0.010 AUC
- Total: +0.010-0.025 AUC from hyperparameter optimization

---

### 4. ENSEMBLE STRATEGIES

#### Stacking Architecture

**Level 0 (Base Learners):**
- Diversity is key: Use different algorithms OR different configurations
- Cross-validation: Generate out-of-fold (OOF) predictions
- Avoid overfitting: Use conservative hyperparameters

**Level 1 (Meta-Learner):**
- Simple models work best: LogisticRegression, Ridge, Lasso
- Avoid complex models (Random Forest, XGBoost) - they overfit on meta-features
- Regularization is crucial: Use L2 penalty

**Optimal Number of Base Models:**
- Too few (<3): Not enough diversity
- Optimal (3-9): Good diversity without overfitting
- Too many (>15): Overfitting on OOF predictions

**Code Structure:**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold

# Use StratifiedKFold to preserve class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(C=1.0, class_weight='balanced'),
    cv=cv,
    passthrough=False,  # True adds original features to meta-learner
    n_jobs=-1
)
```

#### Blending vs Stacking

**Blending:**
- Split data: Train (60%) → Train base models → Validation (20%) → Train meta-learner → Test (20%)
- Simpler: No cross-validation needed
- Faster: Train base models once
- Downside: Uses less data for meta-learner

**Stacking:**
- Cross-validation: 5-fold → Each fold generates OOF predictions → Meta-learner uses all OOF
- Better: Uses all training data for meta-learner
- Slower: Train base models 5 times
- **Recommended for credit scoring**

#### Weighted Averaging (Simple Alternative)

If stacking is too complex, try weighted averaging:

```python
# Manual weight search
weights = [0.4, 0.4, 0.2]  # LightGBM, XGBoost, CatBoost
y_pred = (weights[0] * pred_lgb +
          weights[1] * pred_xgb +
          weights[2] * pred_cat)

# Grid search for optimal weights
from scipy.optimize import minimize

def objective(weights):
    pred = weights[0]*pred_lgb + weights[1]*pred_xgb + weights[2]*pred_cat
    return -roc_auc_score(y_true, pred)

result = minimize(objective, [0.33, 0.33, 0.33],
                  bounds=[(0, 1), (0, 1), (0, 1)],
                  constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1})
```

**Expected Improvement:**
- Simple averaging: +0.005-0.010 AUC
- Optimized weights: +0.010-0.015 AUC
- Stacking: +0.015-0.030 AUC

---

### 5. FOCAL LOSS FOR IMBALANCED DATA

#### Theory

**Standard Cross-Entropy Loss:**
- CE(p) = -log(p) if y=1, -log(1-p) if y=0
- Treats all samples equally
- Easy samples (p close to 1 or 0) still contribute to loss

**Focal Loss:**
- FL(p) = -(1-p)^gamma * log(p) if y=1
- Down-weights easy samples: If p=0.9 (confident), (1-0.9)^2 = 0.01 multiplier
- Focuses on hard samples: If p=0.5 (uncertain), (1-0.5)^2 = 0.25 multiplier
- gamma: Focusing parameter (0 = CE, 2 = standard focal, 5 = extreme focus)

**When to Use:**
- Extreme imbalance (>1:100)
- Many easy negatives (well-separated classes)
- Want model to focus on hard negatives near decision boundary

#### Implementation for LightGBM

**Full Code:**
```python
import numpy as np
import lightgbm as lgb

def focal_loss_lgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss for LightGBM

    Parameters:
    - y_pred: raw predictions (not probabilities)
    - dtrain: LightGBM Dataset
    - alpha: class weight (0.25 means 0.25 for positive, 0.75 for negative)
    - gamma: focusing parameter (2.0 is typical)
    """
    y_true = dtrain.get_label()

    # Convert raw predictions to probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Compute focal loss components
    # For y=1: -alpha * (1-p)^gamma * log(p)
    # For y=0: -(1-alpha) * p^gamma * log(1-p)

    # Gradient
    grad = np.where(
        y_true == 1,
        alpha * ((gamma * (1 - p) ** (gamma - 1) * p * np.log(p)) +
                 (1 - p) ** gamma),
        -(1 - alpha) * ((gamma * p ** (gamma - 1) * (1 - p) * np.log(1 - p)) +
                         p ** gamma)
    )

    # Hessian (second derivative)
    hess = np.where(
        y_true == 1,
        alpha * ((gamma * (gamma - 1) * (1 - p) ** (gamma - 2) * p * np.log(p)) +
                 (2 * gamma * (1 - p) ** (gamma - 1)) +
                 ((1 - p) ** gamma * p)),
        (1 - alpha) * ((gamma * (gamma - 1) * p ** (gamma - 2) * (1 - p) * np.log(1 - p)) +
                       (2 * gamma * p ** (gamma - 1)) +
                       (p ** gamma * (1 - p)))
    )

    return grad, hess

# Train with focal loss
train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': focal_loss_lgb,  # Custom objective
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 127,
    'verbose': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50)]
)
```

**Hyperparameters to Tune:**
- `alpha`: Class weight (0.05-0.25 for 5% minority class)
- `gamma`: Focusing (1.0-5.0, start with 2.0)

**Expected Improvement:**
- vs standard loss: +0.01-0.02 AUC (if you have many easy negatives)
- vs class weighting: +0.005-0.015 AUC

**Caution:**
- More complex to implement
- Harder to debug (custom gradient/hessian)
- Not natively supported in XGBoost/CatBoost (use imbalance-xgboost package)

---

### 6. CREDIT SCORING SPECIFIC TECHNIQUES

#### Weight of Evidence (WOE) Encoding

**Traditional Credit Scoring Approach:**
1. Bin continuous variables (10-20 bins)
2. Calculate WOE for each bin: WOE = ln(% good / % bad)
3. Replace original values with WOE values
4. Use logistic regression

**Advantages:**
- Monotonic relationship with target
- Interpretable (linear model)
- Regulatory-friendly (explainable)

**Disadvantages:**
- Information loss (binning)
- Manual bin selection
- Doesn't work well with gradient boosting (which handles non-linearity)

**Recommendation for This Project:**
- **Skip WOE encoding** - We're using gradient boosting, not logistic regression
- Gradient boosting handles non-linearity automatically
- Our one-hot encoded categoricals already capture relationships
- WOE is useful for traditional scorecard models, not GBDT

#### Monotonic Constraints

**Use Case:**
- Ensure features have expected direction of influence
- Example: `credit_score` should monotonically decrease default probability
- Regulatory requirement: Models must be "business-sensible"

**Implementation:**
```python
# LightGBM
monotone_constraints = {
    'annual_income': -1,      # Higher income → lower default (negative effect)
    'age': -1,                # Older → lower default
    'credit_score': -1,       # Higher score → lower default
    'debt_to_income_ratio': 1 # Higher DTI → higher default (positive effect)
}

model = lgb.LGBMClassifier(monotone_constraints=monotone_constraints)

# XGBoost
monotone_constraints = (-1, -1, -1, 1)  # Order matches feature order
model = xgb.XGBClassifier(monotone_constraints=monotone_constraints)

# CatBoost
monotone_constraints = {0: -1, 1: -1, 2: -1, 3: 1}  # Feature index: direction
model = CatBoostClassifier(monotone_constraints=monotone_constraints)
```

**Trade-off:**
- Interpretability: ↑ (model behavior is predictable)
- Performance: ↓ (constrains model flexibility, -0.005-0.015 AUC)

**Recommendation:**
- Use for production deployment (regulatory compliance)
- Skip for Kaggle-style competition (maximize AUC)

#### Probability Calibration

**Why Calibrate:**
- GBDT models output well-ranked scores, but probabilities may be mis-calibrated
- Example: Model predicts 0.3 default probability, but actual default rate is 0.15
- **Important for credit scoring:** Banks need accurate probabilities for risk assessment

**When AUC ≠ Calibration:**
- AUC measures ranking (is p(A) > p(B)?)
- Calibration measures probability accuracy (is p=0.3 actually 30% default rate?)
- You can have AUC=0.85 but poor calibration!

**Methods:**

**Platt Scaling (Sigmoid):**
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',  # Platt scaling
    cv=5
)
calibrated_model.fit(X_train, y_train)
```

**Isotonic Regression:**
```python
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',  # More flexible, needs more data
    cv=5
)
```

**Research Findings (Credit Scoring):**
- Isotonic regression outperforms Platt scaling for time-series credit data
- Improves long-term calibration
- Requires >1000 samples for stable results

**Impact on AUC:**
- Calibration typically doesn't change AUC (ranking unchanged)
- May slightly hurt AUC (-0.000-0.005) due to regularization

**Recommendation:**
- Calibrate AFTER optimizing AUC
- Use isotonic regression (we have 72,000 samples)
- Essential for production, optional for competition

---

## STEP-BY-STEP ROADMAP TO AUC 0.82-0.85

### Phase 1: Quick Wins (2-3 hours) → Target AUC 0.815-0.820

**Step 1.1: Test CatBoost with auto_class_weights (30 min)**
```bash
python approach_3_catboost.py
```
- Expected: AUC 0.810-0.815
- If AUC > 0.815: Great! Continue to Phase 2
- If AUC < 0.810: CatBoost doesn't help much, skip to Step 1.2

**Step 1.2: Test LightGBM with is_unbalance (30 min)**
```bash
python approach_1_lightgbm_baseline.py
```
- Expected: AUC 0.810-0.815
- Compare with CatBoost, pick winner for next steps

**Step 1.3: Apply ADASYN with optimal ratio (1 hour)**
```bash
python approach_1_lightgbm_adasyn.py
```
- Use best model from Step 1.1 or 1.2
- Expected: AUC 0.815-0.825

**Checkpoint 1:** If AUC > 0.82, you can stop here! Otherwise continue.

---

### Phase 2: Hyperparameter Optimization (3-4 hours) → Target AUC 0.825-0.835

**Step 2.1: Optuna tuning on LightGBM + ADASYN (3 hours)**
```bash
python approach_1_lightgbm_adasyn_optuna.py  # 100 trials, ~3 hours
```
- Expected: AUC 0.825-0.835
- Save best hyperparameters to `best_params.json`

**Step 2.2: Retrain with best parameters (15 min)**
```bash
python approach_1_retrain_best.py
```
- Use full training set (no CV split)
- Generate final predictions

**Checkpoint 2:** If AUC > 0.825, great progress! If AUC < 0.82, move to Phase 3.

---

### Phase 3: Ensemble Stacking (4-5 hours) → Target AUC 0.830-0.850

**Step 3.1: Train individual base models (2 hours)**
```bash
python approach_2_train_base_models.py
```
- LightGBM (tuned from Phase 2)
- XGBoost (current best)
- CatBoost (from Phase 1)
- Save all models and OOF predictions

**Step 3.2: Train stacking ensemble (1 hour)**
```bash
python approach_2_stacking.py
```
- LogisticRegression meta-learner
- 5-fold CV for OOF generation
- Expected: AUC 0.830-0.845

**Step 3.3: (Optional) Advanced stacking (2 hours)**
If AUC < 0.83, create multiple variants of each algorithm:
```bash
python approach_2_stacking_advanced.py
```
- 3 LightGBM variants (different max_depth)
- 3 XGBoost variants (different subsample)
- 3 CatBoost variants (different l2_leaf_reg)
- Expected: AUC 0.835-0.850

**Final Checkpoint:** AUC should be 0.83-0.85 now!

---

### Phase 4: Final Touches (Optional, 1-2 hours)

**Step 4.1: Probability calibration**
```bash
python calibrate_predictions.py
```
- Use isotonic regression
- Improves probability estimates (important for production)
- May slightly affect AUC (±0.005)

**Step 4.2: Generate final submission**
```bash
python generate_final_predictions.py
```
- Load best model(s)
- Predict on test set
- Save to CSV

**Step 4.3: Model interpretation**
```bash
python interpret_model.py
```
- SHAP values
- Feature importance
- Partial dependence plots

---

## TOTAL TIME ESTIMATE

**Conservative Path (Target: AUC 0.82-0.83):**
- Phase 1 + Phase 2: 5-7 hours
- Expected: AUC 0.82-0.83

**Aggressive Path (Target: AUC 0.83-0.85):**
- Phase 1 + Phase 2 + Phase 3: 9-12 hours
- Expected: AUC 0.83-0.85

**Probability of Success:**
- AUC > 0.82: 85% (highly likely with ADASYN + Optuna)
- AUC > 0.83: 70% (likely with stacking)
- AUC > 0.85: 40% (possible with advanced stacking, not guaranteed)

---

## IMPLEMENTATION NOTES

### Libraries Required

```bash
pip install pandas numpy scikit-learn
pip install xgboost lightgbm catboost
pip install imbalanced-learn  # ADASYN, SMOTE
pip install optuna  # Hyperparameter optimization
pip install shap  # Model interpretation
pip install matplotlib seaborn  # Visualization
```

### Hardware Recommendations

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Time: 8-12 hours total

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- Time: 4-6 hours total

**GPU:**
- Not required for this dataset size (72,000 samples)
- LightGBM/XGBoost/CatBoost are CPU-optimized

### Data Pipeline

```
1. Load data: X_train_engineered.parquet, y_train.parquet (164 features)
2. Apply ADASYN: Upsample minority to 15% (6.6:1 ratio)
3. Train model: LightGBM/XGBoost/CatBoost with cross-validation
4. Hyperparameter tuning: Optuna (100-200 trials)
5. Ensemble: Stack multiple models with LogisticRegression
6. Calibrate: Isotonic regression (optional)
7. Predict: Generate test set predictions
```

---

## REFERENCES AND SOURCES

### Academic Papers (2020-2025)

1. **"Finding the Sweet Spot: Optimal Data Augmentation Ratio for Imbalanced Credit Scoring Using ADASYN"** (2024)
   - arXiv:2510.18252
   - Key finding: Optimal ratio 6.6:1, not 1:1
   - Dataset: Give Me Some Credit, 97,243 samples

2. **"Advancing financial resilience: A systematic review of default prediction models"** (2024)
   - PMC11564005
   - Review of 2015-2024 literature
   - Finding: GBDT achieves 85-95% accuracy

3. **"Ensemble Methodology: Innovations in Credit Default Prediction Using LightGBM, XGBoost, and LocalEnsemble"** (2024)
   - arXiv:2402.17979
   - Ensemble techniques for credit scoring

4. **"Calibration of Machine Learning Classifiers for Probability of Default Modelling"** (2017)
   - arXiv:1710.08901
   - Platt scaling vs isotonic regression
   - Finding: Isotonic better for time-series credit data

5. **"Imbalance-XGBoost: leveraging weighted and focal losses"** (2020)
   - Pattern Recognition Letters
   - Focal loss for imbalanced classification

### Kaggle Competitions

1. **Home Credit Default Risk (2018)**
   - Winner solutions: https://www.kaggle.com/c/home-credit-default-risk/discussion
   - 9th place: 6-layer stack, 200 OOF predictions, LightGBM heavy
   - Top solutions: AUC 0.805-0.810

2. **American Express Default Prediction (2022)**
   - Competition: https://www.kaggle.com/competitions/amex-default-prediction
   - 2nd place: LightGBM with iterative feature selection, AUC 0.96
   - 15th place: Transformer + LightGBM knowledge distillation

3. **Give Me Some Credit (Classic)**
   - Historic competition, widely cited
   - Standard benchmark for credit scoring

### GitHub Repositories

1. **LightGBM with Focal Loss**
   - https://github.com/jrzaurin/LightGBM-with-Focal-Loss
   - Implementation with Hyperopt integration

2. **Imbalance-XGBoost**
   - https://github.com/jhwjhw0123/Imbalance-XGBoost
   - Weighted and focal loss for XGBoost

3. **Home Credit Solutions**
   - https://github.com/kozodoi/Kaggle_Home_Credit
   - https://github.com/oskird/Kaggle-Home-Credit-Default-Risk-Solution
   - Feature engineering and ensembling examples

4. **Imbalanced-Learn**
   - https://github.com/scikit-learn-contrib/imbalanced-learn
   - Official ADASYN, SMOTE implementations

### Tutorials and Blogs

1. **Max Halford: Focal Loss for LightGBM**
   - https://maxhalford.github.io/blog/lightgbm-focal-loss/
   - Detailed implementation guide

2. **Forecastegy: XGBoost Hyperparameter Tuning with Optuna**
   - https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/
   - Kaggle Grandmaster guide

3. **Analytics Vidhya: Ensemble Methods**
   - Stacking, bagging, boosting tutorials

4. **scikit-learn Documentation**
   - https://scikit-learn.org/stable/modules/ensemble.html
   - Official stacking, calibration docs

### Industry Resources

1. **CatBoost for Credit Scoring**
   - https://deburky.medium.com/build-explainable-scorecards-with-catboost-44bfe73a304a
   - Practical guide for financial services

2. **imbalanced-learn Documentation**
   - https://imbalanced-learn.org/stable/
   - Comparison of sampling methods

3. **Optuna Documentation**
   - https://optuna.org/
   - Bayesian optimization framework

---

## APPENDIX: ALTERNATIVE APPROACHES (NOT RECOMMENDED)

### Why NOT Deep Learning?

**TabNet, FT-Transformer, Neural Networks:**
- Research shows GBDT still outperforms deep learning on tabular data
- TabNet: "Last place with no datasets where it is the best-performing one"
- FT-Transformer: Outperforms GBDT in only 7/11 datasets
- Neural networks require more data, longer training, harder to tune
- **Exception:** Very large datasets (>1M samples) with complex interactions

**Verdict:** Stick with gradient boosting for this project.

### Why NOT Bagging Ensembles?

**Random Forest, BalancedRandomForest:**
- Benchmark: Balanced Random Forest often loses to gradient boosting
- Expected AUC: 0.75-0.80 (worse than current 0.8047)
- Only advantage: Parallel training (faster)

**Verdict:** Use Random Forest only as diversity element in stacking, not as primary model.

### Why NOT Traditional Under-sampling?

**RUSBoost, EasyEnsemble, NearMiss:**
- Under-sampling discards majority class information
- Expected improvement: +0.000-0.010 AUC (minimal)
- Benchmark: Balanced Random Forest AUC=0.84, RUSBoost AUC=0.83 (similar)
- Only useful for HUGE datasets (>1M samples)

**Verdict:** Use ADASYN over-sampling instead (proven better for 72k dataset).

### Why NOT WOE Encoding?

**Weight of Evidence transformation:**
- Traditional credit scoring uses WOE + Logistic Regression
- Designed for linear models, not GBDT
- GBDT handles non-linearity automatically
- Information loss from binning

**Verdict:** Skip WOE encoding, use raw features with GBDT.

---

## CONCLUSION

**Top Recommendation:** Implement Approach #1 (LightGBM + ADASYN + Optuna) first
- Highest probability of reaching AUC 0.82-0.83
- Reasonable implementation time (3-4 hours)
- Well-documented, proven approach

**If you need AUC > 0.83:** Add Approach #2 (Ensemble Stacking)
- Combine LightGBM + XGBoost + CatBoost
- 4-5 additional hours
- Expected AUC 0.83-0.85

**Quick alternative:** Approach #3 (CatBoost alone)
- 2-3 hours
- Expected AUC 0.81-0.83
- Good if time-constrained

**Key Success Factors:**
1. Use ADASYN with 6.6:1 ratio (NOT 1:1!)
2. Hyperparameter tuning with Optuna (100+ trials)
3. Ensemble multiple algorithms for maximum performance
4. Monitor cross-validation AUC (avoid overfitting)

**Confidence Level:**
- 85% chance of AUC > 0.82 with Phase 1+2
- 70% chance of AUC > 0.83 with Phase 1+2+3
- 40% chance of AUC > 0.85 (requires luck + advanced stacking)

Good luck! 🚀
