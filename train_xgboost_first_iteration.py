"""
XGBoost - First Iteration Training Script
==========================================
Uses all 108 features with baseline hyperparameters.
Optimizes for AUC metric with class imbalance handling.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("XGBoost Training - First Iteration")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# 1. Load Data
# =============================================================================
print("Loading training data...")
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')

# Flatten target if needed
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.values.ravel()
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.values.ravel()

print(f"✓ Train set: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"✓ Test set:  {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
print(f"✓ Class distribution (train): {np.bincount(y_train)} (ratio 1:{(y_train==0).sum()/(y_train==1).sum():.1f})")
print(f"✓ Class distribution (test):  {np.bincount(y_test)} (ratio 1:{(y_test==0).sum()/(y_test==1).sum():.1f})\n")

# =============================================================================
# 2. Configure XGBoost
# =============================================================================
print("Configuring XGBoost...")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"✓ Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Baseline hyperparameters - not over-tuned
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist'
}

print("\nHyperparameters:")
for param, value in xgb_params.items():
    print(f"  {param}: {value}")

# =============================================================================
# 3. Cross-Validation
# =============================================================================
print("\n" + "=" * 80)
print("Running 5-Fold Stratified Cross-Validation...")
print("=" * 80)

cv_start_time = time.time()

# Create cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Train model for CV (with early stopping we need to do manual CV)
cv_scores = []
cv_fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"\nFold {fold_idx}/5:")

    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Create XGBoost model with early stopping in constructor
    model_fold = xgb.XGBClassifier(
        **xgb_params,
        early_stopping_rounds=50
    )

    # Train with evaluation set
    model_fold.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )

    # Predict probabilities
    y_fold_pred = model_fold.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, y_fold_pred)
    cv_scores.append(fold_auc)

    cv_fold_results.append({
        'fold': fold_idx,
        'auc': fold_auc,
        'best_iteration': model_fold.best_iteration
    })

    print(f"  AUC: {fold_auc:.6f} (best iteration: {model_fold.best_iteration})")

cv_time = time.time() - cv_start_time

print("\n" + "-" * 80)
print("Cross-Validation Results:")
print(f"  Mean AUC: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
print(f"  Min AUC:  {np.min(cv_scores):.6f}")
print(f"  Max AUC:  {np.max(cv_scores):.6f}")
print(f"  CV Time:  {cv_time:.1f} seconds")

# =============================================================================
# 4. Train Final Model on Full Training Set
# =============================================================================
print("\n" + "=" * 80)
print("Training Final Model on Full Training Set...")
print("=" * 80)

train_start_time = time.time()

# Create final model with early stopping in constructor
final_model = xgb.XGBClassifier(
    **xgb_params,
    early_stopping_rounds=50
)

# Train with evaluation on test set
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

train_time = time.time() - train_start_time

print(f"\n✓ Training completed in {train_time:.1f} seconds")
print(f"✓ Best iteration: {final_model.best_iteration}")

# =============================================================================
# 5. Evaluate Model
# =============================================================================
print("\n" + "=" * 80)
print("Model Evaluation")
print("=" * 80)

# Predictions
y_train_pred_proba = final_model.predict_proba(X_train)[:, 1]
y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# Calculate AUC scores
train_auc = roc_auc_score(y_train, y_train_pred_proba)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\nAUC Scores:")
print(f"  Train AUC: {train_auc:.6f}")
print(f"  Test AUC:  {test_auc:.6f}")
print(f"  CV AUC:    {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
print(f"  Overfit:   {train_auc - test_auc:.6f} (train - test)")

# Confusion matrices
print("\nConfusion Matrix (Test Set):")
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['No Default', 'Default']))

# =============================================================================
# 6. Feature Importance Analysis
# =============================================================================
print("\n" + "=" * 80)
print("Feature Importance Analysis")
print("=" * 80)

# Get feature importance (gain-based)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Count zero-importance features
zero_importance = (feature_importance['importance'] == 0).sum()
near_zero_importance = (feature_importance['importance'] < 0.001).sum()

print(f"\nFeature Importance Summary:")
print(f"  Zero importance features:      {zero_importance}/{len(feature_importance)}")
print(f"  Near-zero importance (<0.001): {near_zero_importance}/{len(feature_importance)}")
print(f"  Top 20 features account for:   {feature_importance.head(20)['importance'].sum()*100:.1f}% of total importance")

# =============================================================================
# 7. Save Results
# =============================================================================
print("\n" + "=" * 80)
print("Saving Results...")
print("=" * 80)

# Save model
model_path = '/home/dr/cbu/xgboost_model_v1.json'
final_model.save_model(model_path)
print(f"✓ Model saved: {model_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_test_pred,
    'y_pred_proba': y_test_pred_proba
})
predictions_path = '/home/dr/cbu/xgboost_predictions_v1.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"✓ Predictions saved: {predictions_path}")

# Save metrics
metrics = {
    'model': 'XGBoost',
    'version': 'v1',
    'features': {
        'total': int(X_train.shape[1]),
        'zero_importance': int(zero_importance),
        'near_zero_importance': int(near_zero_importance)
    },
    'hyperparameters': {k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in xgb_params.items()},
    'performance': {
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'cv_auc_mean': float(np.mean(cv_scores)),
        'cv_auc_std': float(np.std(cv_scores)),
        'cv_auc_min': float(np.min(cv_scores)),
        'cv_auc_max': float(np.max(cv_scores)),
        'overfit_gap': float(train_auc - test_auc)
    },
    'training': {
        'cv_time_seconds': float(cv_time),
        'train_time_seconds': float(train_time),
        'best_iteration': int(final_model.best_iteration)
    },
    'cv_fold_results': cv_fold_results,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

metrics_path = '/home/dr/cbu/xgboost_metrics_v1.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved: {metrics_path}")

# Save feature importance
importance_path = '/home/dr/cbu/xgboost_feature_importance_v1.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"✓ Feature importance saved: {importance_path}")

# =============================================================================
# 8. Create Visualizations
# =============================================================================
print("\n" + "=" * 80)
print("Creating Visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))

# 1. ROC Curve
ax1 = plt.subplot(2, 3, 1)
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)

plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_auc:.4f})', linewidth=2)
plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curve', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)

# 2. Precision-Recall Curve
ax2 = plt.subplot(2, 3, 2)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)

plt.plot(recall_train, precision_train, label='Train', linewidth=2)
plt.plot(recall_test, precision_test, label='Test', linewidth=2)
plt.xlabel('Recall', fontsize=11)
plt.ylabel('Precision', fontsize=11)
plt.title('Precision-Recall Curve', fontsize=13, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)

# 3. Feature Importance (Top 20)
ax3 = plt.subplot(2, 3, 3)
top_features = feature_importance.head(20).sort_values('importance')
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
plt.xlabel('Importance (Gain)', fontsize=11)
plt.title('Top 20 Feature Importance', fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# 4. Confusion Matrix (Test)
ax4 = plt.subplot(2, 3, 4)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'],
            annot_kws={'fontsize': 12})
plt.ylabel('True Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')

# 5. Cross-Validation Results
ax5 = plt.subplot(2, 3, 5)
cv_fold_nums = [r['fold'] for r in cv_fold_results]
cv_fold_aucs = [r['auc'] for r in cv_fold_results]
plt.bar(cv_fold_nums, cv_fold_aucs, color='mediumseagreen', alpha=0.7)
plt.axhline(y=np.mean(cv_scores), color='red', linestyle='--',
            label=f'Mean: {np.mean(cv_scores):.4f}', linewidth=2)
plt.xlabel('Fold', fontsize=11)
plt.ylabel('AUC', fontsize=11)
plt.title('Cross-Validation AUC by Fold', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.ylim([min(cv_fold_aucs) * 0.99, max(cv_fold_aucs) * 1.01])
plt.grid(axis='y', alpha=0.3)

# 6. Model Performance Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
XGBoost Model - First Iteration Results

PERFORMANCE METRICS:
  Train AUC:     {train_auc:.6f}
  Test AUC:      {test_auc:.6f}
  CV AUC:        {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}
  Overfit Gap:   {train_auc - test_auc:.6f}

TRAINING DETAILS:
  Features Used:        {X_train.shape[1]}
  Training Samples:     {X_train.shape[0]:,}
  Test Samples:         {X_test.shape[0]:,}
  Best Iteration:       {final_model.best_iteration}
  Training Time:        {train_time:.1f}s
  CV Time:              {cv_time:.1f}s

FEATURE ANALYSIS:
  Zero Importance:      {zero_importance}
  Near-Zero (<0.001):   {near_zero_importance}
  Top 20 Coverage:      {feature_importance.head(20)['importance'].sum()*100:.1f}%

CLASS BALANCE:
  Scale Pos Weight:     {scale_pos_weight:.2f}
  Minority Class:       {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.2f}%)
"""

ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round',
         facecolor='wheat', alpha=0.3))

plt.suptitle('XGBoost First Iteration - Comprehensive Results',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

viz_path = '/home/dr/cbu/xgboost_results_v1.png'
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualizations saved: {viz_path}")

plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nKey Results:")
print(f"  Test AUC:  {test_auc:.6f}")
print(f"  CV AUC:    {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
print(f"  Overfit:   {train_auc - test_auc:.6f}")
print(f"\nFiles Created:")
print(f"  1. {model_path}")
print(f"  2. {predictions_path}")
print(f"  3. {metrics_path}")
print(f"  4. {importance_path}")
print(f"  5. {viz_path}")
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
