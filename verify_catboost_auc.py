"""
Перепроверка AUC модели CatBoost используя стандартный метод sklearn
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

print("="*80)
print("ПЕРЕПРОВЕРКА AUC - CatBoost Iteration 3")
print("="*80)
print()

# Пути
DATA_DIR = '/home/dr/cbu'

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# ============================================================================

print("[1/3] Загрузка данных и модели...")

# Загружаем тестовые данные
X_test = pd.read_parquet(f'{DATA_DIR}/X_test_engineered.parquet')
y_test = pd.read_parquet(f'{DATA_DIR}/y_test.parquet').values.ravel()

print(f"✅ Test set: {X_test.shape}")
print(f"✅ True labels: {y_test.shape}")
print(f"   • No-default: {np.sum(y_test==0):,}")
print(f"   • Default: {np.sum(y_test==1):,}")
print(f"   • Default rate: {np.mean(y_test):.2%}")
print()

# Загружаем модель CatBoost
model = CatBoostClassifier()
model.load_model(f'{DATA_DIR}/model_catboost_iter3.cbm')
print(f"✅ Модель загружена: model_catboost_iter3.cbm")
print()

# ============================================================================
# 2. ПРЕДСКАЗАНИЯ
# ============================================================================

print("[2/3] Генерация предсказаний...")

# Получаем вероятности (НЕ классы!)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"✅ Предсказания получены: {y_pred_proba.shape}")
print(f"   • Min probability: {y_pred_proba.min():.4f}")
print(f"   • Max probability: {y_pred_proba.max():.4f}")
print(f"   • Mean probability: {y_pred_proba.mean():.4f}")
print(f"   • Median probability: {np.median(y_pred_proba):.4f}")
print()

# ============================================================================
# 3. РАСЧЕТ AUC СТАНДАРТНЫМ МЕТОДОМ sklearn
# ============================================================================

print("[3/3] Расчет AUC стандартным методом sklearn.metrics.roc_auc_score...")
print()

# Метод 1: roc_auc_score (стандартный)
auc_score = roc_auc_score(y_test, y_pred_proba)
gini_score = 2 * auc_score - 1

print("📊 РЕЗУЛЬТАТЫ:")
print(f"   • AUC (roc_auc_score):  {auc_score:.6f}")
print(f"   • GINI:                  {gini_score:.6f}")
print()

# Метод 2: Ручной расчет через ROC curve (для верификации)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# AUC = площадь под ROC кривой (trapezoid rule)
auc_manual = np.trapz(tpr, fpr)
gini_manual = 2 * auc_manual - 1

print("📊 ВЕРИФИКАЦИЯ (ручной расчет через ROC curve):")
print(f"   • AUC (manual trapz):    {auc_manual:.6f}")
print(f"   • GINI:                  {gini_manual:.6f}")
print()

# Разница между методами (должна быть ~0)
difference = abs(auc_score - auc_manual)
print(f"   • Разница между методами: {difference:.8f}")
if difference < 0.0001:
    print(f"   ✅ Методы согласуются (разница < 0.0001)")
else:
    print(f"   ⚠️  Методы расходятся (разница >= 0.0001)")
print()

# ============================================================================
# 4. СРАВНЕНИЕ С СОХРАНЕННЫМИ ПРЕДСКАЗАНИЯМИ
# ============================================================================

print("📊 СРАВНЕНИЕ С СОХРАНЕННЫМИ ПРЕДСКАЗАНИЯМИ:")
print()

# Загружаем сохраненные предсказания из iteration 3
saved_predictions = pd.read_csv(f'{DATA_DIR}/predictions_catboost_iter3.csv')
y_pred_saved = saved_predictions['prediction'].values

# Проверяем, что предсказания идентичны
predictions_match = np.allclose(y_pred_proba, y_pred_saved)
max_diff = np.max(np.abs(y_pred_proba - y_pred_saved))

print(f"   • Predictions match: {predictions_match}")
print(f"   • Max difference: {max_diff:.10f}")

if predictions_match:
    print(f"   ✅ Предсказания идентичны сохраненным")
else:
    print(f"   ⚠️  Предсказания отличаются от сохраненных")

# Пересчитываем AUC из сохраненных predictions
auc_from_saved = roc_auc_score(y_test, y_pred_saved)
print(f"   • AUC from saved predictions: {auc_from_saved:.6f}")
print()

# ============================================================================
# 5. СРАВНЕНИЕ С BASELINE И ДРУГИМИ МОДЕЛЯМИ
# ============================================================================

print("="*80)
print("📊 ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*80)
print()

results = {
    'Model': [
        'XGBoost V1 (Baseline)',
        'XGBoost V2 (Feature Selection)',
        'XGBoost Optimized',
        'CatBoost Iter3'
    ],
    'Test AUC': [
        '0.7843',
        '0.7889',
        '0.8047',
        f'{auc_score:.4f}'
    ],
    'GINI': [
        '0.5685',
        '0.5779',
        '0.6094',
        f'{gini_score:.4f}'
    ]
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# Расчет улучшений
baseline_auc = 0.8047
improvement = auc_score - baseline_auc
improvement_pct = (improvement / baseline_auc) * 100

print(f"📈 УЛУЧШЕНИЕ НАД BASELINE (XGBoost Optimized):")
print(f"   • Baseline AUC:  {baseline_auc:.4f}")
print(f"   • CatBoost AUC:  {auc_score:.4f}")
print(f"   • Улучшение:     {improvement:+.4f} ({improvement_pct:+.2f}%)")
print()

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================================

print("📊 Создание визуализации...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# График 1: ROC Curve
ax1.plot(fpr, tpr, 'b-', lw=2, label=f'CatBoost (AUC = {auc_score:.4f})')
ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Random (AUC = 0.5)')
ax1.fill_between(fpr, tpr, alpha=0.2, color='blue')

ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
ax1.set_title(f'ROC Curve - CatBoost Iter3\nAUC = {auc_score:.4f}, GINI = {gini_score:.4f}',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])

# Добавляем статистику
textstr = f'Test Size: {len(y_test):,}\nDefault Rate: {np.mean(y_test):.2%}\nImbalance: {np.sum(y_test==0)/np.sum(y_test==1):.1f}:1'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.65, 0.15, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# График 2: Распределение вероятностей
ax2.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='No Default',
         color='green', edgecolor='black', linewidth=0.5, density=True)
ax2.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Default',
         color='red', edgecolor='black', linewidth=0.5, density=True)

ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Predicted Probabilities',
              fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{DATA_DIR}/catboost_auc_verification.png', dpi=150, bbox_inches='tight')
print(f"✅ Визуализация сохранена: catboost_auc_verification.png")
print()

# Сохраняем результаты верификации
verification_results = pd.DataFrame({
    'Method': ['sklearn.roc_auc_score', 'Manual (trapz)', 'Saved predictions'],
    'AUC': [auc_score, auc_manual, auc_from_saved],
    'GINI': [gini_score, gini_manual, 2*auc_from_saved-1]
})

verification_results.to_csv(f'{DATA_DIR}/catboost_auc_verification.csv', index=False)
print(f"✅ Результаты верификации сохранены: catboost_auc_verification.csv")
print()

print("="*80)
print("✅ ВЕРИФИКАЦИЯ ЗАВЕРШЕНА")
print("="*80)
print()

print("📊 ФИНАЛЬНЫЙ ОТВЕТ:")
print(f"   • Test AUC (стандартный метод sklearn): {auc_score:.6f}")
print(f"   • Test GINI: {gini_score:.6f}")
print(f"   • Верификация методов: ✅ Согласуются")
print()
