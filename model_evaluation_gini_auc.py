"""
Оценка точности модели первой итерации
Метрики: GINI и AUC

GINI коэффициент - важная метрика в кредитном скоринге:
- GINI = 2 * AUC - 1
- Диапазон: от 0 (случайная модель) до 1 (идеальная модель)
- GINI > 0.4 считается хорошим результатом в кредитном скоринге
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("ОЦЕНКА МОДЕЛИ ПЕРВОЙ ИТЕРАЦИИ - МЕТРИКИ GINI И AUC")
print("=" * 100)

# ============================================================================
# 1. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ И ДАННЫХ
# ============================================================================

print("\n" + "=" * 100)
print("ШАГ 1: ЗАГРУЗКА МОДЕЛИ И ТЕСТОВЫХ ДАННЫХ")
print("=" * 100)

# Загружаем обученную XGBoost модель
model = xgb.Booster()
model.load_model('/home/dr/cbu/xgboost_model_v1.json')
print("✅ Модель загружена: xgboost_model_v1.json")

# Загружаем тестовые данные
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')['default'].values

print(f"✅ X_test загружен: {X_test.shape}")
print(f"✅ y_test загружен: {len(y_test)} записей")
print(f"   - Класс 0 (нет дефолта): {(y_test == 0).sum():,} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")
print(f"   - Класс 1 (дефолт):      {(y_test == 1).sum():,} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")

# ============================================================================
# 2. ПРЕДСКАЗАНИЯ МОДЕЛИ
# ============================================================================

print("\n" + "=" * 100)
print("ШАГ 2: ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ")
print("=" * 100)

# Конвертируем в DMatrix для XGBoost
dtest = xgb.DMatrix(X_test)

# Получаем вероятности класса 1 (дефолт)
y_pred_proba = model.predict(dtest)
print(f"✅ Предсказания сгенерированы")
print(f"   - Минимальная вероятность: {y_pred_proba.min():.6f}")
print(f"   - Максимальная вероятность: {y_pred_proba.max():.6f}")
print(f"   - Средняя вероятность:     {y_pred_proba.mean():.6f}")
print(f"   - Медианная вероятность:   {np.median(y_pred_proba):.6f}")

# Бинарные предсказания (порог 0.5)
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

# ============================================================================
# 3. РАСЧЕТ ОСНОВНЫХ МЕТРИК: AUC И GINI
# ============================================================================

print("\n" + "=" * 100)
print("ШАГ 3: РАСЧЕТ МЕТРИК AUC И GINI")
print("=" * 100)

# Расчет AUC (Area Under ROC Curve)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Расчет GINI коэффициента
gini_score = 2 * auc_score - 1

print("\n" + "🎯 " + "=" * 96)
print("ОСНОВНЫЕ МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
print("=" * 100)
print(f"\n📊 AUC (Area Under ROC Curve):  {auc_score:.6f}")
print(f"📊 GINI коэффициент:             {gini_score:.6f}")
print(f"\n💡 Интерпретация GINI:")
print(f"   - GINI = 0:     Модель не лучше случайного угадывания")
print(f"   - GINI = 0.3:   Приемлемый результат")
print(f"   - GINI = 0.4:   Хороший результат для кредитного скоринга")
print(f"   - GINI = 0.5+:  Очень хороший результат")
print(f"   - GINI = 1.0:   Идеальная модель (теоретически)")

# Интерпретация результата
if gini_score < 0.3:
    quality = "❌ НИЗКОЕ КАЧЕСТВО - требуется улучшение"
elif gini_score < 0.4:
    quality = "⚠️  ПРИЕМЛЕМОЕ КАЧЕСТВО - можно улучшить"
elif gini_score < 0.5:
    quality = "✅ ХОРОШЕЕ КАЧЕСТВО"
elif gini_score < 0.6:
    quality = "✅✅ ОЧЕНЬ ХОРОШЕЕ КАЧЕСТВО"
else:
    quality = "🏆 ОТЛИЧНОЕ КАЧЕСТВО"

print(f"\n🎯 Оценка модели: {quality}")
print("=" * 100)

# ============================================================================
# 4. ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ
# ============================================================================

print("\n" + "=" * 100)
print("ШАГ 4: ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ КАЧЕСТВА")
print("=" * 100)

# Average Precision Score (для несбалансированных классов)
avg_precision = average_precision_score(y_test, y_pred_proba)
print(f"\n📊 Average Precision Score: {avg_precision:.6f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
print("\n📊 Confusion Matrix (порог = 0.5):")
print(f"   True Negatives (TN):  {cm[0, 0]:,}")
print(f"   False Positives (FP): {cm[0, 1]:,}")
print(f"   False Negatives (FN): {cm[1, 0]:,}")
print(f"   True Positives (TP):  {cm[1, 1]:,}")

# Метрики из Confusion Matrix
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # Recall, True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"\n📊 Метрики классификации (порог = 0.5):")
print(f"   Sensitivity (Recall):  {sensitivity:.4f}  - процент правильно обнаруженных дефолтов")
print(f"   Specificity:           {specificity:.4f}  - процент правильно обнаруженных недефолтов")
print(f"   Precision:             {precision:.4f}  - точность предсказания дефолта")
print(f"   F1-Score:              {f1_score:.4f}  - гармоническое среднее Precision и Recall")

# Classification Report
print("\n📊 Подробный отчет классификации:")
print(classification_report(y_test, y_pred_binary, target_names=['Нет дефолта', 'Дефолт']))

# ============================================================================
# 5. ПОИСК ОПТИМАЛЬНОГО ПОРОГА
# ============================================================================

print("\n" + "=" * 100)
print("ШАГ 5: АНАЛИЗ ПОРОГОВ КЛАССИФИКАЦИИ")
print("=" * 100)

# Вычисляем метрики для разных порогов
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("\n📊 Метрики для разных порогов:")
print(f"{'Порог':>6} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Предсказано дефолтов':>20}")
print("-" * 70)

for threshold in thresholds_to_test:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()

    precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    recall_t = tp_t / (tp_t + fn_t)
    f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0

    predicted_defaults = (y_pred_proba >= threshold).sum()

    print(f"{threshold:>6.1f} | {precision_t:>10.4f} | {recall_t:>10.4f} | {f1_t:>10.4f} | {predicted_defaults:>20,}")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 100)
print("ШАГ 6: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
print("=" * 100)

# Создаем фигуру с 6 подграфиками
fig = plt.figure(figsize=(20, 12))

# 1. ROC кривая
ax1 = plt.subplot(2, 3, 1)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.4f}\nGINI = {gini_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Model (AUC=0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)

# 2. Precision-Recall кривая
ax2 = plt.subplot(2, 3, 2)
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_curve, precision_curve, linewidth=2, label=f'Avg Precision = {avg_precision:.4f}')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)

# 3. Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Предсказано: НЕТ дефолта', 'Предсказано: ДЕФОЛТ'],
            yticklabels=['Факт: НЕТ дефолта', 'Факт: ДЕФОЛТ'],
            annot_kws={'size': 14})
plt.title('Confusion Matrix (порог = 0.5)', fontsize=14, fontweight='bold')
plt.ylabel('Фактический класс', fontsize=12)
plt.xlabel('Предсказанный класс', fontsize=12)

# 4. Распределение предсказанных вероятностей
ax4 = plt.subplot(2, 3, 4)
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='Класс 0 (нет дефолта)', color='green', edgecolor='black')
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Класс 1 (дефолт)', color='red', edgecolor='black')
plt.xlabel('Предсказанная вероятность дефолта', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.title('Распределение предсказанных вероятностей', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# 5. F1-Score vs Threshold
ax5 = plt.subplot(2, 3, 5)
f1_scores = []
precision_scores = []
recall_scores = []
threshold_range = np.linspace(0.01, 0.99, 100)

for thresh in threshold_range:
    y_pred_t = (y_pred_proba >= thresh).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0

    f1_scores.append(f1_t)
    precision_scores.append(prec_t)
    recall_scores.append(rec_t)

plt.plot(threshold_range, f1_scores, linewidth=2, label='F1-Score', color='blue')
plt.plot(threshold_range, precision_scores, linewidth=2, label='Precision', color='green', alpha=0.7)
plt.plot(threshold_range, recall_scores, linewidth=2, label='Recall', color='orange', alpha=0.7)
plt.xlabel('Порог классификации', fontsize=12)
plt.ylabel('Значение метрики', fontsize=12)
plt.title('Метрики vs Порог', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)

# Находим оптимальный порог по F1
optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = threshold_range[optimal_threshold_idx]
optimal_f1 = f1_scores[optimal_threshold_idx]
plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
            label=f'Оптимум F1={optimal_f1:.3f} при {optimal_threshold:.3f}')
plt.legend(loc='best', fontsize=10)

# 6. Метрики модели (текстовая информация)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

metrics_text = f"""
ОСНОВНЫЕ МЕТРИКИ МОДЕЛИ
{'=' * 40}

AUC Score:           {auc_score:.6f}
GINI Coefficient:    {gini_score:.6f}
Avg Precision:       {avg_precision:.6f}

КАЧЕСТВО: {quality}

{'=' * 40}
МЕТРИКИ ПРИ ПОРОГЕ 0.5:

Sensitivity (Recall): {sensitivity:.4f}
Specificity:          {specificity:.4f}
Precision:            {precision:.4f}
F1-Score:             {f1_score:.4f}

{'=' * 40}
ОПТИМАЛЬНЫЙ ПОРОГ (по F1):

Порог:     {optimal_threshold:.4f}
F1-Score:  {optimal_f1:.4f}

{'=' * 40}
CONFUSION MATRIX (порог=0.5):

True Negatives:   {cm[0,0]:>6,}
False Positives:  {cm[0,1]:>6,}
False Negatives:  {cm[1,0]:>6,}
True Positives:   {cm[1,1]:>6,}
"""

ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/dr/cbu/model_evaluation_gini_auc.png', dpi=150, bbox_inches='tight')
print("✅ Визуализация сохранена: model_evaluation_gini_auc.png")

# ============================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 100)
print("ШАГ 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 100)

# Сохраняем детальные метрики в CSV
results_df = pd.DataFrame({
    'customer_ref': X_test.index if hasattr(X_test, 'index') else range(len(X_test)),
    'actual': y_test,
    'predicted_proba': y_pred_proba,
    'predicted_class_05': y_pred_binary,
    'predicted_class_optimal': (y_pred_proba >= optimal_threshold).astype(int)
})
results_df.to_csv('/home/dr/cbu/model_predictions_with_gini.csv', index=False)
print("✅ Предсказания сохранены: model_predictions_with_gini.csv")

# Сохраняем сводку метрик
metrics_summary = pd.DataFrame({
    'Metric': ['AUC', 'GINI', 'Average_Precision', 'Sensitivity', 'Specificity',
               'Precision', 'F1_Score', 'Optimal_Threshold', 'Optimal_F1'],
    'Value': [auc_score, gini_score, avg_precision, sensitivity, specificity,
              precision, f1_score, optimal_threshold, optimal_f1]
})
metrics_summary.to_csv('/home/dr/cbu/model_metrics_summary.csv', index=False)
print("✅ Сводка метрик сохранена: model_metrics_summary.csv")

# ============================================================================
# ФИНАЛЬНАЯ СВОДКА
# ============================================================================

print("\n" + "=" * 100)
print("ФИНАЛЬНАЯ СВОДКА ОЦЕНКИ МОДЕЛИ")
print("=" * 100)
print(f"\n🎯 AUC:  {auc_score:.6f}")
print(f"🎯 GINI: {gini_score:.6f}")
print(f"\n💡 Модель показывает {quality}")
print(f"\n📁 Файлы результатов:")
print(f"   1. model_evaluation_gini_auc.png - визуализация всех метрик")
print(f"   2. model_predictions_with_gini.csv - детальные предсказания")
print(f"   3. model_metrics_summary.csv - сводка метрик")
print("\n" + "=" * 100)
print("✅ ОЦЕНКА МОДЕЛИ ЗАВЕРШЕНА")
print("=" * 100)
