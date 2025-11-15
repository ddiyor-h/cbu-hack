"""
Проверка точности модели с использованием sklearn метрик
Включает: AUC, ROC-кривую, Cross-Validation, Out-of-Time Validation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ПРОВЕРКА ТОЧНОСТИ МОДЕЛИ - КОМПЛЕКСНАЯ ВАЛИДАЦИЯ")
print("="*80)

# ============================================================================
# 1. ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
# ============================================================================

print("\n[Шаг 1/5] Загрузка модели и данных...")

# Загружаем обученную модель
model_xgb = xgb.Booster()
model_xgb.load_model('/home/dr/cbu/xgboost_model_v1.json')
print("✅ Модель XGBoost загружена")

# Загружаем тестовые данные
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')['default'].values

print(f"✅ X_test: {X_test.shape}")
print(f"✅ y_test: {y_test.shape}")

# Загружаем тренировочные данные для cross-validation
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')['default'].values

print(f"✅ X_train: {X_train.shape}")
print(f"✅ y_train: {y_train.shape}")

# ============================================================================
# 2. БАЗОВЫЙ РАСЧЕТ AUC НА ТЕСТОВОЙ ВЫБОРКЕ
# ============================================================================

print("\n" + "="*80)
print("[Шаг 2/5] БАЗОВЫЙ РАСЧЕТ AUC")
print("="*80)

# Получаем вероятности (НЕ классы!)
dtest = xgb.DMatrix(X_test)
y_pred_proba = model_xgb.predict(dtest)

# Считаем AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
gini_score = 2 * auc_score - 1  # Gini = 2*AUC - 1

print(f"\n📊 МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
print(f"   • AUC  = {auc_score:.4f}")
print(f"   • GINI = {gini_score:.4f}")

if gini_score > 0.6:
    quality = "ОТЛИЧНО"
elif gini_score > 0.4:
    quality = "ОЧЕНЬ ХОРОШО"
elif gini_score > 0.3:
    quality = "ХОРОШО"
else:
    quality = "УДОВЛЕТВОРИТЕЛЬНО"

print(f"   • Качество модели: {quality}")

# ============================================================================
# 3. ВИЗУАЛИЗАЦИЯ ROC-КРИВОЙ
# ============================================================================

print("\n" + "="*80)
print("[Шаг 3/5] ВИЗУАЛИЗАЦИЯ ROC-КРИВОЙ")
print("="*80)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Создаем фигуру с 2 графиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# График 1: ROC-кривая
ax1.plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Random Classifier (AUC = 0.5)')
ax1.fill_between(fpr, tpr, alpha=0.2, color='blue', label='AUC Area')

ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
ax1.set_title(f'ROC Curve\nAUC = {auc_score:.4f} | GINI = {gini_score:.4f}',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])

# Добавляем текстовую аннотацию
textstr = f'Quality: {quality}\nTest Size: {len(y_test):,}\nPositive Rate: {y_test.mean():.2%}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.65, 0.15, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# График 2: Распределение вероятностей по классам
ax2.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='Class 0 (No Default)',
         color='green', edgecolor='black', linewidth=0.5)
ax2.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Class 1 (Default)',
         color='red', edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Predicted Probabilities by True Class',
              fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/dr/cbu/model_accuracy_roc_curve.png', dpi=150, bbox_inches='tight')
print("✅ ROC-кривая сохранена: model_accuracy_roc_curve.png")
plt.close()

# ============================================================================
# 4. CROSS-VALIDATION (важно для оценки стабильности!)
# ============================================================================

print("\n" + "="*80)
print("[Шаг 4/5] CROSS-VALIDATION (5-FOLD STRATIFIED)")
print("="*80)

print("\n⏳ Выполняется 5-fold cross-validation...")
print("   (это может занять несколько минут)")

# Создаем XGBoost классификатор для sklearn API
from xgboost import XGBClassifier

# Параметры должны соответствовать обученной модели
model_cv = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=18.6,  # Для балансировки классов
    n_jobs=-1
)

# Stratified K-Fold для сохранения баланса классов
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Выполняем cross-validation
auc_scores = cross_val_score(model_cv, X_train, y_train,
                             cv=cv, scoring='roc_auc', n_jobs=-1)

print("\n📊 РЕЗУЛЬТАТЫ CROSS-VALIDATION:")
print(f"   • Fold 1: AUC = {auc_scores[0]:.4f}")
print(f"   • Fold 2: AUC = {auc_scores[1]:.4f}")
print(f"   • Fold 3: AUC = {auc_scores[2]:.4f}")
print(f"   • Fold 4: AUC = {auc_scores[3]:.4f}")
print(f"   • Fold 5: AUC = {auc_scores[4]:.4f}")
print(f"\n   • Среднее AUC:     {auc_scores.mean():.4f}")
print(f"   • Стд. отклонение: ±{auc_scores.std():.4f}")
print(f"   • Min AUC:         {auc_scores.min():.4f}")
print(f"   • Max AUC:         {auc_scores.max():.4f}")

cv_gini_mean = 2 * auc_scores.mean() - 1
cv_gini_std = 2 * auc_scores.std()

print(f"\n   • Средний GINI:    {cv_gini_mean:.4f} (±{cv_gini_std:.4f})")

# Проверка на overfitting
if abs(auc_score - auc_scores.mean()) > 0.02:
    print("\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Разница между Test AUC и CV AUC > 0.02")
    print(f"   Возможен overfitting. Test AUC = {auc_score:.4f}, CV AUC = {auc_scores.mean():.4f}")
else:
    print(f"\n✅ Модель стабильна: разница между Test и CV = {abs(auc_score - auc_scores.mean()):.4f}")

# ============================================================================
# 5. OUT-OF-TIME VALIDATION (если есть временные переменные)
# ============================================================================

print("\n" + "="*80)
print("[Шаг 5/5] OUT-OF-TIME VALIDATION")
print("="*80)

# Проверяем наличие временных признаков
time_features = [col for col in X_train.columns if 'year' in col.lower() or
                 'month' in col.lower() or 'day' in col.lower() or
                 'hour' in col.lower()]

print(f"\n🔍 Найдено временных признаков: {len(time_features)}")
if len(time_features) > 0:
    print(f"   Признаки: {', '.join(time_features[:5])}")

# Если есть признак account_open_year, используем его для временной валидации
if 'account_open_year' in X_train.columns:
    print("\n⏰ Выполняется Out-of-Time Validation по признаку 'account_open_year'...")

    # Объединяем данные
    df_full = pd.concat([
        pd.concat([X_train, pd.DataFrame({'default': y_train})], axis=1),
        pd.concat([X_test, pd.DataFrame({'default': y_test})], axis=1)
    ], axis=0)

    # Сортируем по году
    df_sorted = df_full.sort_values('account_open_year').reset_index(drop=True)

    # Разделяем на старые и новые данные (80/20 по времени)
    train_size = int(0.8 * len(df_sorted))

    df_train_time = df_sorted.iloc[:train_size]
    df_test_time = df_sorted.iloc[train_size:]

    X_train_time = df_train_time.drop('default', axis=1)
    y_train_time = df_train_time['default'].values
    X_test_time = df_test_time.drop('default', axis=1)
    y_test_time = df_test_time['default'].values

    print(f"   • Train period: {X_train_time['account_open_year'].min():.0f} - {X_train_time['account_open_year'].max():.0f}")
    print(f"   • Test period:  {X_test_time['account_open_year'].min():.0f} - {X_test_time['account_open_year'].max():.0f}")
    print(f"   • Train size: {len(X_train_time):,}")
    print(f"   • Test size:  {len(X_test_time):,}")

    # Обучаем модель на старых данных
    model_oot = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=18.6,
        n_jobs=-1
    )

    print("\n⏳ Обучение модели на исторических данных...")
    model_oot.fit(X_train_time, y_train_time)

    # Предсказываем на новых данных
    y_pred_oot = model_oot.predict_proba(X_test_time)[:, 1]
    auc_oot = roc_auc_score(y_test_time, y_pred_oot)
    gini_oot = 2 * auc_oot - 1

    print("\n📊 РЕЗУЛЬТАТЫ OUT-OF-TIME VALIDATION:")
    print(f"   • Out-of-Time AUC:  {auc_oot:.4f}")
    print(f"   • Out-of-Time GINI: {gini_oot:.4f}")
    print(f"   • Разница с Test AUC: {auc_score - auc_oot:+.4f}")

    if abs(auc_score - auc_oot) > 0.05:
        print("\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Значительная деградация во времени!")
        print("   Модель может нестабильно работать на будущих данных")
    else:
        print("\n✅ Модель стабильна во времени")
else:
    print("\n⚠️  Временные признаки не найдены или недостаточно информации")
    print("   Out-of-Time Validation пропущена")

# ============================================================================
# 6. ИТОГОВЫЙ ОТЧЕТ
# ============================================================================

print("\n" + "="*80)
print("📊 ИТОГОВЫЙ ОТЧЕТ ПО ТОЧНОСТИ МОДЕЛИ")
print("="*80)

results_summary = {
    'Metric': [
        'Test AUC',
        'Test GINI',
        'Cross-Val AUC (mean)',
        'Cross-Val AUC (std)',
        'Cross-Val GINI (mean)',
    ],
    'Value': [
        f'{auc_score:.4f}',
        f'{gini_score:.4f}',
        f'{auc_scores.mean():.4f}',
        f'{auc_scores.std():.4f}',
        f'{cv_gini_mean:.4f}',
    ]
}

if 'account_open_year' in X_train.columns:
    results_summary['Metric'].extend(['Out-of-Time AUC', 'Out-of-Time GINI'])
    results_summary['Value'].extend([f'{auc_oot:.4f}', f'{gini_oot:.4f}'])

results_df = pd.DataFrame(results_summary)
results_df.to_csv('/home/dr/cbu/model_accuracy_summary.csv', index=False)

print("\n", results_df.to_string(index=False))

print(f"\n✅ Качество модели: {quality}")

# Рекомендации
print("\n" + "="*80)
print("💡 РЕКОМЕНДАЦИИ")
print("="*80)

if auc_score > 0.8:
    print("✅ Отличная дискриминационная способность модели")
elif auc_score > 0.75:
    print("✅ Хорошая дискриминационная способность модели")
else:
    print("⚠️  Модель имеет среднюю дискриминационную способность")
    print("   Рекомендуется:")
    print("   • Добавить новые признаки")
    print("   • Провести feature engineering")
    print("   • Попробовать другие алгоритмы (LightGBM, CatBoost)")

if auc_scores.std() > 0.02:
    print("\n⚠️  Высокая вариативность между фолдами")
    print("   Рекомендуется увеличить размер тренировочной выборки")

print("\n" + "="*80)
print("✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА!")
print("="*80)

print("\n📁 Созданные файлы:")
print("   1. model_accuracy_roc_curve.png    - ROC-кривая и распределения")
print("   2. model_accuracy_summary.csv      - Итоговые метрики")
