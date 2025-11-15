"""
ИТЕРАЦИЯ 2: Обучение модели с выбранными признаками
Стратегия: Удаление мультиколлинеарности + отбор наиболее предсказательных признаков
Валидация: AUC, ROC, Cross-Validation, Out-of-Time
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("МОДЕЛЬ V2: ОБУЧЕНИЕ С ОТОБРАННЫМИ ПРИЗНАКАМИ")
print("="*80)

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n[Шаг 1/8] Загрузка данных...")

X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')['default'].values
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')['default'].values

print(f"✅ X_train: {X_train.shape}")
print(f"✅ y_train: {y_train.shape}")
print(f"✅ X_test: {X_test.shape}")
print(f"✅ y_test: {y_test.shape}")

# ============================================================================
# 2. ОТБОР ПРИЗНАКОВ - УДАЛЕНИЕ МУЛЬТИКОЛЛИНЕАРНОСТИ
# ============================================================================

print("\n" + "="*80)
print("[Шаг 2/8] ОТБОР ПРИЗНАКОВ")
print("="*80)

# Загружаем результаты корреляционного анализа
target_corr = pd.read_csv('/home/dr/cbu/target_correlations.csv')
high_corr = pd.read_csv('/home/dr/cbu/high_correlations_multicollinearity.csv')

print(f"\n📊 Всего признаков в исходном датасете: {X_train.shape[1]}")
print(f"⚠️  Найдено пар с высокой коррелацией (|r| > 0.8): {len(high_corr)}")

# Стратегия отбора признаков:
# 1. Из пар с очень высокой корреляцией (r > 0.95) оставляем только 1 признак
# 2. Выбираем признак с более высокой корреляцией с таргетом

features_to_remove = set()

# Обрабатываем пары с экстремально высокой корреляцией (> 0.95)
extreme_pairs = high_corr[high_corr['correlation'].abs() > 0.95]

print(f"\n🔍 Обработка {len(extreme_pairs)} пар с |r| > 0.95...")

for idx, row in extreme_pairs.iterrows():
    feat1 = row['feature_1']
    feat2 = row['feature_2']

    # Получаем корреляции с таргетом
    corr1 = target_corr[target_corr['feature'] == feat1]['abs_correlation'].values
    corr2 = target_corr[target_corr['feature'] == feat2]['abs_correlation'].values

    if len(corr1) > 0 and len(corr2) > 0:
        # Удаляем признак с меньшей корреляцией с таргетом
        if corr1[0] < corr2[0]:
            features_to_remove.add(feat1)
        else:
            features_to_remove.add(feat2)

print(f"📝 Признаков отмечено для удаления: {len(features_to_remove)}")

# Дополнительно: удаляем признаки с очень низкой корреляцией с таргетом (< 0.01)
# но только если они не входят в важные категории
low_corr_features = target_corr[target_corr['abs_correlation'] < 0.01]['feature'].tolist()
print(f"📝 Признаков с очень низкой корреляцией (< 0.01): {len(low_corr_features)}")

# Удаляем низко коррелированные признаки, которые не несут информации
for feat in low_corr_features:
    # Сохраняем некоторые категориальные признаки даже с низкой корреляцией
    # (они могут быть важны в комбинации с другими)
    if not any(x in feat for x in ['education_', 'employment_', 'marital_',
                                     'age_group_', 'loan_purpose_']):
        features_to_remove.add(feat)

# Финальный список признаков
selected_features = [col for col in X_train.columns if col not in features_to_remove]

print(f"\n✅ ИТОГО:")
print(f"   • Исходных признаков:  {X_train.shape[1]}")
print(f"   • Удалено признаков:   {len(features_to_remove)}")
print(f"   • Выбрано признаков:   {len(selected_features)}")

# Сохраняем список выбранных признаков
pd.DataFrame({'feature': selected_features}).to_csv(
    '/home/dr/cbu/selected_features_v2.csv', index=False
)
print(f"\n💾 Список выбранных признаков сохранен: selected_features_v2.csv")

# Показываем топ-20 признаков по корреляции с таргетом, которые были выбраны
top_selected = target_corr[target_corr['feature'].isin(selected_features)].head(20)
print(f"\n📈 ТОП-20 ВЫБРАННЫХ ПРИЗНАКОВ:")
print("="*80)
for i, (idx, row) in enumerate(top_selected.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:50s} : {row['correlation_with_default']:+.4f}")

# Создаем обучающие и тестовые выборки с выбранными признаками
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"\n✅ Подготовленные датасеты:")
print(f"   • X_train_selected: {X_train_selected.shape}")
print(f"   • X_test_selected:  {X_test_selected.shape}")

# ============================================================================
# 3. ОБУЧЕНИЕ МОДЕЛИ V2
# ============================================================================

print("\n" + "="*80)
print("[Шаг 3/8] ОБУЧЕНИЕ XGBOOST МОДЕЛИ V2")
print("="*80)

print("\n⏳ Обучение модели с отобранными признаками...")
print(f"   Используется {len(selected_features)} признаков")

# Создаем DMatrix для XGBoost
dtrain = xgb.DMatrix(X_train_selected, label=y_train)
dtest = xgb.DMatrix(X_test_selected, label=y_test)

# Параметры модели (те же, что и в v1)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 18.6,  # Балансировка классов
    'seed': 42
}

# Обучаем модель с early stopping
evals = [(dtrain, 'train'), (dtest, 'test')]
evals_result = {}

model_v2 = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=50,
    verbose_eval=50
)

print(f"\n✅ Модель обучена!")
print(f"   • Оптимальное количество деревьев: {model_v2.best_iteration}")

# Сохраняем модель
model_v2.save_model('/home/dr/cbu/xgboost_model_v2.json')
print(f"💾 Модель сохранена: xgboost_model_v2.json")

# ============================================================================
# 4. БАЗОВЫЙ РАСЧЕТ AUC НА ТЕСТОВОЙ ВЫБОРКЕ
# ============================================================================

print("\n" + "="*80)
print("[Шаг 4/8] БАЗОВЫЙ РАСЧЕТ AUC")
print("="*80)

# Получаем вероятности (НЕ классы!)
y_pred_proba = model_v2.predict(dtest)

# Считаем AUC
auc_score_v2 = roc_auc_score(y_test, y_pred_proba)
gini_score_v2 = 2 * auc_score_v2 - 1

print(f"\n📊 МЕТРИКИ МОДЕЛИ V2 (НА ТЕСТОВОЙ ВЫБОРКЕ):")
print(f"   • AUC  = {auc_score_v2:.4f}")
print(f"   • GINI = {gini_score_v2:.4f}")

# Загружаем результаты v1 для сравнения
try:
    results_v1 = pd.read_csv('/home/dr/cbu/model_accuracy_summary.csv')
    auc_v1 = float(results_v1[results_v1['Metric'] == 'Test AUC']['Value'].values[0])
    gini_v1 = float(results_v1[results_v1['Metric'] == 'Test GINI']['Value'].values[0])

    print(f"\n📊 СРАВНЕНИЕ С МОДЕЛЬЮ V1:")
    print(f"   • V1 AUC:  {auc_v1:.4f}  →  V2 AUC:  {auc_score_v2:.4f}  (Δ = {auc_score_v2 - auc_v1:+.4f})")
    print(f"   • V1 GINI: {gini_v1:.4f}  →  V2 GINI: {gini_score_v2:.4f}  (Δ = {gini_score_v2 - gini_v1:+.4f})")

    if auc_score_v2 > auc_v1:
        print(f"\n✅ УЛУЧШЕНИЕ! V2 лучше V1 на {(auc_score_v2 - auc_v1) * 100:.2f}%")
    elif auc_score_v2 < auc_v1:
        print(f"\n⚠️  V2 хуже V1 на {(auc_v1 - auc_score_v2) * 100:.2f}%")
    else:
        print(f"\n📊 V2 и V1 показывают одинаковые результаты")
except:
    print("\n⚠️  Не удалось загрузить результаты V1 для сравнения")

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ ROC-КРИВОЙ
# ============================================================================

print("\n" + "="*80)
print("[Шаг 5/8] ВИЗУАЛИЗАЦИЯ ROC-КРИВОЙ")
print("="*80)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# График 1: ROC-кривая
ax1.plot(fpr, tpr, 'b-', lw=2, label=f'V2 Model (AUC = {auc_score_v2:.4f})')
ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Random (AUC = 0.5)')
ax1.fill_between(fpr, tpr, alpha=0.2, color='blue')

ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
ax1.set_title(f'ROC Curve - Model V2\nAUC = {auc_score_v2:.4f} | GINI = {gini_score_v2:.4f}',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])

# Статистика
textstr = f'Features: {len(selected_features)}\nTest Size: {len(y_test):,}\nPositive Rate: {y_test.mean():.2%}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.65, 0.15, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# График 2: Распределение вероятностей
ax2.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='No Default',
         color='green', edgecolor='black', linewidth=0.5)
ax2.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Default',
         color='red', edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Predicted Probability Distribution by Class',
              fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/dr/cbu/model_v2_roc_curve.png', dpi=150, bbox_inches='tight')
print("✅ ROC-кривая сохранена: model_v2_roc_curve.png")
plt.close()

# ============================================================================
# 6. CROSS-VALIDATION (5-FOLD STRATIFIED)
# ============================================================================

print("\n" + "="*80)
print("[Шаг 6/8] CROSS-VALIDATION (5-FOLD STRATIFIED)")
print("="*80)

print("\n⏳ Выполняется 5-fold cross-validation...")

# Создаем XGBClassifier для sklearn API
model_cv = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=model_v2.best_iteration,  # Используем оптимальное количество
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=18.6,
    n_jobs=-1
)

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Выполняем cross-validation
auc_scores_cv = cross_val_score(model_cv, X_train_selected, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=-1)

print("\n📊 РЕЗУЛЬТАТЫ CROSS-VALIDATION:")
for i, score in enumerate(auc_scores_cv, 1):
    print(f"   • Fold {i}: AUC = {score:.4f}")

print(f"\n   • Среднее AUC:     {auc_scores_cv.mean():.4f}")
print(f"   • Стд. отклонение: ±{auc_scores_cv.std():.4f}")
print(f"   • Min AUC:         {auc_scores_cv.min():.4f}")
print(f"   • Max AUC:         {auc_scores_cv.max():.4f}")

cv_gini_mean = 2 * auc_scores_cv.mean() - 1
cv_gini_std = 2 * auc_scores_cv.std()
print(f"\n   • Средний GINI:    {cv_gini_mean:.4f} (±{cv_gini_std:.4f})")

# Проверка на overfitting
if abs(auc_score_v2 - auc_scores_cv.mean()) > 0.02:
    print("\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Разница между Test AUC и CV AUC > 0.02")
    print(f"   Возможен overfitting. Test AUC = {auc_score_v2:.4f}, CV AUC = {auc_scores_cv.mean():.4f}")
else:
    print(f"\n✅ Модель стабильна: разница = {abs(auc_score_v2 - auc_scores_cv.mean()):.4f}")

# ============================================================================
# 7. OUT-OF-TIME VALIDATION
# ============================================================================

print("\n" + "="*80)
print("[Шаг 7/8] OUT-OF-TIME VALIDATION")
print("="*80)

if 'account_open_year' in selected_features:
    print("\n⏰ Выполняется Out-of-Time Validation...")

    # Объединяем данные
    df_full = pd.concat([
        pd.concat([X_train_selected, pd.DataFrame({'default': y_train})], axis=1),
        pd.concat([X_test_selected, pd.DataFrame({'default': y_test})], axis=1)
    ], axis=0)

    # Сортируем по году
    df_sorted = df_full.sort_values('account_open_year').reset_index(drop=True)

    # Разделяем 80/20 по времени
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

    # Обучаем модель
    model_oot = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=model_v2.best_iteration,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=18.6,
        n_jobs=-1
    )

    print("\n⏳ Обучение на исторических данных...")
    model_oot.fit(X_train_time, y_train_time)

    # Предсказываем
    y_pred_oot = model_oot.predict_proba(X_test_time)[:, 1]
    auc_oot = roc_auc_score(y_test_time, y_pred_oot)
    gini_oot = 2 * auc_oot - 1

    print("\n📊 РЕЗУЛЬТАТЫ OUT-OF-TIME VALIDATION:")
    print(f"   • OOT AUC:  {auc_oot:.4f}")
    print(f"   • OOT GINI: {gini_oot:.4f}")
    print(f"   • Разница с Test AUC: {auc_score_v2 - auc_oot:+.4f}")

    if abs(auc_score_v2 - auc_oot) > 0.05:
        print("\n⚠️  Значительная деградация во времени!")
    else:
        print("\n✅ Модель стабильна во времени")
else:
    print("\n⚠️  account_open_year не входит в выбранные признаки")
    print("   Out-of-Time Validation пропущена")
    auc_oot = None
    gini_oot = None

# ============================================================================
# 8. ИТОГОВЫЙ ОТЧЕТ
# ============================================================================

print("\n" + "="*80)
print("📊 ИТОГОВЫЙ ОТЧЕТ - МОДЕЛЬ V2")
print("="*80)

# Создаем таблицу результатов
results_summary = {
    'Metric': [
        'Features Count',
        'Test AUC',
        'Test GINI',
        'Cross-Val AUC (mean)',
        'Cross-Val AUC (std)',
        'Cross-Val GINI (mean)',
    ],
    'Value': [
        str(len(selected_features)),
        f'{auc_score_v2:.4f}',
        f'{gini_score_v2:.4f}',
        f'{auc_scores_cv.mean():.4f}',
        f'{auc_scores_cv.std():.4f}',
        f'{cv_gini_mean:.4f}',
    ]
}

if auc_oot is not None:
    results_summary['Metric'].extend(['Out-of-Time AUC', 'Out-of-Time GINI'])
    results_summary['Value'].extend([f'{auc_oot:.4f}', f'{gini_oot:.4f}'])

results_df = pd.DataFrame(results_summary)
results_df.to_csv('/home/dr/cbu/model_v2_summary.csv', index=False)

print("\n", results_df.to_string(index=False))

# Сравнение с V1
try:
    print("\n" + "="*80)
    print("📊 СРАВНЕНИЕ V1 vs V2")
    print("="*80)

    results_v1 = pd.read_csv('/home/dr/cbu/model_accuracy_summary.csv')

    metrics_comparison = []
    for metric in ['Test AUC', 'Test GINI', 'Cross-Val AUC (mean)']:
        v1_val = results_v1[results_v1['Metric'] == metric]['Value'].values[0]
        v2_val = results_df[results_df['Metric'] == metric]['Value'].values[0]

        try:
            v1_num = float(v1_val)
            v2_num = float(v2_val)
            delta = v2_num - v1_num
            delta_pct = (delta / v1_num) * 100

            metrics_comparison.append({
                'Metric': metric,
                'V1': v1_val,
                'V2': v2_val,
                'Delta': f'{delta:+.4f}',
                'Delta%': f'{delta_pct:+.2f}%'
            })
        except:
            pass

    if metrics_comparison:
        comparison_df = pd.DataFrame(metrics_comparison)
        print("\n", comparison_df.to_string(index=False))
        comparison_df.to_csv('/home/dr/cbu/v1_vs_v2_comparison.csv', index=False)
        print("\n💾 Сравнение сохранено: v1_vs_v2_comparison.csv")
except:
    print("\n⚠️  Не удалось создать сравнение с V1")

print("\n" + "="*80)
print("✅ ОБУЧЕНИЕ МОДЕЛИ V2 ЗАВЕРШЕНО!")
print("="*80)

print("\n📁 Созданные файлы:")
print("   1. xgboost_model_v2.json         - Обученная модель")
print("   2. selected_features_v2.csv      - Список выбранных признаков")
print("   3. model_v2_roc_curve.png        - ROC-кривая")
print("   4. model_v2_summary.csv          - Метрики модели V2")
print("   5. v1_vs_v2_comparison.csv       - Сравнение V1 vs V2")

print("\n" + "="*80)
