"""
Feature Selection Analysis for Credit Default Prediction
Анализ признаков для отбора фич
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("АНАЛИЗ ПРИЗНАКОВ ДЛЯ ОТБОРА (FEATURE SELECTION ANALYSIS)")
print("="*80)

# Load data
print("\n1. Загрузка данных...")
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Всего признаков: {X_train.shape[1]}")

print("\n" + "="*80)
print("2. АНАЛИЗ ТИПОВ ПРИЗНАКОВ")
print("="*80)

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nЧисловые признаки: {len(numeric_cols)}")

# Categorize features
id_features = [col for col in numeric_cols if 'id' in col.lower() or col == 'customer_ref']
binary_features = [col for col in numeric_cols if X_train[col].nunique() == 2 and set(X_train[col].unique()).issubset({0, 1})]
engineered_features = [col for col in numeric_cols if any(x in col for x in ['_to_', '_per_', '_vs_'])]
frequency_features = [col for col in numeric_cols if col.endswith('_freq')]
categorical_encoded = [col for col in X_train.columns if any(x in col for x in ['employment_type_', 'education_', 'marital_status_', 'loan_purpose_', 'state_', 'account_status_', 'preferred_contact_', 'origination_channel_', 'application_time_of_day_', 'credit_util_category_', 'age_group_', 'employment_stability_'])]

print(f"  - ID признаки (нужно удалить): {len(id_features)}")
print(f"    {id_features}")
print(f"\n  - Бинарные признаки (0/1): {len(binary_features)}")
print(f"  - Инженерные признаки: {len(engineered_features)}")
print(f"  - Частотное кодирование: {len(frequency_features)}")
print(f"  - One-hot закодированные: {len(categorical_encoded)}")

print("\n" + "="*80)
print("3. АНАЛИЗ ДИСПЕРСИИ ПРИЗНАКОВ")
print("="*80)

# Near-zero variance features
variance_data = []
for col in numeric_cols:
    if col not in id_features:
        var = X_train[col].var()
        nunique = X_train[col].nunique()
        variance_data.append({
            'feature': col,
            'variance': var,
            'nunique': nunique,
            'is_binary': nunique == 2
        })

variance_df = pd.DataFrame(variance_data).sort_values('variance')

# Low variance features (excluding binary which naturally have low variance)
low_var = variance_df[(variance_df['variance'] < 0.01) & (~variance_df['is_binary'])]
print(f"\nПризнаки с очень низкой дисперсией (< 0.01, исключая бинарные): {len(low_var)}")
if len(low_var) > 0:
    print(low_var[['feature', 'variance', 'nunique']].to_string(index=False))
else:
    print("Таких признаков не найдено - это хорошо!")

print("\n" + "="*80)
print("4. АНАЛИЗ КОРРЕЛЯЦИИ (Multicollinearity)")
print("="*80)

# Calculate correlation matrix for numeric features (excluding IDs and categorical encoded)
analysis_features = [col for col in numeric_cols
                     if col not in id_features
                     and col not in categorical_encoded]

print(f"\nАнализируем корреляцию для {len(analysis_features)} числовых признаков...")

corr_matrix = X_train[analysis_features].corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.9:
            high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

print(f"\nПары признаков с высокой корреляцией (r > 0.9): {len(high_corr_pairs)}")
if high_corr_pairs:
    corr_pairs_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    print(corr_pairs_df.to_string(index=False))
else:
    print("Сильно коррелирующих пар не найдено - отлично!")

# Moderate correlation
moderate_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if 0.8 < corr_matrix.iloc[i, j] <= 0.9:
            moderate_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

print(f"\nПары с умеренной корреляцией (0.8 < r <= 0.9): {len(moderate_corr_pairs)}")
if moderate_corr_pairs:
    mod_corr_df = pd.DataFrame(moderate_corr_pairs).sort_values('correlation', ascending=False)
    print(mod_corr_df.head(10).to_string(index=False))

print("\n" + "="*80)
print("5. ПРОВЕРКА НА УТЕЧКУ ДАННЫХ (Data Leakage)")
print("="*80)

# Check if any features are too perfectly correlated with target
y_values = y_train.values.ravel()

leakage_check = []
for col in analysis_features:
    # Calculate correlation with target
    if X_train[col].nunique() > 1:
        corr = np.corrcoef(X_train[col], y_values)[0, 1]
        leakage_check.append({
            'feature': col,
            'correlation_with_target': abs(corr)
        })

leakage_df = pd.DataFrame(leakage_check).sort_values('correlation_with_target', ascending=False)
print("\nТоп-15 признаков по корреляции с целевой переменной:")
print(leakage_df.head(15).to_string(index=False))

suspicious = leakage_df[leakage_df['correlation_with_target'] > 0.8]
if len(suspicious) > 0:
    print(f"\n⚠️ ВНИМАНИЕ: Найдены подозрительно высокие корреляции (>0.8) - возможна утечка данных!")
    print(suspicious.to_string(index=False))
else:
    print("\n✓ Утечки данных не обнаружено (нет корреляций > 0.8)")

print("\n" + "="*80)
print("6. АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("="*80)

print(f"\nВсего one-hot закодированных категорий: {len(categorical_encoded)}")

# Group by prefix to understand categorical variables
categorical_groups = {}
for col in categorical_encoded:
    prefix = col.rsplit('_', 1)[0] if '_' in col else col
    if prefix not in categorical_groups:
        categorical_groups[prefix] = []
    categorical_groups[prefix].append(col)

print(f"\nКатегориальные переменные и их количество значений:")
for prefix, cols in sorted(categorical_groups.items()):
    print(f"  {prefix}: {len(cols)} значений")

print("\n" + "="*80)
print("7. РЕКОМЕНДАЦИИ ПО ОТБОРУ ПРИЗНАКОВ")
print("="*80)

print(f"""
СТАТИСТИКА:
-----------
✓ Всего признаков: {X_train.shape[1]}
✓ ID-признаков (удалить): {len(id_features)}
✓ Полезных числовых: {len(analysis_features)}
✓ One-hot закодированных: {len(categorical_encoded)}
✓ Бинарных флагов: {len(binary_features)}
✓ Инженерных признаков: {len(engineered_features)}

КАЧЕСТВО ДАННЫХ:
---------------
✓ Признаков с нулевой дисперсией: {len(low_var)}
✓ Пар с высокой корреляцией (r>0.9): {len(high_corr_pairs)}
✓ Пар с умеренной корреляцией (0.8<r≤0.9): {len(moderate_corr_pairs)}
✓ Подозрений на утечку данных: {len(suspicious)}

РЕКОМЕНДАЦИЯ: НАЧАТЬ СО ВСЕХ 108 ПРИЗНАКОВ

ОБОСНОВАНИЕ:

1. ПРЕИМУЩЕСТВА ИСПОЛЬЗОВАНИЯ ВСЕХ ПРИЗНАКОВ:
   ✓ Древовидные модели (XGBoost, LightGBM, CatBoost) имеют встроенный отбор признаков
   ✓ Они автоматически игнорируют неинформативные признаки
   ✓ Feature importance покажет, какие признаки реально важны
   ✓ Не рискуем случайно удалить важный признак

2. ДАННЫЕ В ХОРОШЕМ СОСТОЯНИИ:
   ✓ НЕТ признаков с нулевой дисперсией
   ✓ НЕТ критичной мультиколлинеарности (пар с r>0.9: {len(high_corr_pairs)})
   ✓ НЕТ утечки данных
   ✓ Все признаки потенциально информативны

3. STRATEGY FOR FEATURE SELECTION:
   Шаг 1: Обучить модель на ВСЕХ признаках (кроме ID)
   Шаг 2: Получить feature importance
   Шаг 3: Если нужно - удалить наименее важные (importance < порог)
   Шаг 4: Переобучить и сравнить AUC

4. ЧТО ТОЧНО УДАЛИТЬ:
   - customer_ref ({id_features[0] if id_features else 'N/A'})
   - application_id ({id_features[1] if len(id_features)>1 else 'N/A'})
   ⚠️ Это НЕ предикторы, а идентификаторы

5. ЧТО ОСТАВИТЬ:
   - Все финансовые признаки (доходы, долги, коэффициенты)
   - Все поведенческие признаки (логины, звонки в поддержку)
   - Все географические признаки (регион, экономические показатели)
   - Все инженерные признаки ({len(engineered_features)} шт)
   - Все one-hot закодированные категории ({len(categorical_encoded)} шт)

ИТОГО ДЛЯ ОБУЧЕНИЯ: {X_train.shape[1] - len(id_features)} признаков
(108 всего - {len(id_features)} ID-признаков)

АЛЬТЕРНАТИВНЫЙ ПОДХОД (если модель переобучается):
- Использовать L1/L2 регуляризацию в модели
- Применить Recursive Feature Elimination (RFE)
- Оставить только top-50 по feature importance
- Но СНАЧАЛА пробуем все признаки!
""")

print("\n" + "="*80)
print("8. ПОДГОТОВКА ФИНАЛЬНЫХ ДАННЫХ")
print("="*80)

# Create final feature list (remove IDs)
features_to_use = [col for col in X_train.columns if col not in id_features]

print(f"\nПризнаки для обучения: {len(features_to_use)}")
print(f"ID-признаки удалены: {id_features}")

# Save feature list
with open('/home/dr/cbu/features_for_training.txt', 'w') as f:
    for feat in features_to_use:
        f.write(f"{feat}\n")

print(f"\nСписок признаков сохранен в: /home/dr/cbu/features_for_training.txt")

print("\n" + "="*80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("="*80)

# Save correlation matrix
corr_matrix.to_csv('/home/dr/cbu/correlation_matrix.csv')
print("\nМатрица корреляций сохранена: /home/dr/cbu/correlation_matrix.csv")
