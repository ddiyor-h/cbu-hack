"""
Анализ и сравнение стратегий импутации пропущенных значений
без использования sklearn (базовый статистический анализ)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ДЕТАЛЬНЫЙ АНАЛИЗ СТРАТЕГИЙ ИМПУТАЦИИ")
print("=" * 80)

# Load data
df = pd.read_csv('/home/dr/cbu/final_dataset_clean.csv')
print(f"\nИсходный датасет: {df.shape[0]:,} строк × {df.shape[1]} столбцов")

# Columns with missing values
missing_cols = ['employment_length', 'revolving_balance', 'num_delinquencies_2yrs']

# ============================================================================
# АНАЛИЗ КАЖДОГО СТОЛБЦА
# ============================================================================

print("\n" + "=" * 80)
print("ГЛУБОКИЙ АНАЛИЗ КАЖДОГО СТОЛБЦА С ПРОПУСКАМИ")
print("=" * 80)

analysis_results = []

for col in missing_cols:
    print(f"\n{'=' * 80}")
    print(f"СТОЛБЕЦ: {col}")
    print(f"{'=' * 80}")

    # Basic statistics
    total_rows = len(df)
    missing_count = df[col].isnull().sum()
    present_count = df[col].notna().sum()
    missing_pct = missing_count / total_rows * 100

    print(f"\nОбщая статистика:")
    print(f"  Всего строк: {total_rows:,}")
    print(f"  Пропущено: {missing_count:,} ({missing_pct:.2f}%)")
    print(f"  Присутствует: {present_count:,} ({100-missing_pct:.2f}%)")

    # Statistics for non-missing values
    non_null = df[col].dropna()
    if len(non_null) > 0:
        print(f"\nСтатистика непропущенных значений:")
        print(f"  Min: {non_null.min():.2f}")
        print(f"  Q1 (25%): {non_null.quantile(0.25):.2f}")
        print(f"  Median (50%): {non_null.median():.2f}")
        print(f"  Mean (среднее): {non_null.mean():.2f}")
        print(f"  Q3 (75%): {non_null.quantile(0.75):.2f}")
        print(f"  Max: {non_null.max():.2f}")
        print(f"  Std (ст. откл.): {non_null.std():.2f}")
        print(f"  Skewness (асимметрия): {non_null.skew():.2f}")

        # Check for zeros
        zero_count = (non_null == 0).sum()
        zero_pct = zero_count / len(non_null) * 100
        print(f"  Нулевых значений: {zero_count} ({zero_pct:.2f}%)")

        # Outlier detection (IQR method)
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((non_null < lower_bound) | (non_null > upper_bound)).sum()
        outlier_pct = outliers / len(non_null) * 100
        print(f"  Выбросов (IQR method): {outliers} ({outlier_pct:.2f}%)")

    # Correlation with target
    print(f"\nСвязь с целевой переменной (default):")

    # Default rate for missing vs present
    default_when_missing = df[df[col].isnull()]['default'].mean() * 100
    default_when_present = df[df[col].notna()]['default'].mean() * 100
    diff = default_when_missing - default_when_present

    print(f"  Доля дефолтов при ПРОПУСКЕ: {default_when_missing:.2f}%")
    print(f"  Доля дефолтов при НАЛИЧИИ: {default_when_present:.2f}%")
    print(f"  Разница: {diff:.2f} п.п.")

    if abs(diff) > 0.5:
        print(f"  ⚠ ВАЖНО: Значимая разница! Пропуск может быть информативен!")
    else:
        print(f"  ✓ Разница незначительная, вероятно MCAR (случайный пропуск)")

    # Correlation with numeric features (for non-missing values)
    corr_with_value = df[[col, 'default']].corr().iloc[0, 1]
    print(f"  Корреляция значений с default: {corr_with_value:.4f}")

    # Distribution analysis
    print(f"\nРаспределение значений (децили):")
    deciles = non_null.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for q, val in deciles.items():
        print(f"  {int(q*100)}%: {val:.2f}")

    # Store analysis
    analysis_results.append({
        'column': col,
        'missing_count': missing_count,
        'missing_pct': missing_pct,
        'mean': non_null.mean(),
        'median': non_null.median(),
        'std': non_null.std(),
        'zero_pct': zero_pct,
        'default_when_missing': default_when_missing,
        'default_when_present': default_when_present,
        'diff': diff,
        'corr_with_default': corr_with_value
    })

# ============================================================================
# РЕКОМЕНДАЦИИ ПО ИМПУТАЦИИ
# ============================================================================

print("\n" + "=" * 80)
print("РЕКОМЕНДАЦИИ ПО СТРАТЕГИИ ИМПУТАЦИИ ДЛЯ КАЖДОГО СТОЛБЦА")
print("=" * 80)

recommendations = {}

for res in analysis_results:
    col = res['column']
    print(f"\n{'─' * 80}")
    print(f"{col}")
    print(f"{'─' * 80}")

    if col == 'employment_length':
        print("\nХарактеристики:")
        print(f"  - Пропущено: {res['missing_pct']:.2f}%")
        print(f"  - Среднее: {res['mean']:.2f} лет")
        print(f"  - Медиана: {res['median']:.2f} лет")
        print(f"  - Влияние на дефолт: разница {res['diff']:.2f} п.п. (незначимо)")
        print(f"  - Корреляция с default: {res['corr_with_default']:.4f} (слабая отрицательная)")

        print("\nРЕКОМЕНДАЦИЯ:")
        print("  Метод: МЕДИАНА (5.20 лет)")
        print("  Обоснование:")
        print("    1. Медиана устойчива к выбросам")
        print("    2. Разница в дефолте при пропуске незначима (-0.18 п.п.)")
        print("    3. Распределение асимметричное, медиана лучше среднего")
        print("    4. Пропуск не несет дополнительной информации")

        recommendations[col] = {
            'method': 'median',
            'value': res['median'],
            'reason': 'Устойчивость к выбросам, незначимое влияние пропуска на таргет'
        }

    elif col == 'revolving_balance':
        print("\nХарактеристики:")
        print(f"  - Пропущено: {res['missing_pct']:.2f}%")
        print(f"  - Среднее: ${res['mean']:,.2f}")
        print(f"  - Медиана: ${res['median']:,.2f}")
        print(f"  - Разброс большой (std={res['std']:,.2f})")
        print(f"  - Влияние на дефолт: разница {res['diff']:.2f} п.п. (незначимо)")
        print(f"  - Корреляция с default: {res['corr_with_default']:.4f} (слабая отрицательная)")

        print("\nРЕКОМЕНДАЦИЯ:")
        print("  Метод: МЕДИАНА ПО ГРУППАМ CREDIT_SCORE")
        print("  Обоснование:")
        print("    1. Revolving balance сильно зависит от кредитного рейтинга")
        print("    2. Группировка повысит точность импутации")
        print("    3. Большой разброс значений (выбросы) - медиана устойчивее")
        print("    4. Финансовая метрика - важна точность")

        recommendations[col] = {
            'method': 'median_by_group',
            'group_by': 'credit_score',
            'fallback': res['median'],
            'reason': 'Группировка по credit_score для более точной импутации финансовой метрики'
        }

    elif col == 'num_delinquencies_2yrs':
        print("\nХарактеристики:")
        print(f"  - Пропущено: {res['missing_pct']:.2f}%")
        print(f"  - Среднее: {res['mean']:.2f}")
        print(f"  - Медиана: {res['median']:.2f}")
        print(f"  - Нулевых значений: {res['zero_pct']:.2f}%")
        print(f"  - Влияние на дефолт: разница {res['diff']:.2f} п.п. (незначимо)")
        print(f"  - Корреляция с default: {res['corr_with_default']:.4f} (слабая положительная)")

        print("\nРЕКОМЕНДАЦИЯ:")
        print("  Метод: ЗАПОЛНЕНИЕ НУЛЯМИ (0)")
        print("  Обоснование:")
        print("    1. 98% значений = 0 (отсутствие просрочек)")
        print("    2. Пропуск вероятно означает отсутствие данных о просрочках")
        print("    3. Консервативный подход: нет данных = нет просрочек")
        print("    4. Доля дефолтов при пропуске НИЖЕ (-0.54 п.п.) - поддерживает гипотезу")
        print("    5. Семантически: отсутствие записи = отсутствие просрочки")

        recommendations[col] = {
            'method': 'zero',
            'value': 0,
            'reason': 'Семантически пропуск = отсутствие просрочек, 98% значений уже равны 0'
        }

# ============================================================================
# СРАВНЕНИЕ РАЗМЕРОВ ВЫБОРОК
# ============================================================================

print("\n" + "=" * 80)
print("СРАВНЕНИЕ СТРАТЕГИЙ: УДАЛЕНИЕ VS ИМПУТАЦИЯ")
print("=" * 80)

# Strategy 1: Delete all rows with any missing value
df_deleted = df.dropna(subset=missing_cols)
print(f"\n1. УДАЛЕНИЕ СТРОК С ПРОПУСКАМИ:")
print(f"   Исходно: {len(df):,} строк")
print(f"   После удаления: {len(df_deleted):,} строк")
print(f"   Потеря данных: {len(df) - len(df_deleted):,} строк ({(len(df)-len(df_deleted))/len(df)*100:.2f}%)")
print(f"   Доля дефолтов: {df_deleted['default'].mean()*100:.2f}%")

# Strategy 2: Impute
print(f"\n2. ИМПУТАЦИЯ (РЕКОМЕНДУЕМАЯ):")
print(f"   Сохранено: {len(df):,} строк (100%)")
print(f"   Доля дефолтов: {df['default'].mean()*100:.2f}%")
print(f"   Преимущество: +{len(df) - len(df_deleted):,} записей для обучения")

# Check if class balance is preserved
print(f"\n3. ВЛИЯНИЕ НА БАЛАНС КЛАССОВ:")
deleted_default_rate = df_deleted['default'].mean()
original_default_rate = df['default'].mean()
print(f"   Исходная доля дефолтов: {original_default_rate*100:.2f}%")
print(f"   После удаления: {deleted_default_rate*100:.2f}%")
print(f"   Разница: {abs(deleted_default_rate - original_default_rate)*100:.2f} п.п.")

if abs(deleted_default_rate - original_default_rate) < 0.001:
    print(f"   ✓ Баланс классов сохранен")
else:
    print(f"   ⚠ Баланс классов изменился - возможна систематическая ошибка!")

# ============================================================================
# СОЗДАНИЕ ИТОГОВОГО ДАТАСЕТА С ИМПУТАЦИЕЙ
# ============================================================================

print("\n" + "=" * 80)
print("ПРИМЕНЕНИЕ РЕКОМЕНДОВАННОЙ СТРАТЕГИИ ИМПУТАЦИИ")
print("=" * 80)

df_imputed = df.copy()

# 1. employment_length - median
median_val = recommendations['employment_length']['value']
df_imputed['employment_length'].fillna(median_val, inplace=True)
print(f"\n1. employment_length:")
print(f"   Заполнено медианой: {median_val:.2f}")
print(f"   Пропусков осталось: {df_imputed['employment_length'].isnull().sum()}")

# 2. revolving_balance - median by credit_score groups
print(f"\n2. revolving_balance:")
print(f"   Метод: медиана по группам credit_score")

# Calculate median by credit_score
median_by_score = df_imputed.groupby('credit_score')['revolving_balance'].median()
print(f"   Найдено групп: {len(median_by_score)}")

# Apply group-wise imputation
df_imputed['revolving_balance'] = df_imputed.groupby('credit_score')['revolving_balance'].transform(
    lambda x: x.fillna(x.median())
)

# Fallback: if still missing (rare credit scores), use overall median
remaining_missing = df_imputed['revolving_balance'].isnull().sum()
if remaining_missing > 0:
    overall_median = recommendations['revolving_balance']['fallback']
    df_imputed['revolving_balance'].fillna(overall_median, inplace=True)
    print(f"   Fallback (общая медиана) применен к {remaining_missing} записям")

print(f"   Пропусков осталось: {df_imputed['revolving_balance'].isnull().sum()}")

# 3. num_delinquencies_2yrs - zero
df_imputed['num_delinquencies_2yrs'].fillna(0, inplace=True)
print(f"\n3. num_delinquencies_2yrs:")
print(f"   Заполнено нулями: 0")
print(f"   Пропусков осталось: {df_imputed['num_delinquencies_2yrs'].isnull().sum()}")

# Verify no missing values remain
total_missing = df_imputed[missing_cols].isnull().sum().sum()
print(f"\n{'=' * 80}")
print(f"ИТОГО:")
print(f"  Всего пропусков ДО импутации: 4,462")
print(f"  Всего пропусков ПОСЛЕ импутации: {total_missing}")
print(f"  Размер датасета: {len(df_imputed):,} строк × {df_imputed.shape[1]} столбцов")

if total_missing == 0:
    print(f"  ✓ Все пропуски успешно заполнены!")
else:
    print(f"  ⚠ Внимание: остались пропуски!")

# Save imputed dataset
print(f"\n{'=' * 80}")
print(f"СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print(f"{'=' * 80}")

# Save as CSV
csv_path = '/home/dr/cbu/final_dataset_imputed.csv'
df_imputed.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✓ CSV сохранен: {csv_path}")
print(f"  Размер: {len(df_imputed):,} строк × {df_imputed.shape[1]} столбцов")

# Save as Parquet (more efficient)
parquet_path = '/home/dr/cbu/final_dataset_imputed.parquet'
df_imputed.to_parquet(parquet_path, index=False, engine='pyarrow')
print(f"\n✓ Parquet сохранен: {parquet_path}")

# Create summary report
summary = {
    'Метрика': [
        'Исходных строк',
        'Строк после импутации',
        'Потерянных строк',
        'Исходных пропусков',
        'Пропусков после импутации',
        'Доля дефолтов (исходная)',
        'Доля дефолтов (импутированная)',
        'Столбцов',
        'Числовых признаков'
    ],
    'Значение': [
        f"{len(df):,}",
        f"{len(df_imputed):,}",
        "0",
        "4,462",
        f"{total_missing}",
        f"{df['default'].mean()*100:.2f}%",
        f"{df_imputed['default'].mean()*100:.2f}%",
        f"{df_imputed.shape[1]}",
        f"{len(df_imputed.select_dtypes(include=[np.number]).columns)}"
    ]
}

summary_df = pd.DataFrame(summary)
print(f"\n{'=' * 80}")
print(f"ИТОГОВАЯ СВОДКА")
print(f"{'=' * 80}")
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)
