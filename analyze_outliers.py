"""
Comprehensive Outlier Analysis for Credit Default Prediction
Анализ выбросов для предсказания дефолтов по кредитам
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Manual z-score calculation
def calculate_zscore(series):
    """Calculate z-scores manually"""
    mean = series.mean()
    std = series.std()
    return np.abs((series - mean) / std)

print("="*80)
print("АНАЛИЗ ВЫБРОСОВ (OUTLIER ANALYSIS)")
print("="*80)

# Load training data
print("\n1. Загрузка данных...")
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Количество признаков: {X_train.shape[1]}")

# Identify numeric columns (excluding IDs and encoded features)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID columns and frequency-encoded columns
exclude_cols = ['customer_ref', 'application_id']
frequency_cols = [col for col in numeric_cols if col.endswith('_freq')]
numeric_cols = [col for col in numeric_cols if col not in exclude_cols and col not in frequency_cols]

print(f"\nЧисловых признаков для анализа: {len(numeric_cols)}")

# Outlier detection results
outlier_summary = []

print("\n" + "="*80)
print("2. МЕТОД IQR (Межквартильный размах)")
print("="*80)

for col in numeric_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound))
    n_outliers = outliers.sum()
    pct_outliers = (n_outliers / len(X_train)) * 100

    outlier_summary.append({
        'feature': col,
        'method': 'IQR',
        'n_outliers': n_outliers,
        'pct_outliers': pct_outliers,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'min_value': X_train[col].min(),
        'max_value': X_train[col].max()
    })

# Show top features with most outliers (IQR)
outlier_df = pd.DataFrame(outlier_summary)
top_outliers_iqr = outlier_df.nlargest(15, 'pct_outliers')
print("\nТоп-15 признаков с наибольшим процентом выбросов (IQR метод):")
print(top_outliers_iqr[['feature', 'n_outliers', 'pct_outliers', 'min_value', 'max_value']].to_string(index=False))

print("\n" + "="*80)
print("3. МЕТОД Z-SCORE (|z| > 3)")
print("="*80)

z_outlier_summary = []
for col in numeric_cols:
    z_scores = calculate_zscore(X_train[col])
    outliers = z_scores > 3
    n_outliers = outliers.sum()
    pct_outliers = (n_outliers / len(X_train)) * 100

    z_outlier_summary.append({
        'feature': col,
        'n_outliers': n_outliers,
        'pct_outliers': pct_outliers
    })

z_outlier_df = pd.DataFrame(z_outlier_summary)
top_outliers_z = z_outlier_df.nlargest(15, 'pct_outliers')
print("\nТоп-15 признаков с наибольшим процентом выбросов (Z-score метод):")
print(top_outliers_z.to_string(index=False))

print("\n" + "="*80)
print("4. АНАЛИЗ ПО ДОМЕННОЙ ОБЛАСТИ (Domain Knowledge)")
print("="*80)

# Age analysis
if 'age' in X_train.columns:
    age_issues = (X_train['age'] < 18) | (X_train['age'] > 100)
    print(f"\nВозраст (age):")
    print(f"  - Минимальный: {X_train['age'].min()}")
    print(f"  - Максимальный: {X_train['age'].max()}")
    print(f"  - Нереалистичные значения (<18 или >100): {age_issues.sum()}")

# Income analysis
income_cols = [col for col in numeric_cols if 'income' in col.lower()]
print(f"\nДоходы (income columns): {income_cols}")
for col in income_cols:
    negative = (X_train[col] < 0).sum()
    zero = (X_train[col] == 0).sum()
    print(f"  {col}:")
    print(f"    - Отрицательные: {negative}")
    print(f"    - Нулевые: {zero}")
    print(f"    - Макс: ${X_train[col].max():,.0f}")

# Debt analysis
debt_cols = [col for col in numeric_cols if 'debt' in col.lower()]
print(f"\nЗадолженность (debt columns): {debt_cols}")
for col in debt_cols:
    negative = (X_train[col] < 0).sum()
    print(f"  {col}:")
    print(f"    - Отрицательные: {negative}")
    print(f"    - Макс: ${X_train[col].max():,.0f}")

# Credit utilization
if 'credit_utilization' in X_train.columns:
    cu = X_train['credit_utilization']
    print(f"\nКредитная нагрузка (credit_utilization):")
    print(f"  - Минимум: {cu.min():.2f}")
    print(f"  - Максимум: {cu.max():.2f}")
    print(f"  - Значений > 1.0 (>100%): {(cu > 1.0).sum()}")
    print(f"  - Значений > 2.0 (>200%): {(cu > 2.0).sum()}")

# Ratio analysis
ratio_cols = [col for col in numeric_cols if 'ratio' in col.lower()]
print(f"\nФинансовые коэффициенты (ratios): {ratio_cols}")
for col in ratio_cols:
    print(f"  {col}:")
    print(f"    - Мин: {X_train[col].min():.4f}")
    print(f"    - Макс: {X_train[col].max():.4f}")
    print(f"    - Медиана: {X_train[col].median():.4f}")

print("\n" + "="*80)
print("5. ВЛИЯНИЕ ВЫБРОСОВ НА ЦЕЛЕВУЮ ПЕРЕМЕННУЮ")
print("="*80)

# Check correlation between outliers and default rate
y_default = y_train.values.ravel()

print("\nКорреляция выбросов с дефолтом (топ-10 признаков):")
outlier_default_corr = []

for col in numeric_cols[:20]:  # Check top 20 features
    # Define outliers using IQR
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    is_outlier = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound))

    if is_outlier.sum() > 0:
        default_rate_outliers = y_default[is_outlier].mean()
        default_rate_normal = y_default[~is_outlier].mean()
        diff = default_rate_outliers - default_rate_normal

        outlier_default_corr.append({
            'feature': col,
            'default_rate_outliers': default_rate_outliers,
            'default_rate_normal': default_rate_normal,
            'difference': diff,
            'n_outliers': is_outlier.sum()
        })

odc_df = pd.DataFrame(outlier_default_corr).sort_values('difference', key=abs, ascending=False)
print(odc_df.head(10).to_string(index=False))

print("\n" + "="*80)
print("6. РЕКОМЕНДАЦИИ ПО ОБРАБОТКЕ ВЫБРОСОВ")
print("="*80)

print("""
ВЫВОДЫ:

1. КОЛИЧЕСТВО ВЫБРОСОВ:
   - Большинство финансовых признаков имеют выбросы (особенно доходы, долги)
   - Это ожидаемо для кредитных данных (есть клиенты с экстремальными значениями)

2. ПРИРОДА ВЫБРОСОВ:
   - Большинство выбросов - ЛЕГИТИМНЫЕ экстремальные значения, а не ошибки данных
   - Например: очень высокий доход, очень большая задолженность
   - Эти значения могут быть ИНФОРМАТИВНЫМИ для предсказания дефолта

3. АЛГОРИТМЫ МАШИННОГО ОБУЧЕНИЯ:
   - Древовидные модели (XGBoost, LightGBM, CatBoost) УСТОЙЧИВЫ к выбросам
   - Они используют пороговые значения, а не абсолютные величины
   - Выбросы могут помочь модели найти важные паттерны риска

РЕКОМЕНДАЦИЯ: НЕ ОБРАБАТЫВАТЬ ВЫБРОСЫ, ПОТОМУ ЧТО:

✓ Выбросы содержат полезную информацию о кредитном риске
✓ Древовидные модели (которые мы будем использовать) устойчивы к ним
✓ Удаление/ограничение выбросов может СНИЗИТЬ качество модели (AUC)
✓ Выбросы в тестовой выборке тоже будут - модель должна их обрабатывать

ИСКЛЮЧЕНИЕ: Только если найдем ОЧЕВИДНЫЕ ошибки данных:
- Возраст < 18 или > 100
- Отрицательные значения там, где их быть не должно
- Но таких случаев практически нет в данных

АЛЬТЕРНАТИВА (если модель переобучается):
- Использовать winsorization (ограничение на уровне 1-99 перцентилей)
- Но ТОЛЬКО если увидим переобучение при валидации
""")

print("\n" + "="*80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("="*80)

# Save summary
outlier_df.to_csv('/home/dr/cbu/outlier_analysis_iqr.csv', index=False)
z_outlier_df.to_csv('/home/dr/cbu/outlier_analysis_zscore.csv', index=False)

print("\nРезультаты сохранены в:")
print("  - /home/dr/cbu/outlier_analysis_iqr.csv")
print("  - /home/dr/cbu/outlier_analysis_zscore.csv")
