"""
Глубокий анализ датасета для credit default prediction
Цель: Подготовить полную картину данных для ml-data-preparation-specialist и research-specialist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ГЛУБОКИЙ АНАЛИЗ ДАТАСЕТА - CREDIT DEFAULT PREDICTION")
print("="*100)

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n[1/10] Загрузка данных...")

X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Target: {y_train.shape}, {y_test.shape}")

# Объединяем для полного анализа
df_train = X_train.copy()
df_train['default'] = y_train['default'].values

# ============================================================================
# 2. БАЗОВАЯ СТАТИСТИКА
# ============================================================================

print("\n" + "="*100)
print("[2/10] БАЗОВАЯ СТАТИСТИКА ДАТАСЕТА")
print("="*100)

print(f"\n📊 РАЗМЕРЫ:")
print(f"   • Тренировочная выборка: {len(df_train):,} записей")
print(f"   • Тестовая выборка:      {len(X_test):,} записей")
print(f"   • Всего записей:         {len(df_train) + len(X_test):,}")
print(f"   • Признаков:             {X_train.shape[1]}")

# Баланс классов
default_rate_train = y_train['default'].mean()
default_rate_test = y_test['default'].mean()
imbalance_ratio = (1 - default_rate_train) / default_rate_train

print(f"\n⚖️  БАЛАНС КЛАССОВ:")
print(f"   • Train default rate:    {default_rate_train:.2%} ({y_train['default'].sum():,} дефолтов)")
print(f"   • Test default rate:     {default_rate_test:.2%} ({y_test['default'].sum():,} дефолтов)")
print(f"   • Imbalance ratio:       1:{imbalance_ratio:.1f}")
print(f"   • Scale_pos_weight:      {imbalance_ratio:.1f}")

# Типы данных
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\n📋 ТИПЫ ПРИЗНАКОВ:")
print(f"   • Числовых:      {len(numeric_features)}")
print(f"   • Категориальных: {len(categorical_features)}")

# ============================================================================
# 3. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ
# ============================================================================

print("\n" + "="*100)
print("[3/10] АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("="*100)

missing_train = X_train.isnull().sum()
missing_test = X_test.isnull().sum()

features_with_missing = missing_train[missing_train > 0].sort_values(ascending=False)

if len(features_with_missing) > 0:
    print(f"\n⚠️  Найдено {len(features_with_missing)} признаков с пропусками:")
    for feat, count in features_with_missing.head(10).items():
        pct = (count / len(X_train)) * 100
        print(f"   • {feat:50s}: {count:6,} ({pct:5.2f}%)")
else:
    print("\n✅ Пропущенных значений не найдено!")

# ============================================================================
# 4. АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ
# ============================================================================

print("\n" + "="*100)
print("[4/10] АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("="*100)

numeric_stats = X_train[numeric_features].describe().T
numeric_stats['skewness'] = X_train[numeric_features].skew()
numeric_stats['kurtosis'] = X_train[numeric_features].kurtosis()
numeric_stats['zeros_pct'] = (X_train[numeric_features] == 0).sum() / len(X_train) * 100

print(f"\n📊 Всего числовых признаков: {len(numeric_features)}")

# Ищем признаки с выбросами (high skewness)
high_skew = numeric_stats[abs(numeric_stats['skewness']) > 3].sort_values('skewness', key=abs, ascending=False)

if len(high_skew) > 0:
    print(f"\n⚠️  Признаки с высокой асимметрией (|skew| > 3): {len(high_skew)}")
    for feat in high_skew.head(10).index:
        print(f"   • {feat:50s}: skew={numeric_stats.loc[feat, 'skewness']:8.2f}")

# Ищем константные/почти константные признаки
low_variance = numeric_stats[numeric_stats['std'] < 0.01]
if len(low_variance) > 0:
    print(f"\n⚠️  Признаки с очень низкой вариативностью (std < 0.01): {len(low_variance)}")
    for feat in low_variance.index[:10]:
        print(f"   • {feat:50s}: std={numeric_stats.loc[feat, 'std']:.6f}")

# Признаки с большим количеством нулей
high_zeros = numeric_stats[numeric_stats['zeros_pct'] > 50].sort_values('zeros_pct', ascending=False)
if len(high_zeros) > 0:
    print(f"\n⚠️  Признаки с >50% нулевых значений: {len(high_zeros)}")
    for feat in high_zeros.head(10).index:
        print(f"   • {feat:50s}: {numeric_stats.loc[feat, 'zeros_pct']:.1f}% нулей")

# Сохраняем статистику
numeric_stats.to_csv('/home/dr/cbu/numeric_features_statistics.csv')
print(f"\n💾 Статистика сохранена: numeric_features_statistics.csv")

# ============================================================================
# 5. АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ============================================================================

print("\n" + "="*100)
print("[5/10] АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("="*100)

if len(categorical_features) > 0:
    print(f"\n📊 Всего категориальных признаков: {len(categorical_features)}")

    cat_stats = []
    for feat in categorical_features:
        n_unique = X_train[feat].nunique()
        most_common = X_train[feat].mode()[0] if len(X_train[feat].mode()) > 0 else None
        most_common_pct = (X_train[feat] == most_common).sum() / len(X_train) * 100 if most_common else 0

        cat_stats.append({
            'feature': feat,
            'unique_values': n_unique,
            'most_common': most_common,
            'most_common_pct': most_common_pct
        })

    cat_df = pd.DataFrame(cat_stats).sort_values('unique_values', ascending=False)

    print("\n📋 Топ-10 по количеству уникальных значений:")
    for idx, row in cat_df.head(10).iterrows():
        print(f"   • {row['feature']:50s}: {row['unique_values']:6,} уникальных")

    # Признаки с высокой кардинальностью (>100 уникальных)
    high_cardinality = cat_df[cat_df['unique_values'] > 100]
    if len(high_cardinality) > 0:
        print(f"\n⚠️  Признаки с высокой кардинальностью (>100): {len(high_cardinality)}")

    cat_df.to_csv('/home/dr/cbu/categorical_features_statistics.csv', index=False)
    print(f"\n💾 Статистика категориальных признаков: categorical_features_statistics.csv")
else:
    print("\n📋 Категориальных признаков нет (все one-hot encoded)")

# ============================================================================
# 6. РАСПРЕДЕЛЕНИЕ ПРИЗНАКОВ ПО КЛАССАМ
# ============================================================================

print("\n" + "="*100)
print("[6/10] РАСПРЕДЕЛЕНИЕ ПРИЗНАКОВ ПО КЛАССАМ")
print("="*100)

# Загружаем корреляции
target_corr = pd.read_csv('/home/dr/cbu/target_correlations.csv')
top_features = target_corr.head(20)['feature'].tolist()

print(f"\n🔍 Анализируем топ-20 признаков по корреляции с default...")

class_separation = []

for feat in top_features[:10]:  # Берем топ-10 для детального анализа
    if feat not in X_train.columns:
        continue

    default_vals = df_train[df_train['default'] == 1][feat]
    no_default_vals = df_train[df_train['default'] == 0][feat]

    # KS-тест для оценки разделимости
    ks_stat, ks_pval = ks_2samp(default_vals.dropna(), no_default_vals.dropna())

    # Разница медиан
    median_diff = default_vals.median() - no_default_vals.median()
    median_diff_pct = (median_diff / no_default_vals.median() * 100) if no_default_vals.median() != 0 else 0

    class_separation.append({
        'feature': feat,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'median_default': default_vals.median(),
        'median_no_default': no_default_vals.median(),
        'median_diff_pct': median_diff_pct
    })

sep_df = pd.DataFrame(class_separation).sort_values('ks_statistic', ascending=False)

print("\n📊 РАЗДЕЛИМОСТЬ КЛАССОВ (Kolmogorov-Smirnov):")
print("="*100)
for idx, row in sep_df.iterrows():
    print(f"{row['feature']:50s}: KS={row['ks_statistic']:.4f}, Median Δ={row['median_diff_pct']:+7.1f}%")

sep_df.to_csv('/home/dr/cbu/class_separation_analysis.csv', index=False)
print(f"\n💾 Анализ разделимости: class_separation_analysis.csv")

# ============================================================================
# 7. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ - ИНТЕРАКЦИИ
# ============================================================================

print("\n" + "="*100)
print("[7/10] АНАЛИЗ ПОТЕНЦИАЛЬНЫХ ИНТЕРАКЦИЙ")
print("="*100)

# Берем топ признаки для поиска интеракций
top_10_features = target_corr.head(10)['feature'].tolist()
top_10_features = [f for f in top_10_features if f in X_train.columns]

print(f"\n🔍 Поиск перспективных интеракций среди топ-10 признаков...")

interaction_candidates = []

for i, feat1 in enumerate(top_10_features):
    for feat2 in top_10_features[i+1:]:
        # Создаем интеракцию
        interaction = df_train[feat1] * df_train[feat2]

        # Корреляция интеракции с таргетом
        corr_with_target = interaction.corr(df_train['default'])

        # Корреляция с исходными признаками
        corr_with_feat1 = interaction.corr(df_train[feat1])
        corr_with_feat2 = interaction.corr(df_train[feat2])

        # Интеракция интересна, если она сильно коррелирует с таргетом
        # но не полностью дублирует исходные признаки
        if abs(corr_with_target) > 0.1 and abs(corr_with_feat1) < 0.95 and abs(corr_with_feat2) < 0.95:
            interaction_candidates.append({
                'feature1': feat1,
                'feature2': feat2,
                'interaction_corr_target': corr_with_target,
                'corr_feat1': corr_with_feat1,
                'corr_feat2': corr_with_feat2,
                'interaction_name': f'{feat1}_x_{feat2}'
            })

if len(interaction_candidates) > 0:
    int_df = pd.DataFrame(interaction_candidates).sort_values('interaction_corr_target',
                                                               key=abs, ascending=False)

    print(f"\n✅ Найдено {len(int_df)} перспективных интеракций:")
    for idx, row in int_df.head(15).iterrows():
        print(f"   • {row['feature1']:30s} × {row['feature2']:30s}: corr={row['interaction_corr_target']:+.4f}")

    int_df.to_csv('/home/dr/cbu/interaction_candidates.csv', index=False)
    print(f"\n💾 Кандидаты на интеракции: interaction_candidates.csv")
else:
    print("\n⚠️  Перспективных интеракций не найдено")

# ============================================================================
# 8. ВРЕМЕННОЙ АНАЛИЗ
# ============================================================================

print("\n" + "="*100)
print("[8/10] ВРЕМЕННОЙ АНАЛИЗ (ACCOUNT_OPEN_YEAR)")
print("="*100)

if 'account_open_year' in X_train.columns:
    print("\n📅 Анализ распределения по годам...")

    temporal_stats = df_train.groupby('account_open_year').agg({
        'default': ['count', 'sum', 'mean']
    }).round(4)

    temporal_stats.columns = ['count', 'defaults', 'default_rate']
    temporal_stats['default_rate_pct'] = temporal_stats['default_rate'] * 100

    print("\n📊 РАСПРЕДЕЛЕНИЕ ПО ГОДАМ:")
    print(temporal_stats.to_string())

    # Проверяем тренд дефолтности
    years = temporal_stats.index.values
    default_rates = temporal_stats['default_rate'].values

    from scipy.stats import pearsonr
    corr_year_default, pval = pearsonr(years, default_rates)

    print(f"\n📈 ВРЕМЕННОЙ ТРЕНД:")
    print(f"   • Корреляция год-дефолт: {corr_year_default:+.4f} (p={pval:.4f})")

    if abs(corr_year_default) > 0.3:
        trend = "РАСТЕТ" if corr_year_default > 0 else "ПАДАЕТ"
        print(f"   • ⚠️  Дефолтность {trend} со временем!")
    else:
        print(f"   • ✅ Дефолтность стабильна во времени")

    temporal_stats.to_csv('/home/dr/cbu/temporal_analysis.csv')
    print(f"\n💾 Временной анализ: temporal_analysis.csv")
else:
    print("\n⚠️  account_open_year не найден в признаках")

# ============================================================================
# 9. FEATURE IMPORTANCE (базовая оценка)
# ============================================================================

print("\n" + "="*100)
print("[9/10] ОЦЕНКА ВАЖНОСТИ ПРИЗНАКОВ")
print("="*100)

# Используем корреляции как прокси для важности
feature_importance = target_corr.copy()
feature_importance['importance_score'] = feature_importance['abs_correlation']

# Добавляем информацию о типе признака
feature_importance['is_numeric'] = feature_importance['feature'].apply(
    lambda x: x in numeric_features
)

print("\n📊 ТОП-30 НАИБОЛЕЕ ВАЖНЫХ ПРИЗНАКОВ (по корреляции):")
print("="*100)
for i, (idx, row) in enumerate(feature_importance.head(30).iterrows(), 1):
    feat_type = "NUM" if row['is_numeric'] else "CAT"
    print(f"{i:2d}. [{feat_type}] {row['feature']:50s}: {row['correlation_with_default']:+.4f}")

# ============================================================================
# 10. ИТОГОВЫЙ ОТЧЕТ И РЕКОМЕНДАЦИИ
# ============================================================================

print("\n" + "="*100)
print("[10/10] ИТОГОВЫЙ ОТЧЕТ И РЕКОМЕНДАЦИИ")
print("="*100)

# Собираем все находки
findings = {
    'dataset_size': {
        'train': len(df_train),
        'test': len(X_test),
        'features': X_train.shape[1]
    },
    'class_balance': {
        'default_rate_train': default_rate_train,
        'default_rate_test': default_rate_test,
        'imbalance_ratio': imbalance_ratio
    },
    'feature_types': {
        'numeric': len(numeric_features),
        'categorical': len(categorical_features)
    },
    'data_quality': {
        'features_with_missing': len(features_with_missing),
        'high_skew_features': len(high_skew),
        'low_variance_features': len(low_variance),
        'high_zeros_features': len(high_zeros)
    },
    'potential_improvements': {
        'interaction_candidates': len(interaction_candidates) if len(interaction_candidates) > 0 else 0,
        'high_separation_features': len(sep_df[sep_df['ks_statistic'] > 0.3])
    }
}

import json
with open('/home/dr/cbu/dataset_analysis_summary.json', 'w') as f:
    json.dump(findings, f, indent=2)

print("\n📊 КЛЮЧЕВЫЕ НАХОДКИ:")
print("="*100)
print(f"\n✅ РАЗМЕР ДАННЫХ:")
print(f"   • Train: {findings['dataset_size']['train']:,}, Test: {findings['dataset_size']['test']:,}")
print(f"   • Признаков: {findings['dataset_size']['features']}")

print(f"\n⚖️  БАЛАНС КЛАССОВ:")
print(f"   • Default rate: {findings['class_balance']['default_rate_train']:.2%}")
print(f"   • Imbalance: 1:{findings['class_balance']['imbalance_ratio']:.1f}")

print(f"\n🔍 КАЧЕСТВО ДАННЫХ:")
print(f"   • Признаков с пропусками: {findings['data_quality']['features_with_missing']}")
print(f"   • С высокой асимметрией: {findings['data_quality']['high_skew_features']}")
print(f"   • С низкой вариативностью: {findings['data_quality']['low_variance_features']}")
print(f"   • С >50% нулей: {findings['data_quality']['high_zeros_features']}")

print(f"\n💡 ВОЗМОЖНОСТИ ДЛЯ УЛУЧШЕНИЯ:")
print(f"   • Кандидатов на интеракции: {findings['potential_improvements']['interaction_candidates']}")
print(f"   • Признаков с хорошей разделимостью: {findings['potential_improvements']['high_separation_features']}")

print("\n" + "="*100)
print("📁 СОЗДАННЫЕ ФАЙЛЫ:")
print("="*100)
print("   1. numeric_features_statistics.csv       - Статистика числовых признаков")
print("   2. categorical_features_statistics.csv   - Статистика категориальных признаков")
print("   3. class_separation_analysis.csv         - Анализ разделимости классов")
print("   4. interaction_candidates.csv            - Кандидаты на интеракции")
print("   5. temporal_analysis.csv                 - Временной анализ")
print("   6. dataset_analysis_summary.json         - Итоговый JSON отчет")

print("\n" + "="*100)
print("✅ ГЛУБОКИЙ АНАЛИЗ ЗАВЕРШЕН!")
print("="*100)

print("\n💡 РЕКОМЕНДАЦИИ ДЛЯ СЛЕДУЮЩИХ ШАГОВ:")
print("   1. Запустить ml-data-preparation-specialist для feature engineering")
print("   2. Запустить research-specialist для подбора оптимальных алгоритмов")
print("   3. Протестировать интеракции из interaction_candidates.csv")
print("   4. Рассмотреть binning для признаков с высокой асимметрией")
print("   5. Применить advanced sampling techniques для class imbalance")
