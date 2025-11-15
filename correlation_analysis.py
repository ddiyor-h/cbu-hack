"""
Корреляционный анализ тренировочного датасета
Создает correlation matrix и визуализации для данных, использованных при обучении модели
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ТРЕНИРОВОЧНОГО ДАТАСЕТА")
print("="*80)

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n[1/5] Загрузка тренировочных данных...")

# Загружаем тренировочные данные
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')

print(f"✅ X_train загружен: {X_train.shape}")
print(f"✅ y_train загружен: {y_train.shape}")

# Объединяем признаки и таргет для полного анализа
df_train = X_train.copy()
df_train['default'] = y_train['default'].values

print(f"\n📊 Итоговый датасет для анализа: {df_train.shape}")
print(f"   - Признаков: {df_train.shape[1] - 1}")
print(f"   - Записей: {df_train.shape[0]:,}")

# ============================================================================
# 2. ВЫЧИСЛЕНИЕ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ
# ============================================================================

print("\n[2/5] Вычисление корреляционной матрицы...")

# Полная корреляционная матрица
correlation_matrix = df_train.corr(method='pearson')

print(f"✅ Корреляционная матрица вычислена: {correlation_matrix.shape}")

# Сохраняем полную матрицу в CSV
correlation_matrix.to_csv('/home/dr/cbu/correlation_matrix_full.csv')
print(f"💾 Полная матрица сохранена: correlation_matrix_full.csv")

# ============================================================================
# 3. АНАЛИЗ КОРРЕЛЯЦИЙ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ============================================================================

print("\n[3/5] Анализ корреляций с целевой переменной (default)...")

# Корреляции с таргетом
target_correlations = correlation_matrix['default'].drop('default').sort_values(ascending=False)

print(f"\n📈 ТОП-20 ПОЛОЖИТЕЛЬНЫХ КОРРЕЛЯЦИЙ С DEFAULT:")
print("="*80)
for i, (feature, corr) in enumerate(target_correlations.head(20).items(), 1):
    print(f"{i:2d}. {feature:50s} : {corr:+.4f}")

print(f"\n📉 ТОП-20 ОТРИЦАТЕЛЬНЫХ КОРРЕЛЯЦИЙ С DEFAULT:")
print("="*80)
for i, (feature, corr) in enumerate(target_correlations.tail(20).items(), 1):
    print(f"{i:2d}. {feature:50s} : {corr:+.4f}")

# Сохраняем корреляции с таргетом
target_corr_df = pd.DataFrame({
    'feature': target_correlations.index,
    'correlation_with_default': target_correlations.values,
    'abs_correlation': np.abs(target_correlations.values)
}).sort_values('abs_correlation', ascending=False)

target_corr_df.to_csv('/home/dr/cbu/target_correlations.csv', index=False)
print(f"\n💾 Корреляции с таргетом сохранены: target_correlations.csv")

# ============================================================================
# 4. ПОИСК МУЛЬТИКОЛЛИНЕАРНОСТИ
# ============================================================================

print("\n[4/5] Поиск сильно коррелированных пар признаков (мультиколлинеарность)...")

# Находим пары признаков с высокой корреляцией (>0.8 или <-0.8)
# Исключаем диагональ и дубликаты
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]

        # Пропускаем корреляции с самим default
        if col1 == 'default' or col2 == 'default':
            continue

        if abs(corr_value) > 0.8:
            high_corr_pairs.append({
                'feature_1': col1,
                'feature_2': col2,
                'correlation': corr_value
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation',
                                                          key=abs,
                                                          ascending=False)

print(f"\n⚠️  Найдено {len(high_corr_df)} пар с |корреляцией| > 0.8:")
if len(high_corr_df) > 0:
    print("="*80)
    for idx, row in high_corr_df.head(20).iterrows():
        print(f"{row['feature_1']:40s} ↔ {row['feature_2']:40s} : {row['correlation']:+.4f}")

high_corr_df.to_csv('/home/dr/cbu/high_correlations_multicollinearity.csv', index=False)
print(f"\n💾 Мультиколлинеарность сохранена: high_correlations_multicollinearity.csv")

# ============================================================================
# 5. ВИЗУАЛИЗАЦИИ
# ============================================================================

print("\n[5/5] Создание визуализаций...")

# Создаем фигуру с 3 подграфиками
fig = plt.figure(figsize=(24, 18))

# ============================================================================
# График 1: Тепловая карта топ-30 признаков по корреляции с default
# ============================================================================

ax1 = plt.subplot(3, 2, 1)

# Выбираем топ-30 признаков по абсолютной корреляции с default
top_features = target_corr_df.head(30)['feature'].tolist()
top_features_with_target = top_features + ['default']

# Создаем матрицу корреляций только для этих признаков
corr_top = df_train[top_features_with_target].corr()

# Рисуем heatmap
sns.heatmap(corr_top,
            annot=False,  # Не показываем числа (их слишком много)
            cmap='RdBu_r',  # Красно-синяя палитра
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            ax=ax1)

ax1.set_title('Correlation Heatmap: Top-30 Features (by correlation with default)',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('')
ax1.set_ylabel('')

# Поворачиваем метки
plt.setp(ax1.get_xticklabels(), rotation=90, ha='right', fontsize=8)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)

# ============================================================================
# График 2: Bar plot корреляций с default (топ-30)
# ============================================================================

ax2 = plt.subplot(3, 2, 2)

top_30_corr = target_corr_df.head(30).copy()
colors = ['#d73027' if x > 0 else '#4575b4' for x in top_30_corr['correlation_with_default']]

ax2.barh(range(len(top_30_corr)),
         top_30_corr['correlation_with_default'].values,
         color=colors,
         alpha=0.7,
         edgecolor='black',
         linewidth=0.5)

ax2.set_yticks(range(len(top_30_corr)))
ax2.set_yticklabels(top_30_corr['feature'], fontsize=8)
ax2.set_xlabel('Correlation with Default', fontsize=11, fontweight='bold')
ax2.set_title('Top-30 Features by Absolute Correlation with Default',
              fontsize=14, fontweight='bold', pad=15)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Добавляем значения
for i, (idx, row) in enumerate(top_30_corr.iterrows()):
    value = row['correlation_with_default']
    x_pos = value + 0.005 if value > 0 else value - 0.005
    ha = 'left' if value > 0 else 'right'
    ax2.text(x_pos, i, f'{value:.3f}',
             va='center', ha=ha, fontsize=7, fontweight='bold')

# ============================================================================
# График 3: Распределение всех корреляций
# ============================================================================

ax3 = plt.subplot(3, 2, 3)

# Извлекаем верхний треугольник (без диагонали)
mask = np.triu(np.ones_like(correlation_matrix), k=1)
upper_triangle_values = correlation_matrix.values[mask.astype(bool)]

ax3.hist(upper_triangle_values, bins=100, color='steelblue',
         edgecolor='black', alpha=0.7, linewidth=0.5)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
ax3.axvline(x=0.8, color='orange', linestyle='--', linewidth=1.5,
            label='High positive (0.8)')
ax3.axvline(x=-0.8, color='orange', linestyle='--', linewidth=1.5,
            label='High negative (-0.8)')

ax3.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Distribution of All Pairwise Correlations',
              fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Статистика
mean_corr = np.mean(upper_triangle_values)
median_corr = np.median(upper_triangle_values)
ax3.text(0.02, 0.98,
         f'Mean: {mean_corr:.4f}\nMedian: {median_corr:.4f}\nTotal pairs: {len(upper_triangle_values):,}',
         transform=ax3.transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# График 4: Heatmap корреляций таргета со всеми признаками (упрощенная)
# ============================================================================

ax4 = plt.subplot(3, 2, 4)

# Создаем матрицу 1xN для всех признаков
target_corr_matrix = target_correlations.values.reshape(1, -1)

im = ax4.imshow(target_corr_matrix,
                cmap='RdBu_r',
                aspect='auto',
                vmin=-0.3,  # Ограничиваем для лучшей видимости
                vmax=0.3)

ax4.set_yticks([0])
ax4.set_yticklabels(['default'])
ax4.set_xticks([])
ax4.set_xlabel(f'All Features (n={len(target_correlations)})',
               fontsize=11, fontweight='bold')
ax4.set_title('Correlation Heatmap: All Features vs Default',
              fontsize=14, fontweight='bold', pad=15)

# Добавляем colorbar
cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1, shrink=0.6)
cbar.set_label('Correlation', fontsize=10)

# ============================================================================
# График 5: Scatter plot - самая сильная положительная корреляция
# ============================================================================

ax5 = plt.subplot(3, 2, 5)

if len(target_correlations) > 0:
    top_pos_feature = target_correlations.index[0]
    top_pos_corr = target_correlations.iloc[0]

    # Добавляем небольшой шум для лучшей визуализации
    x_data = df_train[top_pos_feature]
    y_data = df_train['default'] + np.random.normal(0, 0.02, len(df_train))

    # Рисуем scatter plot с прозрачностью
    ax5.scatter(x_data, y_data,
                alpha=0.1, s=5, color='darkred', edgecolors='none')

    # Добавляем линию тренда
    z = np.polyfit(x_data, df_train['default'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    ax5.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Trend line')

    ax5.set_xlabel(top_pos_feature, fontsize=10, fontweight='bold')
    ax5.set_ylabel('Default (with jitter)', fontsize=10, fontweight='bold')
    ax5.set_title(f'Strongest Positive Correlation\n{top_pos_feature} vs Default (r={top_pos_corr:.4f})',
                  fontsize=12, fontweight='bold', pad=15)
    ax5.set_ylim(-0.15, 1.15)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

# ============================================================================
# График 6: Scatter plot - самая сильная отрицательная корреляция
# ============================================================================

ax6 = plt.subplot(3, 2, 6)

if len(target_correlations) > 0:
    top_neg_feature = target_correlations.index[-1]
    top_neg_corr = target_correlations.iloc[-1]

    # Добавляем небольшой шум для лучшей визуализации
    x_data = df_train[top_neg_feature]
    y_data = df_train['default'] + np.random.normal(0, 0.02, len(df_train))

    # Рисуем scatter plot с прозрачностью
    ax6.scatter(x_data, y_data,
                alpha=0.1, s=5, color='darkblue', edgecolors='none')

    # Добавляем линию тренда
    z = np.polyfit(x_data, df_train['default'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    ax6.plot(x_line, p(x_line), "b-", linewidth=2, label=f'Trend line')

    ax6.set_xlabel(top_neg_feature, fontsize=10, fontweight='bold')
    ax6.set_ylabel('Default (with jitter)', fontsize=10, fontweight='bold')
    ax6.set_title(f'Strongest Negative Correlation\n{top_neg_feature} vs Default (r={top_neg_corr:.4f})',
                  fontsize=12, fontweight='bold', pad=15)
    ax6.set_ylim(-0.15, 1.15)
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

# ============================================================================
# Сохранение
# ============================================================================

plt.tight_layout()
plt.savefig('/home/dr/cbu/correlation_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Визуализация сохранена: correlation_analysis.png")

plt.close()

# ============================================================================
# ИТОГОВАЯ СТАТИСТИКА
# ============================================================================

print("\n" + "="*80)
print("📊 ИТОГОВАЯ СТАТИСТИКА КОРРЕЛЯЦИОННОГО АНАЛИЗА")
print("="*80)

print(f"\n🎯 Корреляции с целевой переменной (default):")
print(f"   • Максимальная положительная: {target_correlations.max():+.4f} ({target_correlations.idxmax()})")
print(f"   • Максимальная отрицательная: {target_correlations.min():+.4f} ({target_correlations.idxmin()})")
print(f"   • Средняя абсолютная:         {np.mean(np.abs(target_correlations)):.4f}")

print(f"\n🔗 Мультиколлинеарность:")
print(f"   • Пар с |r| > 0.9: {len(high_corr_df[high_corr_df['correlation'].abs() > 0.9])}")
print(f"   • Пар с |r| > 0.8: {len(high_corr_df)}")

print(f"\n📁 Созданные файлы:")
print(f"   1. correlation_matrix_full.csv                 - Полная матрица ({correlation_matrix.shape[0]}x{correlation_matrix.shape[1]})")
print(f"   2. target_correlations.csv                     - Корреляции с default (отсортировано)")
print(f"   3. high_correlations_multicollinearity.csv     - Пары с высокой корреляцией")
print(f"   4. correlation_analysis.png                    - Визуализации (6 графиков)")

print("\n" + "="*80)
print("✅ АНАЛИЗ ЗАВЕРШЕН!")
print("="*80)
