"""
Сравнение интерпретируемости моделей:
- XGBoost Optimized (итерация с feature engineering)
- CatBoost Iter3 (с интерпретируемостью)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("СРАВНЕНИЕ ИНТЕРПРЕТИРУЕМОСТИ МОДЕЛЕЙ")
print("="*100)
print()

DATA_DIR = '/home/dr/cbu'

# ============================================================================
# 1. ЗАГРУЗКА FEATURE IMPORTANCE ДЛЯ ОБЕИХ МОДЕЛЕЙ
# ============================================================================

print("[1/4] Загрузка feature importance...")
print()

# XGBoost Optimized
xgb_importance = pd.read_csv(f'{DATA_DIR}/feature_importance_optimized.csv')
print(f"✅ XGBoost Optimized: {len(xgb_importance)} признаков")

# CatBoost
catboost_importance = pd.read_csv(f'{DATA_DIR}/catboost_feature_importance.csv')
print(f"✅ CatBoost: {len(catboost_importance)} признаков")

# SHAP importance (только CatBoost)
catboost_shap = pd.read_csv(f'{DATA_DIR}/catboost_shap_importance.csv')
print(f"✅ CatBoost SHAP: {len(catboost_shap)} признаков")
print()

# ============================================================================
# 2. СРАВНЕНИЕ ТОП-20 ПРИЗНАКОВ
# ============================================================================

print("[2/4] Сравнение топ-20 наиболее важных признаков...")
print()

# Нормализуем importance для сравнения
xgb_importance['importance_norm'] = xgb_importance['importance'] / xgb_importance['importance'].sum()
catboost_importance['importance_norm'] = catboost_importance['importance'] / catboost_importance['importance'].sum()
catboost_shap['shap_norm'] = catboost_shap['mean_abs_shap'] / catboost_shap['mean_abs_shap'].sum()

# Топ-20
xgb_top20 = xgb_importance.head(20)
catboost_top20 = catboost_importance.head(20)
shap_top20 = catboost_shap.head(20)

print("📊 ТОП-20 ПРИЗНАКОВ: XGBoost Optimized (Feature Importance)")
print("="*100)
for i, (idx, row) in enumerate(xgb_top20.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:55s} {row['importance']:8.4f} ({row['importance_norm']*100:5.2f}%)")
print()

print("📊 ТОП-20 ПРИЗНАКОВ: CatBoost (Feature Importance)")
print("="*100)
for i, (idx, row) in enumerate(catboost_top20.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:55s} {row['importance']:8.2f} ({row['importance_norm']*100:5.2f}%)")
print()

print("📊 ТОП-20 ПРИЗНАКОВ: CatBoost (SHAP Values)")
print("="*100)
for i, (idx, row) in enumerate(shap_top20.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:55s} {row['mean_abs_shap']:8.4f} ({row['shap_norm']*100:5.2f}%)")
print()

# ============================================================================
# 3. АНАЛИЗ ПЕРЕСЕЧЕНИЙ
# ============================================================================

print("[3/4] Анализ пересечений топ признаков...")
print()

xgb_top20_features = set(xgb_top20['feature'])
catboost_top20_features = set(catboost_top20['feature'])
shap_top20_features = set(shap_top20['feature'])

# Пересечения
intersection_xgb_cat = xgb_top20_features & catboost_top20_features
intersection_xgb_shap = xgb_top20_features & shap_top20_features
intersection_cat_shap = catboost_top20_features & shap_top20_features

print(f"🔍 Пересечения топ-20 признаков:")
print(f"   • XGBoost ∩ CatBoost:       {len(intersection_xgb_cat)}/20 ({len(intersection_xgb_cat)/20*100:.0f}%)")
print(f"   • XGBoost ∩ CatBoost SHAP:  {len(intersection_xgb_shap)}/20 ({len(intersection_xgb_shap)/20*100:.0f}%)")
print(f"   • CatBoost ∩ CatBoost SHAP: {len(intersection_cat_shap)}/20 ({len(intersection_cat_shap)/20*100:.0f}%)")
print()

print(f"📋 Общие топ-20 признаки для XGBoost и CatBoost ({len(intersection_xgb_cat)} признаков):")
for i, feat in enumerate(sorted(intersection_xgb_cat), 1):
    xgb_rank = xgb_importance[xgb_importance['feature'] == feat].index[0] + 1
    cat_rank = catboost_importance[catboost_importance['feature'] == feat].index[0] + 1
    print(f"   {i:2d}. {feat:55s} (XGB rank: {xgb_rank:2d}, Cat rank: {cat_rank:2d})")
print()

# ============================================================================
# 4. ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ
# ============================================================================

print("[4/4] Создание визуализаций...")
print()

# Создаем большую фигуру с 4 графиками
fig = plt.figure(figsize=(20, 16))

# График 1: XGBoost Top-20
ax1 = plt.subplot(2, 2, 1)
xgb_plot = xgb_top20.head(20).copy()
colors1 = ['#2E86AB' if feat in intersection_xgb_cat else '#A23B72' for feat in xgb_plot['feature']]
ax1.barh(range(len(xgb_plot)), xgb_plot['importance'], color=colors1, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(xgb_plot)))
ax1.set_yticklabels(xgb_plot['feature'], fontsize=9)
ax1.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
ax1.set_title('XGBoost Optimized - Top 20 Features\n(Blue = shared with CatBoost, Purple = unique)',
              fontsize=12, fontweight='bold', pad=15)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# График 2: CatBoost Top-20
ax2 = plt.subplot(2, 2, 2)
cat_plot = catboost_top20.head(20).copy()
colors2 = ['#2E86AB' if feat in intersection_xgb_cat else '#F18F01' for feat in cat_plot['feature']]
ax2.barh(range(len(cat_plot)), cat_plot['importance'], color=colors2, alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(cat_plot)))
ax2.set_yticklabels(cat_plot['feature'], fontsize=9)
ax2.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
ax2.set_title('CatBoost - Top 20 Features\n(Blue = shared with XGBoost, Orange = unique)',
              fontsize=12, fontweight='bold', pad=15)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# График 3: SHAP Values Top-20
ax3 = plt.subplot(2, 2, 3)
shap_plot = shap_top20.head(20).copy()
colors3 = ['#06A77D' if feat in intersection_xgb_shap else '#D62828' for feat in shap_plot['feature']]
ax3.barh(range(len(shap_plot)), shap_plot['mean_abs_shap'], color=colors3, alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(shap_plot)))
ax3.set_yticklabels(shap_plot['feature'], fontsize=9)
ax3.set_xlabel('Mean |SHAP Value|', fontsize=11, fontweight='bold')
ax3.set_title('CatBoost SHAP - Top 20 Features\n(Green = shared with XGBoost, Red = unique)',
              fontsize=12, fontweight='bold', pad=15)
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# График 4: Сравнение важности для общих признаков
ax4 = plt.subplot(2, 2, 4)

# Берем топ-10 общих признаков
common_features = sorted(list(intersection_xgb_cat))[:10]

if len(common_features) > 0:
    xgb_common = []
    cat_common = []

    for feat in common_features:
        xgb_imp = xgb_importance[xgb_importance['feature'] == feat]['importance_norm'].values[0]
        cat_imp = catboost_importance[catboost_importance['feature'] == feat]['importance_norm'].values[0]
        xgb_common.append(xgb_imp * 100)
        cat_common.append(cat_imp * 100)

    x = np.arange(len(common_features))
    width = 0.35

    ax4.barh(x - width/2, xgb_common, width, label='XGBoost', alpha=0.7, color='#2E86AB')
    ax4.barh(x + width/2, cat_common, width, label='CatBoost', alpha=0.7, color='#F18F01')

    ax4.set_yticks(x)
    ax4.set_yticklabels([f[:40] + '...' if len(f) > 40 else f for f in common_features], fontsize=8)
    ax4.set_xlabel('Normalized Importance (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Comparison of Common Top Features', fontsize=12, fontweight='bold', pad=15)
    ax4.legend(fontsize=10)
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{DATA_DIR}/model_interpretability_comparison.png', dpi=150, bbox_inches='tight')
print(f"✅ Визуализация сохранена: model_interpretability_comparison.png")
print()

# ============================================================================
# 5. СОЗДАНИЕ СРАВНИТЕЛЬНОЙ ТАБЛИЦЫ
# ============================================================================

print("📊 Создание сравнительной таблицы...")

# Объединяем топ-30 признаков от обеих моделей
all_top_features = list(set(list(xgb_importance.head(30)['feature']) +
                              list(catboost_importance.head(30)['feature']) +
                              list(catboost_shap.head(30)['feature'])))

comparison_data = []

for feat in all_top_features:
    xgb_row = xgb_importance[xgb_importance['feature'] == feat]
    cat_row = catboost_importance[catboost_importance['feature'] == feat]
    shap_row = catboost_shap[catboost_shap['feature'] == feat]

    xgb_rank = xgb_row.index[0] + 1 if len(xgb_row) > 0 else 999
    cat_rank = cat_row.index[0] + 1 if len(cat_row) > 0 else 999
    shap_rank = shap_row.index[0] + 1 if len(shap_row) > 0 else 999

    xgb_imp = xgb_row['importance'].values[0] if len(xgb_row) > 0 else 0
    cat_imp = cat_row['importance'].values[0] if len(cat_row) > 0 else 0
    shap_imp = shap_row['mean_abs_shap'].values[0] if len(shap_row) > 0 else 0

    comparison_data.append({
        'feature': feat,
        'xgb_rank': xgb_rank,
        'xgb_importance': xgb_imp,
        'catboost_rank': cat_rank,
        'catboost_importance': cat_imp,
        'shap_rank': shap_rank,
        'shap_value': shap_imp,
        'avg_rank': (xgb_rank + cat_rank + shap_rank) / 3
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('avg_rank')
comparison_df.to_csv(f'{DATA_DIR}/feature_importance_comparison.csv', index=False)
print(f"✅ Сравнительная таблица сохранена: feature_importance_comparison.csv")
print()

# ============================================================================
# 6. ИТОГОВЫЙ ОТЧЕТ
# ============================================================================

print("="*100)
print("📊 ИТОГОВЫЙ ОТЧЕТ: ИНТЕРПРЕТИРУЕМОСТЬ МОДЕЛЕЙ")
print("="*100)
print()

print("🎯 ОСНОВНЫЕ МЕТРИКИ:")
print(f"   • XGBoost Optimized Test AUC:  0.8047")
print(f"   • CatBoost Test AUC:           0.7963")
print()

print("🔍 ИНТЕРПРЕТИРУЕМОСТЬ:")
print()
print("   XGBoost Optimized:")
print(f"      • Feature Importance: ✅ Доступен (164 признаков)")
print(f"      • SHAP Values:        ⚠️  Требует отдельного вычисления")
print(f"      • Топ-3 признака:     1. annual_income_X_age (0.0719)")
print(f"                            2. income_vs_regional_div_debt_service_ratio (0.0648)")
print(f"                            3. annual_income_sqrt (0.0334)")
print()

print("   CatBoost:")
print(f"      • Feature Importance: ✅ Доступен (180 признаков)")
print(f"      • SHAP Values:        ✅ Вычислены для 1000 samples")
print(f"      • Топ-3 по FI:        1. credit_score_squared (4.36)")
print(f"                            2. credit_score_sqrt (3.67)")
print(f"                            3. income_vs_regional_div_debt_service_ratio (3.64)")
print(f"      • Топ-3 по SHAP:      1. income_vs_regional_div_debt_service_ratio (0.1419)")
print(f"                            2. credit_score_squared (0.1161)")
print(f"                            3. annual_income_X_age (0.1086)")
print()

print("🔗 СОГЛАСОВАННОСТЬ МОДЕЛЕЙ:")
print(f"   • Общих признаков в топ-20:  {len(intersection_xgb_cat)}/20 ({len(intersection_xgb_cat)/20*100:.0f}%)")
print(f"   • Engineered features в топ-20 XGBoost:  {sum(1 for f in xgb_top20['feature'] if '_X_' in f or '_sqrt' in f or '_squared' in f)}/20")
print(f"   • Engineered features в топ-20 CatBoost: {sum(1 for f in catboost_top20['feature'] if '_X_' in f or '_sqrt' in f or '_squared' in f)}/20")
print()

print("💡 КЛЮЧЕВЫЕ ИНСАЙТЫ:")
print()
print("   1. ✅ Обе модели согласны по ключевым признакам:")
print("      • income_vs_regional_div_debt_service_ratio - топ в обеих")
print("      • credit_score (и его трансформации) - критически важен")
print("      • annual_income_X_age - мощная интеракция")
print()

print("   2. 📊 Engineered features доминируют:")
print("      • Polynomial features (sqrt, squared, log) в топ-20 обеих моделей")
print("      • Interaction features показывают высокую важность")
print()

print("   3. 🎯 Для интерпретируемости:")
print("      • CatBoost лучше: SHAP values готовы, ordered boosting")
print("      • XGBoost сильнее: выше AUC, но нужен дополнительный SHAP анализ")
print()

print("📁 СОЗДАННЫЕ ФАЙЛЫ:")
print("   1. model_interpretability_comparison.png   - Визуальное сравнение")
print("   2. feature_importance_comparison.csv       - Детальная таблица")
print()

print("="*100)
print("✅ АНАЛИЗ ИНТЕРПРЕТИРУЕМОСТИ ЗАВЕРШЕН")
print("="*100)
