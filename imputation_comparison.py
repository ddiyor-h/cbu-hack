"""
Сравнение различных стратегий импутации пропущенных значений
для оптимизации AUC в задаче предсказания дефолта
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("СРАВНЕНИЕ СТРАТЕГИЙ ИМПУТАЦИИ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("=" * 80)

# Load data
df = pd.read_csv('/home/dr/cbu/final_dataset_clean.csv')
print(f"\nИсходный датасет: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
print(f"Пропущенных значений: {df.isnull().sum().sum():,}")

# Columns with missing values
missing_cols = ['employment_length', 'revolving_balance', 'num_delinquencies_2yrs']
print(f"\nСтолбцы с пропусками: {', '.join(missing_cols)}")

# Prepare features and target
target_col = 'default'
exclude_cols = [target_col, 'customer_ref', 'application_id']

# Get feature columns (numeric only for baseline model)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print(f"\nЧисловых признаков для моделирования: {len(feature_cols)}")

# Store results
results = []

def evaluate_imputation(df_imputed, strategy_name, description):
    """
    Оценка качества импутации через кросс-валидацию модели
    """
    print(f"\n{'─' * 80}")
    print(f"Стратегия: {strategy_name}")
    print(f"Описание: {description}")
    print(f"{'─' * 80}")

    # Check for missing values
    missing_after = df_imputed[feature_cols].isnull().sum().sum()
    print(f"Пропусков после импутации: {missing_after}")

    if missing_after > 0:
        print(f"⚠ ВНИМАНИЕ: Остались пропуски!")
        return None

    # Prepare data
    X = df_imputed[feature_cols].copy()
    y = df_imputed[target_col].copy()

    # Check for inf values
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        print(f"⚠ ВНИМАНИЕ: Обнаружены inf значения: {inf_count}")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train simple logistic regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate AUC
    auc_train = roc_auc_score(y_train, y_pred_proba_train)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=5, scoring='roc_auc', n_jobs=-1
    )
    auc_cv_mean = cv_scores.mean()
    auc_cv_std = cv_scores.std()

    print(f"\nРезультаты:")
    print(f"  AUC Train: {auc_train:.4f}")
    print(f"  AUC Test:  {auc_test:.4f}")
    print(f"  AUC CV (5-fold): {auc_cv_mean:.4f} ± {auc_cv_std:.4f}")
    print(f"  Размер обучающей выборки: {len(X_train):,}")
    print(f"  Размер тестовой выборки: {len(X_test):,}")

    # Store results
    results.append({
        'Стратегия': strategy_name,
        'Описание': description,
        'AUC_Train': auc_train,
        'AUC_Test': auc_test,
        'AUC_CV_Mean': auc_cv_mean,
        'AUC_CV_Std': auc_cv_std,
        'Размер_выборки': len(df_imputed)
    })

    return {
        'auc_train': auc_train,
        'auc_test': auc_test,
        'auc_cv_mean': auc_cv_mean,
        'auc_cv_std': auc_cv_std
    }

# ============================================================================
# СТРАТЕГИЯ 1: УДАЛЕНИЕ СТРОК (Listwise Deletion)
# ============================================================================
df_deleted = df.dropna(subset=missing_cols).copy()
evaluate_imputation(
    df_deleted,
    "1. Удаление строк",
    "Удалены все строки с пропусками в любом из 3 столбцов"
)

# ============================================================================
# СТРАТЕГИЯ 2: ИМПУТАЦИЯ СРЕДНИМ (Mean Imputation)
# ============================================================================
df_mean = df.copy()
for col in missing_cols:
    mean_val = df_mean[col].mean()
    df_mean[col].fillna(mean_val, inplace=True)
    print(f"  {col}: заполнено средним = {mean_val:.2f}")

evaluate_imputation(
    df_mean,
    "2. Среднее значение",
    "Пропуски заполнены средним арифметическим по каждому столбцу"
)

# ============================================================================
# СТРАТЕГИЯ 3: ИМПУТАЦИЯ МЕДИАНОЙ (Median Imputation)
# ============================================================================
df_median = df.copy()
for col in missing_cols:
    median_val = df_median[col].median()
    df_median[col].fillna(median_val, inplace=True)
    print(f"  {col}: заполнено медианой = {median_val:.2f}")

evaluate_imputation(
    df_median,
    "3. Медиана",
    "Пропуски заполнены медианой по каждому столбцу (устойчива к выбросам)"
)

# ============================================================================
# СТРАТЕГИЯ 4: ИМПУТАЦИЯ НУЛЯМИ (Zero Imputation)
# ============================================================================
df_zero = df.copy()
for col in missing_cols:
    df_zero[col].fillna(0, inplace=True)

evaluate_imputation(
    df_zero,
    "4. Нулевые значения",
    "Пропуски заполнены нулями (допущение: отсутствие = 0)"
)

# ============================================================================
# СТРАТЕГИЯ 5: ИНДИКАТОРНЫЙ МЕТОД + МЕДИАНА
# ============================================================================
df_indicator = df.copy()
for col in missing_cols:
    # Create missing indicator
    df_indicator[f'{col}_was_missing'] = df_indicator[col].isnull().astype(int)
    # Fill with median
    median_val = df_indicator[col].median()
    df_indicator[col].fillna(median_val, inplace=True)

# Update feature columns to include indicators
feature_cols_indicator = feature_cols + [f'{col}_was_missing' for col in missing_cols]
numeric_cols_indicator = df_indicator.select_dtypes(include=[np.number]).columns.tolist()
feature_cols_indicator = [c for c in numeric_cols_indicator if c not in exclude_cols]

# Evaluate with indicators
print(f"\n{'─' * 80}")
print(f"Стратегия: 5. Индикатор + Медиана")
print(f"Описание: Создан индикатор пропуска + заполнение медианой")
print(f"{'─' * 80}")

missing_after = df_indicator[feature_cols_indicator].isnull().sum().sum()
print(f"Пропусков после импутации: {missing_after}")

X = df_indicator[feature_cols_indicator].copy()
y = df_indicator[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

auc_train = roc_auc_score(y_train, y_pred_proba_train)
auc_test = roc_auc_score(y_test, y_pred_proba_test)

cv_scores = cross_val_score(
    model, X_train_scaled, y_train,
    cv=5, scoring='roc_auc', n_jobs=-1
)
auc_cv_mean = cv_scores.mean()
auc_cv_std = cv_scores.std()

print(f"\nРезультаты:")
print(f"  AUC Train: {auc_train:.4f}")
print(f"  AUC Test:  {auc_test:.4f}")
print(f"  AUC CV (5-fold): {auc_cv_mean:.4f} ± {auc_cv_std:.4f}")
print(f"  Размер обучающей выборки: {len(X_train):,}")
print(f"  Размер тестовой выборки: {len(X_test):,}")

results.append({
    'Стратегия': "5. Индикатор + Медиана",
    'Описание': "Создан индикатор пропуска + заполнение медианой",
    'AUC_Train': auc_train,
    'AUC_Test': auc_test,
    'AUC_CV_Mean': auc_cv_mean,
    'AUC_CV_Std': auc_cv_std,
    'Размер_выборки': len(df_indicator)
})

# ============================================================================
# СТРАТЕГИЯ 6: KNN IMPUTATION
# ============================================================================
print(f"\n{'─' * 80}")
print(f"Стратегия: 6. KNN Импутация")
print(f"Описание: Заполнение на основе k ближайших соседей (k=5)")
print(f"{'─' * 80}")

# Prepare subset for KNN (only numeric, will be slower)
df_knn = df.copy()

# Select a subset of most important numeric features for KNN
# (using all features would be very slow)
important_features = [
    'employment_length', 'revolving_balance', 'num_delinquencies_2yrs',
    'credit_score', 'annual_income', 'monthly_income', 'age',
    'debt_to_income_ratio', 'credit_utilization', 'total_debt_amount',
    'num_credit_accounts', 'oldest_credit_line_age'
]

# KNN imputer
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df_knn[important_features] = knn_imputer.fit_transform(df_knn[important_features])

print(f"Пропусков после KNN импутации: {df_knn[missing_cols].isnull().sum().sum()}")

evaluate_imputation(
    df_knn,
    "6. KNN Импутация",
    "Заполнение на основе k=5 ближайших соседей по важным признакам"
)

# ============================================================================
# СТРАТЕГИЯ 7: DOMAIN-SPECIFIC IMPUTATION
# ============================================================================
df_domain = df.copy()

# employment_length: заполнить медианой (консервативный подход)
df_domain['employment_length'].fillna(df_domain['employment_length'].median(), inplace=True)

# revolving_balance: заполнить медианой по группам credit_score
# (люди с похожим кредитным рейтингом имеют похожие балансы)
df_domain['revolving_balance'] = df_domain.groupby('credit_score')['revolving_balance'].transform(
    lambda x: x.fillna(x.median())
)
# Если еще остались пропуски (редкие credit_score), заполнить общей медианой
df_domain['revolving_balance'].fillna(df_domain['revolving_balance'].median(), inplace=True)

# num_delinquencies_2yrs: заполнить 0 (отсутствие данных = нет просрочек)
# Это логично, т.к. 98% значений = 0
df_domain['num_delinquencies_2yrs'].fillna(0, inplace=True)

evaluate_imputation(
    df_domain,
    "7. Доменная импутация",
    "employment_length=медиана, revolving_balance=медиана по группам credit_score, num_delinquencies=0"
)

# ============================================================================
# СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC_CV_Mean', ascending=False)

print("\n" + results_df.to_string(index=False))

# Find best strategy
best_idx = results_df['AUC_CV_Mean'].idxmax()
best_strategy = results_df.loc[best_idx]

print("\n" + "=" * 80)
print("ЛУЧШАЯ СТРАТЕГИЯ")
print("=" * 80)
print(f"\nНазвание: {best_strategy['Стратегия']}")
print(f"Описание: {best_strategy['Описание']}")
print(f"AUC CV: {best_strategy['AUC_CV_Mean']:.4f} ± {best_strategy['AUC_CV_Std']:.4f}")
print(f"AUC Test: {best_strategy['AUC_Test']:.4f}")
print(f"Размер выборки: {best_strategy['Размер_выборки']:,}")

# Calculate improvement over baseline (deletion)
baseline = results_df[results_df['Стратегия'].str.contains('Удаление')].iloc[0]
improvement = (best_strategy['AUC_CV_Mean'] - baseline['AUC_CV_Mean']) * 100

print(f"\nУлучшение относительно удаления строк: {improvement:.2f}%")
print(f"Сохранено записей: {best_strategy['Размер_выборки'] - baseline['Размер_выборки']:,}")

# Save results
results_df.to_csv('/home/dr/cbu/imputation_comparison_results.csv', index=False, encoding='utf-8-sig')
print(f"\nРезультаты сохранены: /home/dr/cbu/imputation_comparison_results.csv")

print("\n" + "=" * 80)
