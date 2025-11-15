# Стратегия выбора модели для предсказания кредитных дефолтов

## 1. Анализ характеристик датасета

### 1.1 Основные параметры
- **Размер выборки**: 89,999 записей × 62 признака
- **Задача**: Бинарная классификация (предсказание дефолта)
- **Целевая переменная**: `default` (0 = нет дефолта, 1 = дефолт)
- **Основная метрика**: **AUC-ROC** (Area Under Curve)

### 1.2 Дисбаланс классов
- **Класс 0 (нет дефолта)**: 85,405 записей (94.90%)
- **Класс 1 (дефолт)**: 4,594 записи (5.10%)
- **Соотношение классов**: 1:18.6 (сильный дисбаланс)

**КРИТИЧНО**: Дисбаланс классов требует специальных методов обработки!

### 1.3 Типы признаков
- **Численные (float64)**: 24 признака
  - Финансовые коэффициенты: `debt_to_income_ratio`, `credit_utilization`, `payment_to_income_ratio`
  - Кредитная история: `credit_score`, `oldest_credit_line_age`, `total_credit_limit`
  - Экономические показатели региона: `regional_unemployment_rate`, `housing_price_index`

- **Целочисленные (int64)**: 24 признака
  - Временные: `application_hour`, `application_day_of_week`, `account_open_year`
  - Счетчики: `num_login_sessions`, `num_customer_service_calls`, `num_credit_accounts`
  - Параметры кредита: `loan_amount`, `loan_term`

- **Категориальные (object)**: 14 признаков
  - Демографические: `employment_type`, `education`, `marital_status`
  - Географические: `state`, `previous_zip_code`
  - Продуктовые: `loan_type`, `loan_purpose`, `origination_channel`

### 1.4 Качество данных
- Пропущенные значения: 0 (уже импутированы)
- Шумовые признаки: удалены
- Форматирование: стандартизировано

---

## 2. Рекомендации по выбору алгоритмов

### 2.1 BASELINE модель (обязательно начать с неё!)

**Логистическая регрессия с регуляризацией**
- **Библиотека**: `sklearn.linear_model.LogisticRegression`
- **Параметры**:
  - `class_weight='balanced'` - автоматическая корректировка весов классов
  - `penalty='l2'` или `'elasticnet'` - регуляризация для предотвращения переобучения
  - `solver='saga'` - для больших датасетов и elasticnet
  - `max_iter=1000` - достаточное количество итераций

**Обоснование**:
- Быстрое обучение (~5-10 секунд)
- Интерпретируемость (важно для кредитного скоринга)
- Хорошая baseline для сравнения
- Устойчивость к мультиколлинеарности при регуляризации
- Отлично работает с большим количеством признаков

**Ожидаемый результат**: AUC ~ 0.70-0.75

---

### 2.2 ОСНОВНЫЕ кандидаты (рекомендуется тестировать)

#### A. LightGBM (РЕКОМЕНДАЦИЯ #1)

**Почему именно LightGBM для этой задачи**:
1. **Оптимизация AUC**: LightGBM отлично работает с метрикой AUC
2. **Обработка дисбаланса**: Встроенные параметры `scale_pos_weight`, `is_unbalance`
3. **Скорость**: Очень быстрое обучение на 90k записях
4. **Категориальные признаки**: Нативная поддержка (не требует OHE)
5. **Автоматическая обработка пропусков**: Хотя у нас их нет, это преимущество

**Библиотека**: `lightgbm.LGBMClassifier`

**Ключевые гиперпараметры**:
```python
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,              # Начать с малого значения
    'learning_rate': 0.05,         # Умеренная скорость обучения
    'n_estimators': 500,           # Увеличить при необходимости
    'max_depth': -1,               # Нет ограничения (контролируется num_leaves)
    'min_child_samples': 20,       # Минимум объектов в листе
    'subsample': 0.8,              # Сэмплирование строк
    'colsample_bytree': 0.8,       # Сэмплирование признаков
    'reg_alpha': 0.1,              # L1 регуляризация
    'reg_lambda': 0.1,             # L2 регуляризация
    'scale_pos_weight': 18.6,      # Вес положительного класса (85405/4594)
    'random_state': 42,
    'n_jobs': -1
}
```

**Ожидаемый результат**: AUC ~ 0.78-0.82

---

#### B. CatBoost (РЕКОМЕНДАЦИЯ #2)

**Почему CatBoost**:
1. **Категориальные признаки**: Лучшая в классе обработка (14 категориальных признаков)
2. **Устойчивость к переобучению**: Ordered boosting
3. **Автоматическая обработка дисбаланса**: `auto_class_weights='Balanced'`
4. **Не требует масштабирования**: Работает с сырыми признаками
5. **Встроенная защита от переобучения**: Меньше нужно тюнить

**Библиотека**: `catboost.CatBoostClassifier`

**Ключевые гиперпараметры**:
```python
catboost_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,                    # Глубина деревьев
    'l2_leaf_reg': 3,              # L2 регуляризация
    'border_count': 128,           # Для численных признаков
    'auto_class_weights': 'Balanced',  # КРИТИЧНО для дисбаланса
    'random_seed': 42,
    'verbose': 100,
    'task_type': 'CPU',            # Или 'GPU' если доступно
    'thread_count': -1
}

# Указать категориальные признаки
cat_features = ['preferred_contact', 'referral_code', 'account_status_code',
                'employment_type', 'education', 'marital_status', 'loan_type',
                'loan_purpose', 'origination_channel', 'marketing_campaign', 'state']
```

**Ожидаемый результат**: AUC ~ 0.77-0.81

---

#### C. XGBoost (альтернатива)

**Почему XGBoost**:
1. **Производительность**: Высокое качество предсказаний
2. **Популярность**: Стандарт индустрии для табличных данных
3. **Регуляризация**: Встроенная L1/L2 регуляризация
4. **Гибкость**: Множество параметров тюнинга

**Библиотека**: `xgboost.XGBClassifier`

**Ключевые гиперпараметры**:
```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'scale_pos_weight': 18.6,      # Вес для дисбаланса
    'random_state': 42,
    'n_jobs': -1
}
```

**Ожидаемый результат**: AUC ~ 0.77-0.81

---

### 2.3 ДОПОЛНИТЕЛЬНЫЕ опции (при необходимости)

#### D. Random Forest
- **Когда использовать**: Если нужна простая интерпретируемость + устойчивость
- **Недостаток**: Медленнее gradient boosting, обычно ниже качество
- **Параметры**: `class_weight='balanced'`, `n_estimators=300`, `max_depth=15`

#### E. Neural Network (TabNet или MLP)
- **Когда использовать**: Если gradient boosting не дал желаемого результата
- **Требует**: Масштабирование признаков, больше времени на обучение
- **Не рекомендуется**: Для первой итерации (избыточная сложность)

---

## 3. Стратегия обработки дисбаланса классов

### 3.1 ОСНОВНОЙ подход (рекомендуется)
**Корректировка весов классов (Class Weights)**
- ✅ Простота реализации
- ✅ Не изменяет размер датасета
- ✅ Отлично работает с gradient boosting
- ✅ Поддержка всеми алгоритмами

**Реализация**:
```python
# Для scikit-learn
class_weight = 'balanced'  # или {0: 1, 1: 18.6}

# Для LightGBM
scale_pos_weight = 18.6

# Для CatBoost
auto_class_weights = 'Balanced'

# Для XGBoost
scale_pos_weight = 18.6
```

### 3.2 АЛЬТЕРНАТИВНЫЕ методы (при необходимости)

#### A. SMOTE (Synthetic Minority Over-sampling)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Увеличить до 30%
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Когда использовать**: Если class_weight не дает желаемого результата

**⚠️ ВАЖНО**:
- Применять ТОЛЬКО на train, НЕ на test
- Риск переобучения
- Увеличивает время обучения

#### B. Threshold Optimization
```python
from sklearn.metrics import precision_recall_curve

# После обучения модели с вероятностями
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
# Выбрать оптимальный порог для нужного баланса precision/recall
```

**Когда использовать**: После обучения модели для оптимизации бизнес-метрик

---

## 4. Стратегия валидации

### 4.1 Разделение данных

**Стратифицированное разделение** (ОБЯЗАТЕЛЬНО):
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% на тест (18,000 записей)
    stratify=y,              # КРИТИЧНО: сохраняем пропорции классов
    random_state=42
)
```

**Результат**:
- Train: 71,999 записей (80%)
  - Класс 0: ~68,324
  - Класс 1: ~3,675
- Test: 18,000 записей (20%)
  - Класс 0: ~17,081
  - Класс 1: ~919

### 4.2 Кросс-валидация

**Стратифицированная K-Fold** (5 фолдов):
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Для каждой модели
cv_scores = cross_val_score(model, X_train, y_train,
                            cv=cv, scoring='roc_auc', n_jobs=-1)
print(f'CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
```

**Почему 5 фолдов**:
- Баланс между вычислительными затратами и надежностью оценки
- Каждый фолд содержит ~14,400 записей train + ~3,600 validation
- Достаточно для стабильной оценки с дисбалансом классов

### 4.3 Early Stopping (для gradient boosting)

```python
# LightGBM
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='auc',
          early_stopping_rounds=50,
          verbose=100)

# CatBoost
model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          early_stopping_rounds=50,
          verbose=100)
```

**Преимущества**:
- Автоматическая защита от переобучения
- Оптимальное количество итераций
- Экономия времени обучения

---

## 5. План обучения моделей

### Этап 1: Baseline (ДЕНЬ 1)
**Цель**: Установить минимальную планку качества

1. **Обучить Logistic Regression с `class_weight='balanced'`**
2. **Оценить на CV**: ожидается AUC ~ 0.70-0.75
3. **Сохранить предсказания на test для сравнения**

**Критерий успеха**: AUC > 0.70

---

### Этап 2: Gradient Boosting Models (ДЕНЬ 1-2)
**Цель**: Достичь максимального качества

**Параллельное обучение трёх моделей**:
1. **LightGBM** с дефолтными параметрами + `scale_pos_weight`
2. **CatBoost** с дефолтными параметрами + `auto_class_weights`
3. **XGBoost** с дефолтными параметрами + `scale_pos_weight`

**Сравнение результатов**:
```python
results = {
    'LogisticRegression': {'CV': 0.72, 'Test': 0.71},
    'LightGBM': {'CV': 0.79, 'Test': 0.78},
    'CatBoost': {'CV': 0.78, 'Test': 0.77},
    'XGBoost': {'CV': 0.78, 'Test': 0.77}
}
```

**Выбор лучшей модели**: Наивысший Test AUC + минимальный gap между CV и Test

---

### Этап 3: Hyperparameter Tuning (ДЕНЬ 2-3)
**Цель**: Улучшить лучшую модель на 1-3% AUC

**Рекомендуемый подход**: Randomized Search + Bayesian Optimization

```python
from sklearn.model_selection import RandomizedSearchCV

# Для LightGBM
param_distributions = {
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [300, 500, 1000],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1]
}

random_search = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=param_distributions,
    n_iter=50,              # 50 комбинаций
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5),
    random_state=42,
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train, y_train)
print(f'Best params: {random_search.best_params_}')
print(f'Best CV AUC: {random_search.best_score_:.4f}')
```

**Приоритетные гиперпараметры для тюнинга**:
1. **Learning rate** (`learning_rate`) - самый важный
2. **Tree complexity** (`num_leaves`, `max_depth`) - контроль переобучения
3. **Regularization** (`reg_alpha`, `reg_lambda`) - снижение переобучения
4. **Sampling** (`subsample`, `colsample_bytree`) - разнообразие моделей

**Ожидаемое улучшение**: +0.01-0.03 к AUC

---

### Этап 4: Feature Engineering (опционально, ДЕНЬ 3-4)
**Если результаты не удовлетворяют**

**Идеи новых признаков**:
1. **Взаимодействия финансовых коэффициентов**:
   ```python
   df['debt_utilization_product'] = df['debt_to_income_ratio'] * df['credit_utilization']
   df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
   ```

2. **Агрегаты по категориям**:
   ```python
   df['avg_default_by_state'] = df.groupby('state')['default'].transform('mean')
   df['avg_income_by_education'] = df.groupby('education')['annual_income'].transform('mean')
   ```

3. **Временные признаки**:
   ```python
   df['is_weekend'] = df['application_day_of_week'].isin([5, 6]).astype(int)
   df['is_night'] = df['application_hour'].isin(range(22, 24)).astype(int)
   ```

4. **Кредитное здоровье (composite features)**:
   ```python
   df['credit_health_score'] = (
       df['credit_score'] / 850 * 0.4 +
       (1 - df['credit_utilization']) * 0.3 +
       (1 - df['debt_to_income_ratio']) * 0.3
   )
   ```

**⚠️ ВАЖНО**: Проверить на кросс-валидации, не ухудшилось ли качество!

---

### Этап 5: Ensemble Methods (опционально, ДЕНЬ 4-5)
**Если нужно выжать последние проценты**

#### A. Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_best),
        ('catboost', catboost_best),
        ('xgb', xgb_best)
    ],
    voting='soft',  # Использовать вероятности
    weights=[2, 1, 1]  # Больше вес лучшей модели
)
```

#### B. Stacking
```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('lgbm', lgbm_best),
        ('catboost', catboost_best),
        ('xgb', xgb_best)
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5
)
```

**Ожидаемое улучшение**: +0.005-0.015 к AUC (не гарантировано)

---

## 6. Метрики оценки

### 6.1 ОСНОВНАЯ метрика: AUC-ROC
```python
from sklearn.metrics import roc_auc_score, roc_curve

# Вычисление AUC
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f'Test AUC: {auc:.4f}')

# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Визуализация
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
```

### 6.2 ДОПОЛНИТЕЛЬНЫЕ метрики

#### Precision-Recall кривая (важна при дисбалансе)
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {ap_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
```

#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, classification_report

# Выбрать порог (например, 0.5 или оптимальный)
y_pred = (y_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
```

#### Метрики при разных порогах
```python
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]

for threshold in thresholds_to_test:
    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Threshold={threshold:.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}')
```

### 6.3 Важность признаков

```python
# Для LightGBM/XGBoost
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# Визуализация
plt.figure(figsize=(10, 12))
plt.barh(feature_importance['feature'][:20], feature_importance['importance'][:20])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
```

---

## 7. Обработка признаков

### 7.1 Категориальные признаки

**Для LightGBM/XGBoost**: One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder

categorical_features = ['preferred_contact', 'account_status_code',
                        'employment_type', 'education', 'marital_status',
                        'loan_type', 'loan_purpose', 'origination_channel',
                        'marketing_campaign', 'state']

# High cardinality features - можно использовать Label Encoding
high_cardinality = ['referral_code', 'previous_zip_code', 'loan_officer_id']

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_features])
```

**Для CatBoost**: Передать напрямую
```python
cat_features = ['preferred_contact', 'account_status_code',
                'employment_type', 'education', 'marital_status',
                'loan_type', 'loan_purpose', 'origination_channel',
                'marketing_campaign', 'state', 'referral_code']

model.fit(X_train, y_train, cat_features=cat_features)
```

### 7.2 Численные признаки

**Для gradient boosting**: НЕ требуется масштабирование
**Для логистической регрессии**: ОБЯЗАТЕЛЬНО масштабирование

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)
```

### 7.3 Выбросы

**Анализ выбросов**:
```python
# Проверить основные финансовые признаки
features_to_check = ['annual_income', 'loan_amount', 'debt_to_income_ratio',
                     'credit_utilization', 'credit_score']

for feature in features_to_check:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[feature] < Q1 - 3*IQR) | (df[feature] > Q3 + 3*IQR)).sum()
    print(f'{feature}: {outliers} outliers ({outliers/len(df)*100:.2f}%)')
```

**⚠️ РЕКОМЕНДАЦИЯ**: НЕ удалять выбросы на первой итерации
- Gradient boosting устойчив к выбросам
- Выбросы могут быть информативны для предсказания дефолта
- Если необходимо: winsorization (ограничение на уровне 1-99 перцентиля)

---

## 8. Риски и предупреждения

### 8.1 Переобучение (Overfitting)

**Индикаторы**:
- CV AUC значительно выше Test AUC (разница > 0.03)
- Высокая важность незначимых признаков
- Идеальное качество на train (AUC > 0.99)

**Решения**:
1. Увеличить регуляризацию (`reg_alpha`, `reg_lambda`)
2. Уменьшить сложность деревьев (`num_leaves`, `max_depth`)
3. Увеличить `min_child_samples`
4. Применить feature selection

### 8.2 Утечка данных (Data Leakage)

**Потенциальные источники**:
- `customer_ref`, `application_id` - удалить перед обучением (ID переменные)
- Признаки, рассчитанные на всём датасете - пересчитать на train

**Проверка**:
```python
# Удалить ID столбцы
X = df.drop(['customer_ref', 'application_id', 'default'], axis=1)
```

### 8.3 Недообучение (Underfitting)

**Индикаторы**:
- CV AUC и Test AUC оба низкие (< 0.75)
- Логистическая регрессия показывает схожие результаты с gradient boosting

**Решения**:
1. Увеличить сложность модели
2. Добавить feature engineering
3. Уменьшить регуляризацию

### 8.4 Вычислительные ограничения

**Для больших поисков гиперпараметров**:
- Использовать `RandomizedSearchCV` вместо `GridSearchCV`
- Ограничить `n_iter` в зависимости от времени
- Использовать `early_stopping` для gradient boosting
- Распараллелить на все ядра (`n_jobs=-1`)

---

## 9. Финальная рекомендация

### 9.1 Оптимальная последовательность

**ШАГ 1** (30 минут): Logistic Regression baseline
- Быстрая проверка качества данных
- Установка минимальной планки

**ШАГ 2** (1-2 часа): Три gradient boosting модели
- LightGBM, CatBoost, XGBoost с дефолтными параметрами
- Выбор лучшей по Test AUC

**ШАГ 3** (2-4 часа): Hyperparameter tuning лучшей модели
- RandomizedSearchCV с 50-100 итерациями
- Финальная модель

**ШАГ 4** (опционально): Feature engineering + Ensemble
- Только если предыдущие шаги не дали AUC > 0.80

### 9.2 Целевые метрики

**Минимальный результат**: AUC > 0.75 (приемлемо)
**Хороший результат**: AUC > 0.78 (хорошо)
**Отличный результат**: AUC > 0.80 (отлично)

### 9.3 Топ-1 рекомендация

**LightGBM с следующими параметрами**:
```python
best_model = LGBMClassifier(
    objective='binary',
    metric='auc',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=18.6,
    random_state=42,
    n_jobs=-1
)
```

**Обоснование выбора**:
1. Лучшее соотношение скорость/качество
2. Оптимален для метрики AUC
3. Встроенная обработка дисбаланса
4. Быстрое обучение даже с тюнингом
5. Высокая интерпретируемость (feature importance)

---

## 10. Чек-лист перед обучением

- [ ] Удалены ID столбцы (`customer_ref`, `application_id`)
- [ ] Целевая переменная `default` отделена от признаков
- [ ] Применено стратифицированное разделение train/test
- [ ] Установлен `random_state=42` везде для воспроизводимости
- [ ] Для gradient boosting указан `scale_pos_weight` или `auto_class_weights`
- [ ] Настроен `early_stopping` для validation set
- [ ] Метрика оценки - `roc_auc`
- [ ] Сохранены предсказания всех моделей для сравнения
- [ ] Визуализированы ROC и PR кривые
- [ ] Проанализирована важность признаков
- [ ] Проверен gap между CV и Test AUC (должен быть < 0.03)

---

## 11. Код для быстрого старта

См. файлы:
- `/home/dr/cbu/model_training_baseline.py` - Baseline модель
- `/home/dr/cbu/model_training_advanced.py` - Advanced модели
- `/home/dr/cbu/evaluation_framework.py` - Оценка и визуализация

---

**Дата создания**: 2025-11-15
**Версия**: 1.0
**Автор**: ML Strategy for Credit Default Prediction
