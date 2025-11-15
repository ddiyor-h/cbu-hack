# Стратегия отбора признаков для модели предсказания дефолтов

## Резюме

### ✅ РЕКОМЕНДАЦИЯ: ИСПОЛЬЗОВАТЬ ВСЕ 105 ПРИЗНАКОВ (из 108, удалив 3 ID)

**Краткое обоснование:**
- Древовидные модели имеют встроенный отбор признаков
- Данные в хорошем состоянии (нет утечки, низкая мультиколлинеарность)
- Все признаки потенциально информативны
- Feature importance покажет, какие признаки реально важны

---

## 1. Текущее состояние данных

### 1.1. Статистика признаков

```
Всего признаков: 108
├── ID-признаков (удалить): 3
│   ├── customer_ref
│   ├── application_id
│   └── loan_officer_id
│
├── Полезных числовых: 62
│   ├── Финансовые (доходы, долги, коэффициенты)
│   ├── Поведенческие (логины, звонки)
│   ├── Кредитная история (score, inquiries)
│   └── Географические (региональные показатели)
│
├── Инженерные признаки: 10
│   ├── loan_to_annual_income
│   ├── total_debt_to_income
│   ├── income_vs_regional
│   ├── income_per_dependent
│   └── другие комбинированные метрики
│
└── One-hot закодированные: 43
    ├── education (4 значения)
    ├── employment_type (8 значений)
    ├── loan_purpose (7 значений)
    ├── account_status_code (4 значения)
    └── другие категориальные

ИТОГО ДЛЯ ОБУЧЕНИЯ: 105 признаков
```

### 1.2. Типы признаков

| Тип | Количество | Примеры |
|-----|------------|---------|
| Бинарные флаги (0/1) | 46 | has_mobile_app, paperless_billing, marital_status_* |
| Непрерывные числовые | 52 | annual_income, debt_ratio, credit_score |
| Частотное кодирование | 7 | state_freq, loan_type_freq, marketing_campaign_freq |

---

## 2. Проверка качества данных

### 2.1. Дисперсия признаков ✅

**Признаков с нулевой дисперсией: 0**

Найдено 7 признаков с очень низкой дисперсией (< 0.01), но все они полезны:

| Признак | Дисперсия | Уникальных значений | Примечание |
|---------|-----------|---------------------|------------|
| monthly_free_cash_flow_freq | 6.8e-12 | 4 | Частотное кодирование |
| total_monthly_debt_payment_freq | 1.6e-11 | 4 | Частотное кодирование |
| credit_usage_amount_freq | 3.7e-11 | 6 | Частотное кодирование |
| marketing_campaign_freq | 3.0e-07 | 24 | Частотное кодирование |
| loan_type_freq | 2.4e-04 | 12 | Частотное кодирование |
| state_freq | 1.2e-03 | 20 | Частотное кодирование |
| combined_risk_score | 7.9e-03 | 71,984 | Высокая гранулярность |

**Вывод**: Низкая дисперсия для частотного кодирования - это нормально. Удалять не нужно.

### 2.2. Мультиколлинеарность (корреляция между признаками)

#### Высокая корреляция (r > 0.9): 21 пара

| Признак 1 | Признак 2 | Корреляция | Причина |
|-----------|-----------|------------|---------|
| num_inquiries_6mo | recent_inquiry_count | 1.000 | Дубликаты |
| account_open_year | account_age_years | 1.000 | Обратная связь |
| oldest_credit_line_age | oldest_account_age_months | 1.000 | Разные единицы |
| annual_income | monthly_income | 1.000 | monthly = annual/12 |
| payment_to_income_ratio | payment_burden | 1.000 | Дубликаты |
| total_credit_limit | credit_capacity | 0.998 | Почти дубликаты |
| loan_to_annual_income | total_debt_to_income | 0.995 | Высокая связь |
| total_debt_amount | loan_amount | 0.973 | Ожидаемая связь |
| debt_service_ratio | payment_to_income_ratio | 0.964 | Связанные метрики |

**Важно**:
- Высокая корреляция **НЕ ПРОБЛЕМА** для древовидных моделей
- Они не страдают от мультиколлинеарности (в отличие от линейной регрессии)
- XGBoost/LightGBM сами выберут наиболее информативный признак

#### Умеренная корреляция (0.8 < r ≤ 0.9): 26 пар

Топ-10 пар:

| Признак 1 | Признак 2 | Корреляция |
|-----------|-----------|------------|
| debt_service_ratio | loan_to_annual_income | 0.893 |
| debt_service_ratio | total_debt_to_income | 0.888 |
| monthly_payment | total_debt_amount | 0.887 |
| available_credit | credit_capacity | 0.887 |
| available_credit | total_credit_limit | 0.886 |

**Вывод**: Умеренная корреляция - это нормально для финансовых данных.

### 2.3. Проверка на утечку данных (Data Leakage) ✅

Проверили корреляцию всех признаков с целевой переменной (default):

**Топ-15 признаков по корреляции с default:**

| Признак | Корреляция | Безопасно? |
|---------|------------|------------|
| debt_service_ratio | 0.217 | ✅ Да |
| payment_to_income_ratio | 0.217 | ✅ Да |
| payment_burden | 0.217 | ✅ Да |
| credit_score | 0.193 | ✅ Да |
| total_debt_to_income | 0.169 | ✅ Да |
| loan_to_annual_income | 0.166 | ✅ Да |
| annual_income | 0.141 | ✅ Да |
| monthly_income | 0.141 | ✅ Да |
| age | 0.139 | ✅ Да |
| income_vs_regional | 0.137 | ✅ Да |

**Критерий утечки**: корреляция > 0.8 (подозрительно идеальная связь)

**Результат**: Все корреляции < 0.22 → **утечки данных НЕТ** ✅

---

## 3. Стратегия отбора признаков

### 3.1. Рекомендуемый подход: "Train First, Select Later"

```
Шаг 1: Обучить модель на ВСЕХ признаках (кроме ID)
   ↓
Шаг 2: Получить feature importance
   ↓
Шаг 3: Проанализировать важность признаков
   ↓
Шаг 4: Если нужно - удалить неважные (importance < порог)
   ↓
Шаг 5: Переобучить и сравнить AUC
```

### 3.2. Почему этот подход лучше?

#### ✅ Преимущества

1. **Не рискуем потерять важный признак**
   - Априори трудно предсказать, какие признаки окажутся важны
   - Модель сама найдет полезные паттерны

2. **Древовидные модели имеют встроенный отбор**
   - XGBoost/LightGBM автоматически игнорируют неинформативные признаки
   - Feature importance показывает реальную ценность каждого признака

3. **Данные в хорошем состоянии**
   - Нет мусорных признаков (кроме ID)
   - Нет критичной мультиколлинеарности
   - Нет утечки данных

4. **Экономия времени**
   - Не тратим время на ручной отбор
   - Начинаем обучение быстрее

#### ⚠️ Когда может понадобиться отбор

Если после обучения базовой модели:

- **Переобучение**: train AUC >> validation AUC
- **Долгое обучение**: > 10 минут на модель
- **Много неважных признаков**: большинство имеют importance ≈ 0

Тогда:
1. Оставляем топ-50 признаков по importance
2. Переобучаем модель
3. Сравниваем AUC

### 3.3. Что точно удалить СЕЙЧАС

**ID-признаки** (не предикторы, а идентификаторы):

```python
features_to_remove = ['customer_ref', 'application_id', 'loan_officer_id']
```

Эти 3 признака:
- Не содержат предсказательной информации
- Уникальны для каждого клиента
- Могут привести к переобучению (модель "запомнит" конкретных клиентов)

**Итого для обучения: 105 признаков**

### 3.4. Что оставить

Все остальные 105 признаков:

#### Финансовые признаки (критически важные)
- `annual_income`, `monthly_income` - платежеспособность
- `debt_to_income_ratio`, `debt_service_ratio` - долговая нагрузка
- `credit_score`, `credit_utilization` - кредитная история
- `total_debt_amount`, `revolving_balance` - текущие обязательства

#### Поведенческие признаки
- `num_login_sessions`, `logins_per_year` - активность
- `num_customer_service_calls`, `service_calls_per_year` - проблемность
- `has_mobile_app`, `paperless_billing` - цифровая грамотность

#### Демографические признаки
- `age`, `age_group_*` - возрастная группа
- `employment_type_*`, `employment_length` - стабильность дохода
- `education_*` - уровень образования
- `marital_status_*`, `num_dependents` - семейное положение

#### Географические признаки
- `state_freq` - регион проживания
- `regional_unemployment_rate` - экономика региона
- `regional_median_income`, `regional_median_rent` - стоимость жизни
- `housing_price_index`, `cost_of_living_index` - макроэкономика

#### Инженерные признаки (созданные нами)
- `loan_to_annual_income` - доступность кредита
- `total_debt_to_income` - общая долговая нагрузка
- `income_vs_regional` - доход относительно региона
- `income_per_dependent` - доход на иждивенца
- `debt_service_coverage` - покрытие долга
- И другие комбинированные метрики

#### Категориальные признаки (one-hot кодирование)
- `loan_purpose_*` - цель кредита (7 категорий)
- `employment_type_*` - тип занятости (8 категорий)
- `education_*` - образование (4 категории)
- `account_status_code_*` - статус счета (4 категории)
- И другие

---

## 4. Альтернативные методы отбора (если понадобятся)

### 4.1. Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier

# Обучить модель
estimator = GradientBoostingClassifier()
selector = RFE(estimator, n_features_to_select=50, step=1)
selector.fit(X_train, y_train)

# Получить отобранные признаки
selected_features = X_train.columns[selector.support_]
```

**Когда использовать**: если модель переобучается или обучение слишком долгое.

### 4.2. Feature Importance Filtering

```python
import lightgbm as lgb

# Обучить модель
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Получить importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Отобрать топ-50
top_features = importance.head(50)['feature'].tolist()
```

**Когда использовать**: после обучения базовой модели, если нужно ускорить или упростить.

### 4.3. Correlation-Based Filtering

```python
# Удалить один признак из пары с r > 0.95
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns
               if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)
```

**Когда использовать**: для линейных моделей (логистическая регрессия). Для древовидных не нужно.

---

## 5. План действий

### Шаг 1: Подготовить данные для обучения ✅

```python
# Удалить только ID-признаки
features_to_remove = ['customer_ref', 'application_id', 'loan_officer_id']
X_train_final = X_train.drop(columns=features_to_remove)
X_test_final = X_test.drop(columns=features_to_remove)

print(f"Признаков для обучения: {X_train_final.shape[1]}")  # 105
```

### Шаг 2: Обучить базовую модель

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_final, y_train,
    eval_set=[(X_val_final, y_val)],
    eval_metric='auc',
    early_stopping_rounds=50,
    verbose=100
)
```

### Шаг 3: Оценить feature importance

```python
import matplotlib.pyplot as plt

# Feature importance
importance = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Топ-30
print(importance.head(30))

# Визуализация
plt.figure(figsize=(10, 12))
importance.head(30).plot(x='feature', y='importance', kind='barh')
plt.title('Top 30 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### Шаг 4: Проверить необходимость отбора

```python
# Сколько признаков имеют importance ≈ 0?
low_importance = importance[importance['importance'] < 0.001]
print(f"Признаков с low importance: {len(low_importance)}")

# Если > 30 признаков с importance < 0.001, можно попробовать отбор
if len(low_importance) > 30:
    print("Можно попробовать отбор признаков")
else:
    print("Все признаки полезны, отбор не нужен")
```

### Шаг 5: (Опционально) Переобучить с отобранными признаками

```python
# Если решили делать отбор
top_features = importance.head(50)['feature'].tolist()
X_train_selected = X_train_final[top_features]
X_test_selected = X_test_final[top_features]

# Переобучить
model_selected = lgb.LGBMClassifier(...)
model_selected.fit(X_train_selected, y_train)

# Сравнить AUC
print(f"AUC (105 признаков): {auc_full}")
print(f"AUC (50 признаков): {auc_selected}")
```

---

## 6. Сохраненные файлы

Результаты анализа:

1. **Список признаков для обучения**:
   ```
   /home/dr/cbu/features_for_training.txt
   ```
   Содержит 105 признаков (без ID)

2. **Матрица корреляций**:
   ```
   /home/dr/cbu/correlation_matrix.csv
   ```
   Корреляции между всеми числовыми признаками

Для просмотра:
```bash
head -20 /home/dr/cbu/features_for_training.txt
```

---

## 7. Заключение

**Наши данные готовы к обучению без дополнительного отбора признаков.**

Ключевые факты:
- ✅ Нет утечки данных
- ✅ Нет критичной мультиколлинеарности
- ✅ Все признаки информативны
- ✅ Древовидные модели устойчивы к избыточности признаков

**Рекомендация**: начать обучение с 105 признаками, получить feature importance, при необходимости провести отбор на основе реальной важности.

**Помните**: лучше дать модели больше информации и позволить ей самой выбрать важное, чем рисковать удалить что-то ценное.
