# Детальный отчет: Трансформация признаков от 62 до 108 колонок

**Дата создания:** 2025-11-15 13:01:06  
**Проект:** Предсказание кредитных дефолтов  
**Цель:** Объяснить, как из 62 исходных колонок получилось 108 признаков для обучения модели

---

## Содержание

1. [Краткое резюме](#краткое-резюме)
2. [Исходные данные (62 колонки)](#исходные-данные-62-колонки)
3. [Трансформация 1: Feature Engineering (+12 признаков)](#трансформация-1-feature-engineering-12-признаков)
4. [Трансформация 2: One-Hot Encoding (+42 признака)](#трансформация-2-one-hot-encoding-42-признака)
5. [Трансформация 3: Frequency Encoding (+7 признаков)](#трансформация-3-frequency-encoding-7-признаков)
6. [Удаленные исходные колонки (-15)](#удаленные-исходные-колонки-15)
7. [Финальный набор (108 колонок)](#финальный-набор-108-колонок)
8. [Использование в модели (103 полезных)](#использование-в-модели-103-полезных)
9. [Заключение и рекомендации](#заключение-и-рекомендации)

---

## Краткое резюме

### Математика трансформации

```
Исходный датасет:           62 колонки
+ Feature Engineering:     +12 колонок (созданные комбинации)
+ One-Hot Encoding:        +42 колонки (бинарные признаки)
+ Frequency Encoding:       +7 колонок (числовые частоты)
- Удалены категориальные:  -15 колонок (исходные, которые были закодированы)
────────────────────────────────────────
= Финальный X_train:       108 колонок
```

### Использование в XGBoost модели

- **Всего признаков:** 108
- **Полезных (importance > 0):** 103 признаков
- **Бесполезных (importance = 0):** 5 признаков

---

## Исходные данные (62 колонки)

**Файл:** `final_dataset_clean.csv`  
**Размер:** 89,999 строк × 62 колонок  
**Пропущенных значений:** 0 (уже импутированы)

### Структура исходных данных

#### 1. Идентификаторы (3 колонки)
- `customer_ref` - int64
- `application_id` - int64
- `loan_officer_id` - int64


#### 2. Целевая переменная (1 колонка)
- `default` - int64 (0 = нет дефолта, 1 = дефолт)
  - Дефолтов: 4,594 (5.10%)

#### 3. Числовые признаки (44)

**Демографические:**
- `age` - мин: 18, макс: 74
- `annual_income` - мин: 20000, макс: 487200
- `employment_length` - мин: 0.0, макс: 23.9
- `num_dependents` - мин: 0, макс: 5

**Финансовые показатели:**
- `monthly_income` - среднее: 3965.57
- `existing_monthly_debt` - среднее: 894.14
- `monthly_payment` - среднее: 927.72
- `debt_to_income_ratio` - среднее: 0.23
- `debt_service_ratio` - среднее: 0.53
- `payment_to_income_ratio` - среднее: 0.31
- `credit_utilization` - среднее: 0.44
- `revolving_balance` - среднее: 39561.30
- `total_debt_amount` - среднее: 145535.12

**Кредитные метрики:**
- `credit_score` - среднее: 716.28
- `num_credit_accounts` - среднее: 9.40
- `oldest_credit_line_age` - среднее: 9.34
- `num_delinquencies_2yrs` - среднее: 0.02
- `num_inquiries_6mo` - среднее: 1.50

**Прочие:** application_hour, application_day_of_week, account_open_year, num_login_sessions, num_customer_service_calls, has_mobile_app, paperless_billing, available_credit, annual_debt_payment, loan_to_annual_income

#### 4. Категориальные признаки (14)

- `account_status_code` - 5 уникальных значений (топ: 'ACT-1')
- `credit_usage_amount` - 82354 уникальных значений (топ: '$36,582.00')
- `education` - 5 уникальных значений (топ: 'Bachelor')
- `employment_type` - 9 уникальных значений (топ: 'Full Time')
- `loan_purpose` - 8 уникальных значений (топ: 'Revolving Credit')
- `loan_type` - 12 уникальных значений (топ: 'Personal')
- `marital_status` - 3 уникальных значений (топ: 'Married')
- `marketing_campaign` - 26 уникальных значений (топ: 'Q')
- `monthly_free_cash_flow` - 88109 уникальных значений (топ: '$1,426.11')
- `origination_channel` - 4 уникальных значений (топ: 'Online')
- `preferred_contact` - 3 уникальных значений (топ: 'Email')
- `referral_code` - 7805 уникальных значений (топ: 'REF0000')
- `state` - 20 уникальных значений (топ: 'Ca')
- `total_monthly_debt_payment` - 85634 уникальных значений (топ: '653.99')


---

## Трансформация 1: Feature Engineering (+12 признаков)

**Цель:** Создать новые признаки на основе доменных знаний кредитного скоринга

### Созданные инженерные признаки

1. **`account_age_years`**
   - Возраст аккаунта в годах (из account_open_year)
   - Среднее: 8.51, Стд. откл.: 4.02

2. **`available_credit_after_loan`**
   - Доступный кредит - Сумма кредита - остаточная способность
   - Среднее: -54291.05, Стд. откл.: 154549.08

3. **`combined_risk_score`**
   - Комбинированный показатель риска на основе нескольких метрик
   - Среднее: 0.31, Стд. откл.: 0.09

4. **`credit_capacity`**
   - Общий кредитный лимит + Доступный кредит - кредитная емкость
   - Среднее: 91480.45, Стд. откл.: 67942.87

5. **`debt_service_coverage`**
   - Месячный доход / Общие ежемесячные выплаты - способность обслуживать долг
   - Среднее: 2.76, Стд. откл.: 1.83

6. **`income_per_dependent`**
   - Годовой доход / (Иждивенцы + 1) - доход на члена семьи
   - Среднее: 28006.15, Стд. откл.: 22267.72

7. **`income_vs_regional`**
   - Годовой доход / Региональная медиана - относительный уровень дохода
   - Среднее: 0.75, Стд. откл.: 0.44

8. **`logins_per_year`**
   - Количество входов / Возраст аккаунта - активность клиента
   - Среднее: 1.45, Стд. откл.: 1.30

9. **`payment_burden`**
   - Ежемесячный платеж / Месячный доход - доля дохода на кредит
   - Среднее: 0.31, Стд. откл.: 0.37

10. **`revolving_to_income`**
   - Возобновляемый баланс / Месячный доход - нагрузка по кредитным картам
   - Среднее: 0.83, Стд. откл.: 0.49

11. **`service_calls_per_year`**
   - Звонки в поддержку / Возраст аккаунта - проблемность клиента
   - Среднее: 0.33, Стд. откл.: 0.38

12. **`total_debt_to_income`**
   - Общий долг / Годовой доход - ключевой показатель долговой нагрузки
   - Среднее: 3.71, Стд. откл.: 4.65


### Обоснование Feature Engineering

1. **Финансовые коэффициенты** - стандартная практика в кредитном скоринге
   - DTI (Debt-to-Income), Payment Burden - ключевые показатели риска
   - Регуляторы часто требуют DTI < 43% для ипотеки

2. **Региональные сравнения** - учет стоимости жизни
   - $50K в Нью-Йорке ≠ $50K в Огайо
   - `income_vs_regional` нормализует доход по региону

3. **Поведенческие метрики** - индикаторы вовлеченности
   - Активные клиенты дефолтят реже
   - Частые звонки в поддержку = финансовые проблемы

4. **Кредитная емкость** - потенциал к закредитованности
   - Высокий доступный кредит + низкий доход = риск
   - `available_credit_after_loan` показывает запас прочности

---

## Трансформация 2: One-Hot Encoding (+42 признака)

**Цель:** Преобразовать категориальные признаки в числовой формат для ML моделей

**Метод:** One-Hot Encoding с `drop_first=True` (избежание мультиколлинеарности)

### Принцип One-Hot Encoding

```
Пример: employment_type = ['Full Time', 'Part Time', 'Self Employed']

Исходная колонка:          One-Hot кодирование (3 → 2 колонки):
customer_1: Full Time  →   employment_type_Part Time = 0,  employment_type_Self Employed = 0
customer_2: Part Time  →   employment_type_Part Time = 1,  employment_type_Self Employed = 0
customer_3: Self Employed → employment_type_Part Time = 0,  employment_type_Self Employed = 1
```

### Детальная разбивка по категориям


#### 1. `account_status_code` → 4 бинарных признаков

- `account_status_code_ACT-1`
  - Значение: 'ACT-1'
  - Встречается: 14,445 раз (20.1%)

- `account_status_code_ACT-2`
  - Значение: 'ACT-2'
  - Встречается: 14,486 раз (20.1%)

- `account_status_code_ACT-3`
  - Значение: 'ACT-3'
  - Встречается: 14,145 раз (19.6%)

- `account_status_code_ACTIVE`
  - Значение: 'ACTIVE'
  - Встречается: 14,485 раз (20.1%)


#### 2. `age_group` → 4 бинарных признаков

- `age_group_26-35`
  - Значение: '26-35'
  - Встречается: 20,858 раз (29.0%)

- `age_group_36-45`
  - Значение: '36-45'
  - Встречается: 23,521 раз (32.7%)

- `age_group_46-55`
  - Значение: '46-55'
  - Встречается: 14,187 раз (19.7%)

- `age_group_55+`
  - Значение: '55+'
  - Встречается: 5,000 раз (6.9%)


#### 3. `application_time_of_day` → 3 бинарных признаков

- `application_time_of_day_evening`
  - Значение: 'evening'
  - Встречается: 14,841 раз (20.6%)

- `application_time_of_day_morning`
  - Значение: 'morning'
  - Встречается: 18,138 раз (25.2%)

- `application_time_of_day_night`
  - Значение: 'night'
  - Встречается: 21,082 раз (29.3%)


#### 4. `credit_util_category` → 2 бинарных признаков

- `credit_util_category_low`
  - Значение: 'low'
  - Встречается: 17,246 раз (24.0%)

- `credit_util_category_medium`
  - Значение: 'medium'
  - Встречается: 40,457 раз (56.2%)


#### 5. `education` → 4 бинарных признаков

- `education_Bachelor`
  - Значение: 'Bachelor'
  - Встречается: 21,134 раз (29.4%)

- `education_Graduate`
  - Значение: 'Graduate'
  - Встречается: 12,511 раз (17.4%)

- `education_High School`
  - Значение: 'High School'
  - Встречается: 15,004 раз (20.8%)

- `education_Some College`
  - Значение: 'Some College'
  - Встречается: 17,126 раз (23.8%)


#### 6. `employment_stability` → 3 бинарных признаков

- `employment_stability_medium`
  - Значение: 'medium'
  - Встречается: 15,784 раз (21.9%)

- `employment_stability_new`
  - Значение: 'new'
  - Встречается: 5,011 раз (7.0%)

- `employment_stability_short`
  - Значение: 'short'
  - Встречается: 13,096 раз (18.2%)


#### 7. `employment_type` → 8 бинарных признаков

- `employment_type_Contractor`
  - Значение: 'Contractor'
  - Встречается: 1,276 раз (1.8%)

- `employment_type_Ft`
  - Значение: 'Ft'
  - Встречается: 9,949 раз (13.8%)

- `employment_type_Full Time`
  - Значение: 'Full Time'
  - Встречается: 30,414 раз (42.2%)

- `employment_type_Fulltime`
  - Значение: 'Fulltime'
  - Встречается: 10,037 раз (13.9%)

- `employment_type_Part Time`
  - Значение: 'Part Time'
  - Встречается: 5,419 раз (7.5%)

- `employment_type_Pt`
  - Значение: 'Pt'
  - Встречается: 1,792 раз (2.5%)

- `employment_type_Self Emp`
  - Значение: 'Self Emp'
  - Встречается: 2,710 раз (3.8%)

- `employment_type_Self Employed`
  - Значение: 'Self Employed'
  - Встречается: 8,001 раз (11.1%)


#### 8. `loan_purpose` → 7 бинарных признаков

- `loan_purpose_Home Improvement`
  - Значение: 'Home Improvement'
  - Встречается: 5,922 раз (8.2%)

- `loan_purpose_Home Purchase`
  - Значение: 'Home Purchase'
  - Встречается: 15,160 раз (21.1%)

- `loan_purpose_Major Purchase`
  - Значение: 'Major Purchase'
  - Встречается: 5,756 раз (8.0%)

- `loan_purpose_Medical`
  - Значение: 'Medical'
  - Встречается: 4,352 раз (6.0%)

- `loan_purpose_Other`
  - Значение: 'Other'
  - Встречается: 2,869 раз (4.0%)

- `loan_purpose_Refinance`
  - Значение: 'Refinance'
  - Встречается: 10,050 раз (14.0%)

- `loan_purpose_Revolving Credit`
  - Значение: 'Revolving Credit'
  - Встречается: 17,748 раз (24.6%)


#### 9. `marital_status` → 2 бинарных признаков

- `marital_status_Married`
  - Значение: 'Married'
  - Встречается: 41,471 раз (57.6%)

- `marital_status_Single`
  - Значение: 'Single'
  - Встречается: 23,874 раз (33.2%)


#### 10. `origination_channel` → 3 бинарных признаков

- `origination_channel_Broker`
  - Значение: 'Broker'
  - Встречается: 14,469 раз (20.1%)

- `origination_channel_Direct Mail`
  - Значение: 'Direct Mail'
  - Встречается: 3,666 раз (5.1%)

- `origination_channel_Online`
  - Значение: 'Online'
  - Встречается: 32,502 раз (45.1%)


#### 11. `preferred_contact` → 2 бинарных признаков

- `preferred_contact_Mail`
  - Значение: 'Mail'
  - Встречается: 7,120 раз (9.9%)

- `preferred_contact_Phone`
  - Значение: 'Phone'
  - Встречается: 21,695 раз (30.1%)



### Почему One-Hot Encoding?

**Преимущества:**
- ML модели требуют числовые входы
- Нет ложного порядка (Full Time ≠ Part Time + 1)
- Каждая категория - независимый признак

**Недостатки:**
- Увеличивает размерность (42 новых колонок)
- Не подходит для высококардинальных признаков

---

## Трансформация 3: Frequency Encoding (+7 признаков)

**Цель:** Закодировать высококардинальные категориальные признаки без взрыва размерности

**Метод:** Заменить категорию на частоту её встречаемости в данных

### Принцип Frequency Encoding

```
Пример: state = ['Ca', 'Tx', 'Ca', 'Ny', 'Ca', 'Tx']

Частоты:
  Ca: 3/6 = 0.50
  Tx: 2/6 = 0.33
  Ny: 1/6 = 0.17

Исходная колонка:    →    Frequency Encoding:
customer_1: Ca       →    state_freq = 0.50
customer_2: Tx       →    state_freq = 0.33
customer_3: Ca       →    state_freq = 0.50
customer_4: Ny       →    state_freq = 0.17
```

### Закодированные признаки

1. **`credit_usage_amount_freq`** (из `credit_usage_amount`)
   - Исходная колонка имела 82,354 уникальных значений
   - One-Hot создал бы 82,353 колонок!
   - Frequency Encoding: 1 числовая колонка
   - Диапазон частот: [0.0000, 0.0001]
   - Средняя частота: 0.0000

2. **`loan_type_freq`** (из `loan_type`)
   - Исходная колонка имела 12 уникальных значений
   - One-Hot создал бы 11 колонок!
   - Frequency Encoding: 1 числовая колонка
   - Диапазон частот: [0.0605, 0.1018]
   - Средняя частота: 0.0865

3. **`marketing_campaign_freq`** (из `marketing_campaign`)
   - Исходная колонка имела 26 уникальных значений
   - One-Hot создал бы 25 колонок!
   - Frequency Encoding: 1 числовая колонка
   - Диапазон частот: [0.0374, 0.0393]
   - Средняя частота: 0.0385

4. **`monthly_free_cash_flow_freq`** (из `monthly_free_cash_flow`)
   - Исходная колонка имела 88,109 уникальных значений
   - One-Hot создал бы 88,108 колонок!
   - Frequency Encoding: 1 числовая колонка
   - Диапазон частот: [0.0000, 0.0001]
   - Средняя частота: 0.0000

5. **`referral_code_freq`** (из `referral_code`)
   - Исходная колонка имела 7,805 уникальных значений
   - One-Hot создал бы 7,804 колонок!
   - Frequency Encoding: 1 числовая колонка
   - Диапазон частот: [0.0000, 0.8002]
   - Средняя частота: 0.6403

6. **`state_freq`** (из `state`)
   - Исходная колонка имела 20 уникальных значений
   - One-Hot создал бы 19 колонок!
   - Frequency Encoding: 1 числовая колонка
   - Диапазон частот: [0.0310, 0.1280]
   - Средняя частота: 0.0654

7. **`total_monthly_debt_payment_freq`** (из `total_monthly_debt_payment`)
   - Исходная колонка имела 85,634 уникальных значений
   - One-Hot создал бы 85,633 колонок!
   - Frequency Encoding: 1 числовая колонка
   - Диапазон частот: [0.0000, 0.0001]
   - Средняя частота: 0.0000


### Почему Frequency Encoding для этих признаков?

1. **`state`** (20 уникальных штатов)
   - One-Hot: 19 колонок
   - Frequency: 1 колонка
   - **Экономия: 18 колонок**

2. **`referral_code`** (7,805 уникальных значений)
   - One-Hot: 7,804 колонок (НЕВОЗМОЖНО!)
   - Frequency: 1 колонка
   - **Экономия: 7,804 колонок**

3. **Прочие высококардинальные** (loan_type, marketing_campaign, etc.)
   - Балансируют информативность и размерность

### Альтернативы (не использованы)

- **Target Encoding** - может вызвать утечку данных
- **Hash Encoding** - потеря интерпретируемости
- **Embeddings** - требуют нейронные сети

---

## Удаленные исходные колонки (-15)

**Причина удаления:** Эти категориальные признаки были закодированы в One-Hot или Frequency Encoding

### Список удаленных колонок

1. **`account_status_code`** (5 значений) → One-Hot Encoding (4 колонок)
2. **`credit_usage_amount`** (82354 значений) → One-Hot Encoding (1 колонок)
3. **`default`** - целевая переменная (хранится отдельно в y_train)
4. **`education`** (5 значений) → One-Hot Encoding (4 колонок)
5. **`employment_type`** (9 значений) → One-Hot Encoding (8 колонок)
6. **`loan_purpose`** (8 значений) → One-Hot Encoding (7 колонок)
7. **`loan_type`** (12 значений) → One-Hot Encoding (1 колонок)
8. **`marital_status`** (3 значений) → One-Hot Encoding (2 колонок)
9. **`marketing_campaign`** (26 значений) → One-Hot Encoding (1 колонок)
10. **`monthly_free_cash_flow`** (88109 значений) → One-Hot Encoding (1 колонок)
11. **`origination_channel`** (4 значений) → One-Hot Encoding (3 колонок)
12. **`preferred_contact`** (3 значений) → One-Hot Encoding (2 колонок)
13. **`referral_code`** (7805 значений) → One-Hot Encoding (1 колонок)
14. **`state`** (20 значений) → One-Hot Encoding (1 колонок)
15. **`total_monthly_debt_payment`** (85634 значений) → One-Hot Encoding (1 колонок)


---

## Финальный набор (108 колонок)

**Файл:** `X_train.parquet`  
**Размер:** 72,000 строк × 108 колонок

### Распределение признаков по типам

| Категория | Количество | Примеры |
|-----------|------------|---------|
| Исходные числовые (сохранены) | 47 | credit_score, age, annual_income |
| Инженерные признаки | 6 | income_vs_regional, payment_burden |
| One-Hot признаки | 42 | employment_type_Full Time |
| Frequency-encoded | 7 | state_freq, referral_code_freq |
| **ВСЕГО** | **108** | |

### Типы данных

- `int64`: 66 колонок
- `float64`: 42 колонок


### Проверка качества данных


- **Пропущенные значения:** 0 (0.00%)
- **Дубликаты:** 0
- **Константные колонки:** 0
- **Квази-константные (>99% одно значение):** 0

✅ Данные готовы к обучению модели

---

## Использование в модели (103 полезных)


**Модель:** XGBoost  
**Обучено на:** 108 признаках  
**Результат Test AUC:** 0.7900

### Анализ важности признаков

- **Полезные признаки (importance > 0):** 103 (95.4%)
- **Бесполезные признаки (importance = 0):** 5 (4.6%)

### Топ-20 самых важных признаков

| Ранг | Признак | Importance | Тип | Описание |
|------|---------|------------|-----|----------|
| 1 | `credit_score` | 0.0516 | Original | |
| 2 | `income_vs_regional` | 0.0411 | Engineered | |
| 3 | `monthly_income` | 0.0382 | Original | |
| 4 | `debt_service_coverage` | 0.0294 | Engineered | |
| 5 | `annual_income` | 0.0280 | Original | |
| 6 | `age` | 0.0266 | Original | |
| 7 | `payment_burden` | 0.0248 | Engineered | |
| 8 | `debt_service_ratio` | 0.0194 | Original | |
| 9 | `loan_purpose_Home Purchase` | 0.0169 | One-Hot | |
| 10 | `payment_to_income_ratio` | 0.0153 | Original | |
| 11 | `age_group_36-45` | 0.0143 | One-Hot | |
| 12 | `employment_stability_short` | 0.0133 | One-Hot | |
| 13 | `age_group_46-55` | 0.0128 | One-Hot | |
| 14 | `total_debt_to_income` | 0.0118 | Engineered | |
| 15 | `num_public_records` | 0.0110 | Original | |
| 16 | `num_delinquencies_2yrs` | 0.0108 | Original | |
| 17 | `employment_type_Part Time` | 0.0107 | One-Hot | |
| 18 | `num_credit_accounts` | 0.0106 | Original | |
| 19 | `education_Some College` | 0.0103 | One-Hot | |
| 20 | `marital_status_Single` | 0.0103 | One-Hot | |


### Распределение важности по типам признаков

| Тип признака | Количество | Сумма важности | Средняя важность |
|--------------|------------|----------------|------------------|
| Original | 47 | 0.5034 | 0.0107 |
| One-Hot | 42 | 0.2745 | 0.0065 |
| Engineered | 12 | 0.1690 | 0.0141 |
| Frequency | 7 | 0.0531 | 0.0076 |


### Бесполезные признаки (для удаления в итерации 2)

1. `preferred_contact_Mail` (One-Hot)
2. `credit_util_category_low` (One-Hot)
3. `loan_purpose_Revolving Credit` (One-Hot)
4. `employment_stability_new` (One-Hot)
5. `age_group_55+` (One-Hot)


**Рекомендация:** Удалить 5 бесполезных признаков → 103 признаков для итерации 2



---

## Заключение и рекомендации

### Итоговая трансформация

```
final_dataset_clean.csv (62 колонки)
           ↓
   Feature Engineering
           ↓
   +12 новых признаков (доменные знания)
           ↓
   One-Hot Encoding
           ↓
   +42 бинарных (категориальные → числовые)
           ↓
   Frequency Encoding
           ↓
   +7 числовых (высококардинальные)
           ↓
   Удаление исходных категориальных
           ↓
   -15 колонок (заменены кодированными)
           ↓
X_train.parquet (108 колонок)
           ↓
   XGBoost Feature Selection
           ↓
   103 полезных признака
```

### Ключевые достижения

✅ **Размерность под контролем**
- Избежали 7,000+ колонок (если бы One-Hot для всех категорий)
- Frequency Encoding сэкономил тысячи колонок

✅ **Доменные знания включены**
- 12 инженерных признаков из финансовой индустрии
- Учтены региональные различия

✅ **Нет утечки данных**
- Frequency encoding на train данных
- One-Hot без информации из test

✅ **Готовность к производству**
- Все трансформации воспроизводимы
- Метаданные сохранены

### Рекомендации для следующих итераций

#### Итерация 2: Оптимизация признаков

1. **Удалить 5 бесполезных признаков**
   - Уменьшит время обучения на ~5%
   - Снизит риск переобучения
   - Ожидаемый прирост AUC: +0.001-0.005

2. **Feature interaction** для топ-признаков
   - `credit_score × income_vs_regional`
   - `debt_service_coverage × employment_type`
   - Ожидаемый прирост AUC: +0.005-0.010

3. **Polynomial features** для финансовых метрик
   - `debt_to_income²`
   - `sqrt(credit_score)`
   - Ожидаемый прирост: +0.003-0.008


#### Итерация 3: Альтернативные методы

1. **Target Encoding с кросс-валидацией**
   - Для категорий с сильной связью с target
   - Может заменить One-Hot/Frequency для некоторых признаков

2. **Binning непрерывных переменных**
   - `credit_score` → категории (Poor, Fair, Good, Excellent)
   - Может выявить нелинейные зависимости

3. **PCA для коррелирующих признаков**
   - Если обнаружена мультиколлинеарность
   - Снижение размерности с сохранением информации

### Метрики успеха

| Метрика | Текущее | Цель итерации 2 | Цель итерации 3 |
|---------|---------|-----------------|-----------------|
| Test AUC | 0.7900 | 0.80-0.82 | 0.82-0.85 |
| Количество признаков | 108 | 100-105 | 80-100 |
| Время обучения | ~44 сек | <40 сек | <35 сек |
| Overfitting gap | 0.0746 | <0.05 | <0.03 |

---

## Приложения

### A. Формулы инженерных признаков

```python
# Финансовые коэффициенты
total_debt_to_income = total_debt_amount / annual_income
revolving_to_income = revolving_balance / monthly_income
payment_burden = monthly_payment / monthly_income
debt_service_coverage = monthly_income / total_monthly_debt_payment

# Региональное сравнение
income_vs_regional = annual_income / regional_median_income

# Семейная нагрузка
income_per_dependent = annual_income / (num_dependents + 1)

# Поведенческие
logins_per_year = num_login_sessions / account_age_years
service_calls_per_year = num_customer_service_calls / account_age_years

# Кредитная емкость
available_credit_after_loan = available_credit - loan_amount
credit_capacity = total_credit_limit + available_credit
```

### B. Код для воспроизведения

```python
import pandas as pd

# Загрузка обработанных данных
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')

# Удаление ID колонок (если нужно)
id_cols = ['customer_ref', 'application_id', 'loan_officer_id', 'previous_zip_code']
X_train_clean = X_train.drop(columns=[c for c in id_cols if c in X_train.columns])
X_test_clean = X_test.drop(columns=[c for c in id_cols if c in X_test.columns])

print(f"Признаков для обучения: {X_train_clean.shape[1]}")
```

### C. Файлы проекта

- `/home/dr/cbu/final_dataset_clean.csv` - исходные очищенные данные (62 колонки)
- `/home/dr/cbu/final_dataset_imputed.parquet` - импутированные данные
- `/home/dr/cbu/X_train.parquet` - обучающие признаки (108 колонок)
- `/home/dr/cbu/X_test.parquet` - тестовые признаки (108 колонок)
- `/home/dr/cbu/y_train.parquet` - обучающая целевая переменная
- `/home/dr/cbu/y_test.parquet` - тестовая целевая переменная
- `/home/dr/cbu/xgboost_feature_importance_v1.csv` - важность признаков
- `/home/dr/cbu/preprocessing_metadata.json` - метаданные трансформаций

---

**Дата создания отчета:** 2025-11-15 13:03:37  
**Автор:** ML Data Preparation Pipeline  
**Версия:** 1.0
