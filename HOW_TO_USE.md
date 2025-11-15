# КАК ИСПОЛЬЗОВАТЬ ПОДГОТОВЛЕННЫЕ ДАННЫЕ

## Быстрый старт

### 1. Загрузка финального датасета

Рекомендуется использовать **Parquet формат** (в 3 раза меньше и быстрее загружается):

```python
import pandas as pd

# Загрузка финального датасета
df = pd.read_parquet('/home/dr/cbu/final_dataset_clean.parquet')

print(f"Размер: {df.shape[0]} строк × {df.shape[1]} колонок")
print(f"\nЦелевая переменная:")
print(df['default'].value_counts())
```

Альтернативно, можно использовать CSV:

```python
df = pd.read_csv('/home/dr/cbu/final_dataset_clean.csv')
```

---

## 2. Структура датасета

### Целевая переменная:
- **default** - 0 (нет дефолта) / 1 (дефолт)

### Группы признаков:

#### A. Метаданные приложения (application_metadata)
- application_id, customer_ref
- application_hour, application_day_of_week
- account_open_year, account_status_code
- num_login_sessions, num_customer_service_calls
- has_mobile_app, paperless_billing
- preferred_contact, referral_code

#### B. Демография (demographics)
- age, marital_status, num_dependents, education
- employment_type, employment_length
- annual_income

#### C. Финансовые показатели (financial_ratios)
- monthly_income, existing_monthly_debt, monthly_payment
- debt_to_income_ratio, debt_service_ratio, payment_to_income_ratio
- revolving_balance, available_credit, credit_utilization
- total_debt_amount

#### D. Кредитная история (credit_history)
- num_open_accounts, num_closed_accounts
- total_credit_limit, oldest_account_age
- num_delinquencies_2yrs (832 пропуска)
- и др.

#### E. Детали кредита (loan_details)
- loan_amount, loan_term, loan_purpose
- interest_rate
- и др.

#### F. География (geographic_data)
- state, previous_zip_code
- regional_unemployment_rate, regional_median_income
- regional_median_rent, housing_price_index
- cost_of_living_index

---

## 3. Обработка перед обучением модели

### Шаг 1: Обработка пропущенных значений

```python
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Для num_delinquencies_2yrs - заполняем 0 (нет просрочек = нет данных)
df['num_delinquencies_2yrs'] = df['num_delinquencies_2yrs'].fillna(0)

# Для employment_length - используем медиану по группам
df['employment_length'] = df.groupby(['education', 'employment_type'])['employment_length'].transform(
    lambda x: x.fillna(x.median())
)

# Для revolving_balance - KNN imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df['revolving_balance'] = imputer.fit_transform(df[['revolving_balance']])
```

### Шаг 2: Кодирование категориальных переменных

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# One-Hot для низкой кардинальности
categorical_low = ['marital_status', 'education', 'employment_type',
                   'preferred_contact', 'loan_purpose']

df_encoded = pd.get_dummies(df, columns=categorical_low, drop_first=True)

# Target Encoding для state (20 категорий)
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['state'])
df_encoded['state'] = te.fit_transform(df['state'], df['default'])
```

### Шаг 3: Разбиение на train/test с стратификацией

```python
from sklearn.model_selection import train_test_split

# Отделяем признаки от целевой переменной
X = df_encoded.drop(['default', 'customer_ref', 'application_id'], axis=1)
y = df_encoded['default']

# Разбиение с стратификацией (важно для дисбаланса!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")
```

### Шаг 4: Обработка дисбаланса классов

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Комбинированный подход
over = SMOTE(sampling_strategy=0.3, random_state=42)  # Minority до 30% majority
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # Ratio 1:2

# Применяем
X_train_balanced, y_train_balanced = over.fit_resample(X_train, y_train)
X_train_balanced, y_train_balanced = under.fit_resample(X_train_balanced, y_train_balanced)

print(f"Balanced train: {y_train_balanced.value_counts()}")
```

---

## 4. Базовая модель (для бейзлайна)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Обучение
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

# Предсказания
y_pred_proba = lr.predict_proba(X_test)[:, 1]
y_pred = lr.predict(X_test)

# Оценка AUC (целевая метрика!)
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")

# Отчет
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## 5. Продвинутые модели

### XGBoost (рекомендуется для кредитного скоринга)

```python
import xgboost as xgb

# Обучение
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=18.6,  # Ratio негатив/позитив
    eval_metric='auc',
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

# Предсказания
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print(f"XGBoost AUC: {auc_xgb:.4f}")

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model, max_num_features=20)
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### LightGBM (быстрая альтернатива)

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=18.6,
    metric='auc',
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)
print(f"LightGBM AUC: {auc_lgb:.4f}")
```

---

## 6. Кросс-валидация

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 5-fold стратифицированная кросс-валидация
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    xgb_model,
    X_train, y_train,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)

print(f"Cross-validation AUC scores: {cv_scores}")
print(f"Mean AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 7. Оптимизация гиперпараметров

```python
from sklearn.model_selection import RandomizedSearchCV

# Пространство параметров
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Поиск
random_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=18.6, eval_metric='auc', random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"Best AUC: {random_search.best_score_:.4f}")
print(f"Best params: {random_search.best_params_}")

# Обучение с лучшими параметрами
best_model = random_search.best_estimator_
```

---

## 8. Анализ результатов

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_xgb)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300)
plt.show()
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Выбор порога
threshold = 0.5
y_pred_custom = (y_pred_proba_xgb >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_custom)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Метрики
tn, fp, fn, tp = cm.ravel()
print(f"Precision: {tp/(tp+fp):.4f}")
print(f"Recall: {tp/(tp+fn):.4f}")
print(f"Specificity: {tn/(tn+fp):.4f}")
```

---

## 9. Сохранение модели

```python
import pickle

# Сохранение
with open('best_credit_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Загрузка
with open('best_credit_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Проверка
y_pred_loaded = loaded_model.predict_proba(X_test)[:, 1]
print(f"Loaded model AUC: {roc_auc_score(y_test, y_pred_loaded):.4f}")
```

---

## 10. Полный пайплайн (пример)

```python
# Полный пример от загрузки до оценки
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# 1. Загрузка
df = pd.read_parquet('/home/dr/cbu/final_dataset_clean.parquet')

# 2. Обработка пропусков
df['num_delinquencies_2yrs'] = df['num_delinquencies_2yrs'].fillna(0)
df['employment_length'] = df.groupby(['education'])['employment_length'].transform(
    lambda x: x.fillna(x.median())
)

# 3. Кодирование
categorical = ['marital_status', 'education', 'employment_type']
df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)

# 4. Разбиение
X = df_encoded.drop(['default', 'customer_ref', 'application_id'], axis=1)
y = df_encoded['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Обучение
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=18.6,
    eval_metric='auc',
    random_state=42
)
model.fit(X_train, y_train)

# 6. Оценка
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"Final AUC Score: {auc:.4f}")
```

---

## Контакты и поддержка

Для вопросов по данным смотрите:
- `/home/dr/cbu/FINAL_ANALYSIS_REPORT.md` - полный анализ
- `/home/dr/cbu/data_cleaning.log` - лог очистки
- `/home/dr/cbu/data_quality_report.txt` - краткая статистика

**Удачи в обучении модели!**