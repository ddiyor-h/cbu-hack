# Credit Default Prediction - Training Data Ready

## Quick Start

Данные подготовлены и готовы к обучению моделей машинного обучения.

### Загрузка данных

```python
import pandas as pd

# Для древовидных моделей (Random Forest, XGBoost, LightGBM, CatBoost)
X_train = pd.read_parquet('/home/dr/cbu/X_train.parquet')
X_test = pd.read_parquet('/home/dr/cbu/X_test.parquet')

# Для линейных моделей (Logistic Regression)
X_train_scaled = pd.read_parquet('/home/dr/cbu/X_train_scaled.parquet')
X_test_scaled = pd.read_parquet('/home/dr/cbu/X_test_scaled.parquet')

# Целевая переменная
y_train = pd.read_parquet('/home/dr/cbu/y_train.parquet')['default']
y_test = pd.read_parquet('/home/dr/cbu/y_test.parquet')['default']
```

### Запуск обучения

```bash
# Установите ML библиотеки (если еще не установлены)
pip install scikit-learn xgboost lightgbm

# Запустите скрипт обучения
python3 /home/dr/cbu/train_model.py
```

## Основная информация

- **Обучающая выборка:** 72,000 записей × 108 признаков
- **Тестовая выборка:** 17,999 записей × 108 признаков
- **Дисбаланс классов:** 5.10% дефолтов (1:18.6)
- **Целевая метрика:** AUC (Area Under ROC Curve)

## Файлы

### Данные
- `X_train.parquet` - Обучающие признаки (немасштабированные)
- `X_test.parquet` - Тестовые признаки (немасштабированные)
- `X_train_scaled.parquet` - Обучающие признаки (масштабированные)
- `X_test_scaled.parquet` - Тестовые признаки (масштабированные)
- `y_train.parquet` - Целевая переменная (обучение)
- `y_test.parquet` - Целевая переменная (тест)

### Метаданные
- `feature_names.txt` - Список всех 107 признаков
- `preprocessing_metadata.json` - Детали предобработки
- `class_balance_info.json` - Рекомендуемые веса классов

### Документация
- `SUMMARY_RU.md` - Краткая сводка (ЧИТАТЬ ПЕРВЫМ)
- `TRAINING_DATA_PREP_RU.md` - Полная документация
- `train_model.py` - Скрипт обучения моделей
- `visualize_data.py` - Скрипт визуализации данных

## Инженерия признаков

Создано 16 новых признаков:
- 7 финансовых коэффициентов
- 3 поведенческие метрики
- 2 кредитных показателя
- 4 категориальных бина

## Рекомендации

1. Начните с древовидных моделей (XGBoost, LightGBM)
2. Используйте `class_weight='balanced'` для работы с дисбалансом
3. Оценивайте модели по метрике AUC
4. Проведите кросс-валидацию (k=5)
5. Настройте гиперпараметры через GridSearchCV

## Ожидаемые результаты

- Базовые модели: AUC > 0.70
- После оптимизации: AUC > 0.80

---

**Подробная информация:** См. `SUMMARY_RU.md`
