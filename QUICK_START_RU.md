# Быстрый старт: Обучение моделей для предсказания кредитных дефолтов

## Характеристики задачи

**Датасет**: `/home/dr/cbu/final_dataset_imputed.parquet`
- 89,999 записей × 62 признака
- Целевая переменная: `default` (0 = нет дефолта, 1 = дефолт)
- Дисбаланс классов: 5.10% дефолтов (1:18.6)
- Основная метрика: **AUC-ROC**

## Запуск обучения

### Шаг 1: Baseline модель (Логистическая регрессия)

```bash
python3 /home/dr/cbu/model_training_baseline.py
```

**Что делает**:
- Обучает логистическую регрессию с `class_weight='balanced'`
- Проводит 5-fold стратифицированную кросс-валидацию
- Оценивает модель на test set
- Создает визуализации (ROC, PR, confusion matrix)
- Сохраняет результаты в `baseline_predictions.csv` и `baseline_metrics.csv`

**Ожидаемое время**: 5-10 минут
**Ожидаемый AUC**: 0.70-0.75

---

### Шаг 2: Advanced модели (Gradient Boosting)

```bash
python3 /home/dr/cbu/model_training_advanced.py
```

**Что делает**:
- Обучает три модели: LightGBM, CatBoost, XGBoost
- Использует validation set для early stopping
- Сравнивает результаты всех моделей
- Создает сравнительные визуализации
- Сохраняет feature importance для каждой модели
- Сохраняет результаты в `advanced_models_metrics.csv`

**Ожидаемое время**: 30-60 минут
**Ожидаемый AUC**: 0.78-0.82

**Требования**: Установите библиотеки gradient boosting
```bash
pip install lightgbm catboost xgboost
```

---

### Шаг 3: Использование evaluation framework

```python
from evaluation_framework import ModelEvaluator, ModelsComparator

# Оценка одной модели
evaluator = ModelEvaluator(model_name='LightGBM')
evaluator.print_report(y_test, y_pred_proba, threshold=0.5)
evaluator.plot_all(y_test, y_pred_proba, save_prefix='lightgbm_eval')

# Сравнение моделей
comparator = ModelsComparator()
comparator.add_model('Baseline LR', y_test, y_proba_lr)
comparator.add_model('LightGBM', y_test, y_proba_lgbm)
comparator.add_model('CatBoost', y_test, y_proba_catboost)
comparator.compare_roc_curves(save_path='comparison_roc.png')
comparator.print_summary()
```

---

## Рекомендуемая последовательность

1. **Запустить baseline** → Установить минимальную планку качества
2. **Запустить advanced** → Найти лучшую модель
3. **Hyperparameter tuning** → Улучшить лучшую модель (если нужно)
4. **Feature engineering** → Создать новые признаки (опционально)
5. **Ensemble** → Объединить модели (опционально)

---

## Интерпретация результатов

### AUC-ROC
- **< 0.70**: Слабая модель, требует улучшения
- **0.70-0.75**: Приемлемая baseline модель
- **0.75-0.80**: Хорошая модель
- **0.80-0.85**: Очень хорошая модель
- **> 0.85**: Отличная модель (проверить на overfitting!)

### Gap между Train и Test AUC
- **< 0.03**: Нет переобучения
- **0.03-0.05**: Умеренное переобучение
- **> 0.05**: Сильное переобучение, требуется регуляризация

---

## Топ-1 рекомендация

**LightGBM** с параметрами:
```python
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': 18.6,  # Дисбаланс классов
    'random_state': 42,
    'n_jobs': -1
}
```

**Почему LightGBM**:
- Оптимален для метрики AUC
- Быстрое обучение
- Встроенная обработка дисбаланса
- Высокое качество предсказаний

---

## Обработка дисбаланса классов

**Критично**: Датасет имеет сильный дисбаланс (94.9% vs 5.1%)

**Решение 1 (рекомендуется)**: Class weights
```python
# Для sklearn
class_weight='balanced'

# Для LightGBM/XGBoost
scale_pos_weight=18.6

# Для CatBoost
auto_class_weights='Balanced'
```

**Решение 2 (опционально)**: SMOTE
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## Файлы проекта

```
/home/dr/cbu/
├── MODEL_SELECTION_STRATEGY_RU.md      # Полная стратегия (ЭТО ГЛАВНЫЙ ДОКУМЕНТ!)
├── QUICK_START_RU.md                   # Быстрый старт (этот файл)
├── model_training_baseline.py          # Baseline модель
├── model_training_advanced.py          # Advanced модели
├── evaluation_framework.py             # Утилиты для оценки
├── final_dataset_imputed.parquet       # Очищенные данные
└── (результаты обучения будут здесь)
```

---

## Полная документация

Для детальной информации см. **MODEL_SELECTION_STRATEGY_RU.md**, который содержит:
- Анализ характеристик данных
- Обоснование выбора каждого алгоритма
- Детальные рекомендации по гиперпараметрам
- Стратегию кросс-валидации
- Методы feature engineering
- Ensemble подходы
- Чек-лист перед обучением
- Риски и предупреждения

---

**Успехов в обучении моделей!**
