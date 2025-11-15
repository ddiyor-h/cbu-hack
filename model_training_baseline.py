#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline модель для предсказания кредитных дефолтов
Логистическая регрессия с обработкой дисбаланса классов

Автор: ML Pipeline
Дата: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_and_prepare_data(file_path):
    """
    Загрузка и предобработка данных
    """
    print("=" * 80)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 80)

    df = pd.read_parquet(file_path)
    print(f"Загружено: {df.shape[0]:,} записей × {df.shape[1]} признаков\n")

    # Удалить ID столбцы (избежать утечки данных)
    id_columns = ['customer_ref', 'application_id']
    print(f"Удаление ID столбцов: {id_columns}")
    df = df.drop(columns=id_columns, errors='ignore')

    # Отделить целевую переменную
    X = df.drop('default', axis=1)
    y = df['default']

    print(f"\nРаспределение целевой переменной:")
    print(y.value_counts())
    print(f"Доля дефолтов: {y.mean()*100:.2f}%")
    print(f"Дисбаланс классов: 1:{(y==0).sum() / (y==1).sum():.1f}")

    return X, y


def preprocess_features(X_train, X_test):
    """
    Предобработка признаков для логистической регрессии
    """
    print("\n" + "=" * 80)
    print("ПРЕДОБРАБОТКА ПРИЗНАКОВ")
    print("=" * 80)

    # Разделить на числовые и категориальные
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"\nЧисловых признаков: {len(numeric_features)}")
    print(f"Категориальных признаков: {len(categorical_features)}")

    # Обработка категориальных признаков (Label Encoding)
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
        X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
        label_encoders[col] = le

    # Стандартизация всех признаков
    print("\nСтандартизация признаков (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # Преобразовать обратно в DataFrame для удобства
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_processed.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_processed.columns)

    print("Предобработка завершена!")

    return X_train_scaled, X_test_scaled, scaler, label_encoders


def train_baseline_model(X_train, y_train):
    """
    Обучение baseline модели: Логистическая регрессия
    """
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ BASELINE МОДЕЛИ: LOGISTIC REGRESSION")
    print("=" * 80)

    # Модель с balanced class weights
    model = LogisticRegression(
        class_weight='balanced',    # КРИТИЧНО: обработка дисбаланса
        penalty='l2',               # L2 регуляризация
        C=1.0,                      # Сила регуляризации (меньше = сильнее)
        solver='saga',              # Эффективный solver
        max_iter=1000,              # Достаточно итераций
        random_state=42,
        n_jobs=-1,                  # Использовать все ядра
        verbose=0
    )

    print("Параметры модели:")
    print(f"  - class_weight: balanced")
    print(f"  - penalty: l2")
    print(f"  - C: 1.0")
    print(f"  - solver: saga")
    print(f"  - max_iter: 1000")

    print("\nОбучение модели...")
    model.fit(X_train, y_train)
    print("Обучение завершено!")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Полная оценка модели
    """
    print("\n" + "=" * 80)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 80)

    # Предсказания
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC (основная метрика)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\nAUC-ROC Score:")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    print(f"  Gap:   {abs(train_auc - test_auc):.4f}")

    if abs(train_auc - test_auc) > 0.05:
        print("  ⚠️  ВНИМАНИЕ: Большой gap между train и test - возможно переобучение")
    else:
        print("  ✓ Gap приемлемый - нет явного переобучения")

    # Average Precision
    test_ap = average_precision_score(y_test, y_test_proba)
    print(f"\nAverage Precision (AP): {test_ap:.4f}")

    # Confusion Matrix и Classification Report (порог 0.5)
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    print("\n" + "-" * 80)
    print("Confusion Matrix (threshold=0.5):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_test_pred, target_names=['No Default', 'Default']))

    # Метрики при разных порогах
    print("\n" + "-" * 80)
    print("Метрики при разных порогах:")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 48)

    from sklearn.metrics import precision_score, recall_score, f1_score

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred_thresh = (y_test_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        print(f"{threshold:<12.1f} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f}")

    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'y_test_proba': y_test_proba
    }


def cross_validate_model(model, X, y, cv_folds=5):
    """
    Кросс-валидация модели
    """
    print("\n" + "=" * 80)
    print("КРОСС-ВАЛИДАЦИЯ")
    print("=" * 80)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print(f"Стратифицированная {cv_folds}-Fold Cross-Validation...")
    print("(Это может занять несколько минут)\n")

    cv_scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    print(f"CV AUC по фолдам: {cv_scores}")
    print(f"\nСредний CV AUC:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Min CV AUC:       {cv_scores.min():.4f}")
    print(f"Max CV AUC:       {cv_scores.max():.4f}")

    return cv_scores


def plot_results(y_test, y_test_proba, save_prefix='baseline'):
    """
    Визуализация результатов
    """
    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    ap = average_precision_score(y_test, y_test_proba)

    plt.subplot(2, 2, 2)
    plt.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Распределение предсказанных вероятностей
    plt.subplot(2, 2, 3)
    plt.hist(y_test_proba[y_test == 0], bins=50, alpha=0.6, label='No Default (0)', density=True)
    plt.hist(y_test_proba[y_test == 1], bins=50, alpha=0.6, label='Default (1)', density=True)
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Confusion Matrix
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)

    plt.subplot(2, 2, 4)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix (threshold=0.5)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    filename = f'{save_prefix}_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Визуализация сохранена: {filename}")
    plt.close()


def main():
    """
    Основной pipeline
    """
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "BASELINE MODEL: LOGISTIC REGRESSION" + " " * 28 + "║")
    print("║" + " " * 20 + "Credit Default Prediction" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")

    # Путь к данным
    data_path = '/home/dr/cbu/final_dataset_imputed.parquet'

    # 1. Загрузка данных
    X, y = load_and_prepare_data(data_path)

    # 2. Разделение на train/test
    print("\n" + "=" * 80)
    print("РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"Train set: {len(X_train):,} записей ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Класс 0: {(y_train == 0).sum():,}")
    print(f"  - Класс 1: {(y_train == 1).sum():,}")
    print(f"Test set:  {len(X_test):,} записей ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  - Класс 0: {(y_test == 0).sum():,}")
    print(f"  - Класс 1: {(y_test == 1).sum():,}")

    # 3. Предобработка признаков
    X_train_scaled, X_test_scaled, scaler, label_encoders = preprocess_features(X_train, X_test)

    # 4. Обучение модели
    model = train_baseline_model(X_train_scaled, y_train)

    # 5. Кросс-валидация
    cv_scores = cross_validate_model(model, X_train_scaled, y_train, cv_folds=5)

    # 6. Оценка на test set
    results = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # 7. Визуализация
    plot_results(y_test, results['y_test_proba'], save_prefix='baseline_logistic')

    # 8. Сохранение результатов
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # Сохранить предсказания
    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_proba': results['y_test_proba'],
        'y_pred': (results['y_test_proba'] >= 0.5).astype(int)
    })
    predictions_df.to_csv('baseline_predictions.csv', index=False)
    print("Предсказания сохранены: baseline_predictions.csv")

    # Сохранить метрики
    metrics_df = pd.DataFrame({
        'model': ['Logistic Regression'],
        'train_auc': [results['train_auc']],
        'test_auc': [results['test_auc']],
        'cv_auc_mean': [cv_scores.mean()],
        'cv_auc_std': [cv_scores.std()],
        'test_ap': [results['test_ap']]
    })
    metrics_df.to_csv('baseline_metrics.csv', index=False)
    print("Метрики сохранены: baseline_metrics.csv")

    # 9. Итоговый отчет
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "ИТОГОВЫЙ ОТЧЕТ" + " " * 34 + "║")
    print("╠" + "=" * 78 + "╣")
    print(f"║  Модель: Logistic Regression (class_weight='balanced')" + " " * 23 + "║")
    print(f"║  Train AUC:     {results['train_auc']:.4f}" + " " * 58 + "║")
    print(f"║  Test AUC:      {results['test_auc']:.4f}" + " " * 58 + "║")
    print(f"║  CV AUC:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}" + " " * 49 + "║")
    print(f"║  Test AP:       {results['test_ap']:.4f}" + " " * 58 + "║")
    print("╠" + "=" * 78 + "╣")

    if results['test_auc'] >= 0.75:
        print("║  ✓ РЕЗУЛЬТАТ: ОТЛИЧНЫЙ - Baseline модель показала хорошее качество!" + " " * 11 + "║")
    elif results['test_auc'] >= 0.70:
        print("║  ✓ РЕЗУЛЬТАТ: ХОРОШИЙ - Baseline модель адекватна, есть потенциал улучшения" + " " * 1 + "║")
    else:
        print("║  ⚠ РЕЗУЛЬТАТ: НИЗКИЙ - Требуется более сложная модель" + " " * 23 + "║")

    print("╠" + "=" * 78 + "╣")
    print("║  СЛЕДУЮЩИЕ ШАГИ:" + " " * 61 + "║")
    print("║  1. Запустить model_training_advanced.py для gradient boosting моделей" + " " * 8 + "║")
    print("║  2. Сравнить результаты с baseline" + " " * 43 + "║")
    print("║  3. Выбрать лучшую модель для hyperparameter tuning" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")


if __name__ == '__main__':
    main()
