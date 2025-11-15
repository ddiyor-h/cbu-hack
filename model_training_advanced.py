#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced модели для предсказания кредитных дефолтов
Сравнение: LightGBM, CatBoost, XGBoost

Автор: ML Pipeline
Дата: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
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

    # Удалить ID столбцы
    id_columns = ['customer_ref', 'application_id']
    print(f"Удаление ID столбцов: {id_columns}")
    df = df.drop(columns=id_columns, errors='ignore')

    # Отделить целевую переменную
    X = df.drop('default', axis=1)
    y = df['default']

    print(f"\nРаспределение целевой переменной:")
    print(y.value_counts())
    print(f"Доля дефолтов: {y.mean()*100:.2f}%")

    # Определить категориальные признаки
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    print(f"\nКатегориальные признаки ({len(categorical_features)}):")
    print(categorical_features)

    return X, y, categorical_features


def prepare_for_models(X_train, X_test, categorical_features):
    """
    Подготовка данных для разных типов моделей
    """
    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛЕЙ")
    print("=" * 80)

    from sklearn.preprocessing import LabelEncoder

    # Для LightGBM и XGBoost: закодировать категориальные признаки
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
        label_encoders[col] = le

    print(f"Label Encoding применён к {len(categorical_features)} признакам")

    # Для CatBoost: оставить категориальные как есть
    X_train_catboost = X_train.copy()
    X_test_catboost = X_test.copy()

    print("Данные готовы для обучения!")

    return X_train_encoded, X_test_encoded, X_train_catboost, X_test_catboost, label_encoders


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Обучение LightGBM
    """
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛИ: LightGBM")
    print("=" * 80)

    try:
        import lightgbm as lgb
    except ImportError:
        print("⚠️  LightGBM не установлен. Установите: pip install lightgbm")
        return None

    # Вычислить вес положительного класса
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    print("Параметры модели:")
    for key, value in params.items():
        print(f"  - {key}: {value}")

    print("\nОбучение с early stopping...")
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"Лучшее количество итераций: {model.best_iteration_}")
    print("Обучение завершено!")

    return model


def train_catboost(X_train, y_train, X_val, y_val, categorical_features):
    """
    Обучение CatBoost
    """
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛИ: CatBoost")
    print("=" * 80)

    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        print("⚠️  CatBoost не установлен. Установите: pip install catboost")
        return None

    # Индексы категориальных признаков
    cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns]

    print(f"Категориальных признаков для CatBoost: {len(cat_feature_indices)}")

    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': 100,
        'task_type': 'CPU',
        'thread_count': -1,
        'early_stopping_rounds': 50
    }

    print("Параметры модели:")
    for key, value in params.items():
        if key != 'early_stopping_rounds':
            print(f"  - {key}: {value}")

    print("\nОбучение с early stopping...")
    model = CatBoostClassifier(**params)

    model.fit(
        X_train, y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_val, y_val),
        verbose=100,
        plot=False
    )

    print(f"Лучшее количество итераций: {model.get_best_iteration()}")
    print("Обучение завершено!")

    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Обучение XGBoost
    """
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛИ: XGBoost")
    print("=" * 80)

    try:
        import xgboost as xgb
    except ImportError:
        print("⚠️  XGBoost не установлен. Установите: pip install xgboost")
        return None

    # Вычислить вес положительного класса
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }

    print("Параметры модели:")
    for key, value in params.items():
        print(f"  - {key}: {value}")

    print("\nОбучение с early stopping...")
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
        early_stopping_rounds=50
    )

    print(f"Лучшее количество итераций: {model.best_iteration}")
    print("Обучение завершено!")

    return model


def evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Оценка модели на всех наборах данных
    """
    print("\n" + "=" * 80)
    print(f"ОЦЕНКА МОДЕЛИ: {model_name}")
    print("=" * 80)

    # Предсказания
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\nAUC-ROC Score:")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Val:   {val_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    print(f"  Gap (Train-Test): {abs(train_auc - test_auc):.4f}")

    # Average Precision
    test_ap = average_precision_score(y_test, y_test_proba)
    print(f"\nAverage Precision (Test): {test_ap:.4f}")

    # Classification Report
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_test_pred, target_names=['No Default', 'Default']))

    return {
        'model_name': model_name,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'y_test_proba': y_test_proba
    }


def plot_feature_importance(model, model_name, feature_names, top_n=20):
    """
    Визуализация важности признаков
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Модель {model_name} не поддерживает feature_importances_")
        return

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importance['importance'][:top_n][::-1])
    plt.yticks(range(top_n), feature_importance['feature'][:top_n][::-1])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f'{model_name.lower().replace(" ", "_")}_feature_importance.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Feature importance сохранена: {filename}")
    plt.close()


def plot_comparison(results_list):
    """
    Сравнительная визуализация всех моделей
    """
    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ МОДЕЛЕЙ")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ROC Curves
    ax = axes[0, 0]
    for result in results_list:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_test_proba'])
        ax.plot(fpr, tpr, linewidth=2, label=f"{result['model_name']} (AUC={result['test_auc']:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # PR Curves
    ax = axes[0, 1]
    for result in results_list:
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(result['y_test'], result['y_test_proba'])
        ax.plot(recall, precision, linewidth=2, label=f"{result['model_name']} (AP={result['test_ap']:.4f})")

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # AUC Comparison Bar Chart
    ax = axes[1, 0]
    model_names = [r['model_name'] for r in results_list]
    train_aucs = [r['train_auc'] for r in results_list]
    test_aucs = [r['test_auc'] for r in results_list]

    x = np.arange(len(model_names))
    width = 0.35

    ax.bar(x - width/2, train_aucs, width, label='Train AUC', alpha=0.8)
    ax.bar(x + width/2, test_aucs, width, label='Test AUC', alpha=0.8)

    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('AUC Scores Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0.65, 0.85])

    # Metrics Table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for result in results_list:
        table_data.append([
            result['model_name'],
            f"{result['train_auc']:.4f}",
            f"{result['test_auc']:.4f}",
            f"{abs(result['train_auc'] - result['test_auc']):.4f}",
            f"{result['test_ap']:.4f}"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Train AUC', 'Test AUC', 'Gap', 'Test AP'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Заголовок
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    filename = 'models_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Сравнительная визуализация сохранена: {filename}")
    plt.close()


def main():
    """
    Основной pipeline
    """
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "ADVANCED MODELS: GRADIENT BOOSTING" + " " * 29 + "║")
    print("║" + " " * 20 + "Credit Default Prediction" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")

    # Путь к данным
    data_path = '/home/dr/cbu/final_dataset_imputed.parquet'

    # 1. Загрузка данных
    X, y, categorical_features = load_and_prepare_data(data_path)

    # 2. Разделение на train/val/test
    print("\n" + "=" * 80)
    print("РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 80)

    # Сначала отделяем test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Затем делим оставшееся на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
    )

    print(f"Train set:      {len(X_train):,} записей ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val):,} записей ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test):,} записей ({len(X_test)/len(X)*100:.1f}%)")

    # 3. Подготовка данных для моделей
    X_train_enc, X_val_enc, X_train_cat, X_val_cat, label_encoders = prepare_for_models(
        X_train, X_val, categorical_features
    )
    X_test_enc, _, X_test_cat, _, _ = prepare_for_models(
        X_test, X_test, categorical_features
    )

    # 4. Обучение моделей
    models = {}
    results_list = []

    # LightGBM
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "LightGBM" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")

    lgbm_model = train_lightgbm(X_train_enc, y_train, X_val_enc, y_val)
    if lgbm_model is not None:
        models['LightGBM'] = lgbm_model
        lgbm_results = evaluate_model(
            lgbm_model, 'LightGBM',
            X_train_enc, y_train,
            X_val_enc, y_val,
            X_test_enc, y_test
        )
        lgbm_results['y_test'] = y_test
        results_list.append(lgbm_results)
        plot_feature_importance(lgbm_model, 'LightGBM', X_train_enc.columns)

    # CatBoost
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "CatBoost" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")

    catboost_model = train_catboost(X_train_cat, y_train, X_val_cat, y_val, categorical_features)
    if catboost_model is not None:
        models['CatBoost'] = catboost_model
        catboost_results = evaluate_model(
            catboost_model, 'CatBoost',
            X_train_cat, y_train,
            X_val_cat, y_val,
            X_test_cat, y_test
        )
        catboost_results['y_test'] = y_test
        results_list.append(catboost_results)
        plot_feature_importance(catboost_model, 'CatBoost', X_train_cat.columns)

    # XGBoost
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "XGBoost" + " " * 39 + "║")
    print("╚" + "=" * 78 + "╝")

    xgb_model = train_xgboost(X_train_enc, y_train, X_val_enc, y_val)
    if xgb_model is not None:
        models['XGBoost'] = xgb_model
        xgb_results = evaluate_model(
            xgb_model, 'XGBoost',
            X_train_enc, y_train,
            X_val_enc, y_val,
            X_test_enc, y_test
        )
        xgb_results['y_test'] = y_test
        results_list.append(xgb_results)
        plot_feature_importance(xgb_model, 'XGBoost', X_train_enc.columns)

    # 5. Визуализация сравнения
    if len(results_list) > 0:
        plot_comparison(results_list)

    # 6. Сохранение результатов
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # Сохранить метрики
    metrics_df = pd.DataFrame([
        {
            'model': r['model_name'],
            'train_auc': r['train_auc'],
            'val_auc': r['val_auc'],
            'test_auc': r['test_auc'],
            'gap': abs(r['train_auc'] - r['test_auc']),
            'test_ap': r['test_ap']
        }
        for r in results_list
    ])
    metrics_df.to_csv('advanced_models_metrics.csv', index=False)
    print("Метрики сохранены: advanced_models_metrics.csv")

    # Сохранить предсказания лучшей модели
    best_result = max(results_list, key=lambda x: x['test_auc'])
    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_proba': best_result['y_test_proba'],
        'y_pred': (best_result['y_test_proba'] >= 0.5).astype(int)
    })
    predictions_df.to_csv('best_model_predictions.csv', index=False)
    print(f"Предсказания лучшей модели сохранены: best_model_predictions.csv")

    # 7. Итоговый отчет
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 27 + "ИТОГОВЫЙ ОТЧЕТ" + " " * 37 + "║")
    print("╠" + "=" * 78 + "╣")

    for result in sorted(results_list, key=lambda x: x['test_auc'], reverse=True):
        print(f"║  {result['model_name']:15s}  Train: {result['train_auc']:.4f}  Val: {result['val_auc']:.4f}  Test: {result['test_auc']:.4f}  AP: {result['test_ap']:.4f}" + " " * 5 + "║")

    print("╠" + "=" * 78 + "╣")
    print(f"║  ЛУЧШАЯ МОДЕЛЬ: {best_result['model_name']}" + " " * (62 - len(best_result['model_name'])) + "║")
    print(f"║  Test AUC: {best_result['test_auc']:.4f}" + " " * 61 + "║")
    print(f"║  Test AP:  {best_result['test_ap']:.4f}" + " " * 61 + "║")
    print("╠" + "=" * 78 + "╣")

    if best_result['test_auc'] >= 0.80:
        print("║  ✓ РЕЗУЛЬТАТ: ОТЛИЧНЫЙ - Модель показала высокое качество!" + " " * 17 + "║")
    elif best_result['test_auc'] >= 0.75:
        print("║  ✓ РЕЗУЛЬТАТ: ХОРОШИЙ - Качество выше baseline!" + " " * 29 + "║")
    else:
        print("║  ⚠ РЕЗУЛЬТАТ: ТРЕБУЕТ УЛУЧШЕНИЯ - Попробуйте hyperparameter tuning" + " " * 10 + "║")

    print("╠" + "=" * 78 + "╣")
    print("║  СЛЕДУЮЩИЕ ШАГИ:" + " " * 61 + "║")
    print("║  1. Провести hyperparameter tuning лучшей модели (RandomizedSearchCV)" + " " * 7 + "║")
    print("║  2. Попробовать feature engineering если AUC < 0.80" + " " * 27 + "║")
    print("║  3. Рассмотреть ensemble методы (Voting, Stacking)" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")


if __name__ == '__main__':
    main()
