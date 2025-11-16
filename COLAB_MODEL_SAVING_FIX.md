# FIX ДЛЯ СОХРАНЕНИЯ МОДЕЛИ В GOOGLE COLAB

## Проблема
Текущий код в `Google_Colab_Leak_Free_90plus_v3.ipynb` сохраняет только калибратор:
```python
joblib.dump(calibrated, 'xgboost_calibrated_ensemble_v3_colab.pkl')
```

Это сохраняет только IsotonicRegression (2.7KB), а весь XGBoost ансамбль теряется!

## РЕШЕНИЕ

Замените ячейку "💾 Save Results" в notebook на этот код:

```python
import joblib
import pickle

print("💾 Сохранение модели и результатов...\n")

# ПРАВИЛЬНОЕ сохранение: создаем dict со всеми компонентами
model_package = {
    'models': models,  # Список из 3 XGBoost моделей
    'weights': weights,  # Оптимальные веса ансамбля
    'calibrated': calibrated,  # Калиброванный ансамбль
    'feature_names': list(X_train.columns),  # Имена признаков (ВАЖНО!)
    'best_params': best_params,  # Гиперпараметры
    'metrics': {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'oof_auc': roc_auc_score(y_train, oof_predictions),
        'train_test_gap': gap,
        'optimal_threshold': optimal_threshold
    }
}

# Сохранение полного package
joblib.dump(model_package, 'xgboost_calibrated_ensemble_v3_colab.pkl')
file_size_mb = os.path.getsize('xgboost_calibrated_ensemble_v3_colab.pkl') / (1024*1024)
print(f"✅ Модель сохранена: xgboost_calibrated_ensemble_v3_colab.pkl ({file_size_mb:.1f} MB)")

# Проверка, что модель сохранилась правильно
test_load = joblib.load('xgboost_calibrated_ensemble_v3_colab.pkl')
print(f"✅ Проверка: {len(test_load['models'])} моделей в ансамбле")
print(f"✅ Проверка: {len(test_load['feature_names'])} признаков")

# Save predictions
predictions_df = pd.DataFrame({
    'test_predictions': calib_test,
    'true_labels': y_test
})
predictions_df.to_csv('test_predictions_v3_colab.csv', index=False)
print("✅ Предсказания сохранены: test_predictions_v3_colab.csv")

# Save feature importance
importance_df.to_csv('feature_importance_v3_colab.csv', index=False)
print("✅ Важность признаков сохранена: feature_importance_v3_colab.csv")

# Save metrics
metrics_df = pd.DataFrame([{
    'train_auc': train_auc,
    'test_auc': test_auc,
    'oof_auc': roc_auc_score(y_train, oof_predictions),
    'train_test_gap': gap,
    'optimal_threshold': optimal_threshold,
    'n_trials': N_OPTIMIZATION_TRIALS,
    'n_cv_folds': N_CV_FOLDS,
    'n_ensemble_models': N_ENSEMBLE_MODELS
}])
metrics_df.to_csv('model_metrics_v3_colab.csv', index=False)
print("✅ Метрики сохранены: model_metrics_v3_colab.csv")

print("\n📦 Все результаты сохранены и готовы к скачиванию!")
print(f"\n⚠️  ВАЖНО: Размер модели должен быть {file_size_mb:.1f} MB, НЕ 2-3 KB!")
```

## Как использовать:

1. Откройте `Google_Colab_Leak_Free_90plus_v3.ipynb` в Google Colab
2. Найдите ячейку "💾 Save Results" (cell 28)
3. Замените код в этой ячейке на код выше
4. Запустите весь notebook заново
5. Скачайте `xgboost_calibrated_ensemble_v3_colab.pkl` (должен быть ~10-50 MB, НЕ 2.7KB!)
6. Замените файл в `task_result1/model/`

## После обучения

Размер правильной модели должен быть **10-50 MB**, не 2.7KB!

Если файл маленький - модель сохранилась неправильно.
