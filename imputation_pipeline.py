"""
Финальный конвейер импутации пропущенных значений
Проект: Прогнозирование кредитного дефолта

Этот скрипт применяет рекомендованную стратегию импутации:
1. employment_length - медиана (5.20)
2. revolving_balance - медиана по группам credit_score
3. num_delinquencies_2yrs - нули (0)

Автор: Claude Code (Data Science Specialist)
Дата: 2025-11-15
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(file_path):
    """
    Загрузка исходного датасета с пропусками
    """
    print("=" * 80)
    print("КОНВЕЙЕР ИМПУТАЦИИ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
    print("=" * 80)
    print(f"\nЗагрузка данных: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Размер датасета: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    print(f"Доля дефолтов: {df['default'].mean()*100:.2f}%")

    # Check for missing values
    missing_total = df.isnull().sum().sum()
    print(f"Всего пропущенных значений: {missing_total:,}")

    return df


def validate_missing_values(df):
    """
    Валидация наличия ожидаемых пропусков
    """
    print("\n" + "─" * 80)
    print("Валидация пропусков...")
    print("─" * 80)

    expected_cols = ['employment_length', 'revolving_balance', 'num_delinquencies_2yrs']

    for col in expected_cols:
        missing = df[col].isnull().sum()
        pct = missing / len(df) * 100
        print(f"  {col}: {missing:,} пропусков ({pct:.2f}%)")

    return expected_cols


def impute_employment_length(df):
    """
    Импутация employment_length медианой
    """
    print("\n" + "─" * 80)
    print("1. Импутация employment_length")
    print("─" * 80)

    col = 'employment_length'
    missing_before = df[col].isnull().sum()

    # Calculate median from non-missing values
    median_val = df[col].median()

    print(f"Метод: Медиана")
    print(f"Значение: {median_val:.2f} лет")
    print(f"Пропусков до: {missing_before:,}")

    # Impute
    df[col].fillna(median_val, inplace=True)

    missing_after = df[col].isnull().sum()
    print(f"Пропусков после: {missing_after:,}")
    print(f"✓ Заполнено: {missing_before - missing_after:,} значений")

    return df


def impute_revolving_balance(df):
    """
    Импутация revolving_balance медианой по группам credit_score
    """
    print("\n" + "─" * 80)
    print("2. Импутация revolving_balance")
    print("─" * 80)

    col = 'revolving_balance'
    missing_before = df[col].isnull().sum()

    print(f"Метод: Медиана по группам credit_score")
    print(f"Пропусков до: {missing_before:,}")

    # Calculate group medians
    group_medians = df.groupby('credit_score')[col].median()
    print(f"Найдено групп credit_score: {len(group_medians)}")

    # Apply group-wise imputation
    df[col] = df.groupby('credit_score')[col].transform(
        lambda x: x.fillna(x.median())
    )

    # Check if any missing values remain (rare credit_scores with all missing)
    missing_intermediate = df[col].isnull().sum()

    if missing_intermediate > 0:
        # Fallback to overall median
        overall_median = df[col].median()
        df[col].fillna(overall_median, inplace=True)
        print(f"Применен fallback (общая медиана ${overall_median:,.2f}) к {missing_intermediate} записям")

    missing_after = df[col].isnull().sum()
    print(f"Пропусков после: {missing_after:,}")
    print(f"✓ Заполнено: {missing_before - missing_after:,} значений")

    return df


def impute_num_delinquencies(df):
    """
    Импутация num_delinquencies_2yrs нулями
    """
    print("\n" + "─" * 80)
    print("3. Импутация num_delinquencies_2yrs")
    print("─" * 80)

    col = 'num_delinquencies_2yrs'
    missing_before = df[col].isnull().sum()

    print(f"Метод: Заполнение нулями (0)")
    print(f"Обоснование: 98% значений = 0, отсутствие записи = нет просрочек")
    print(f"Пропусков до: {missing_before:,}")

    # Impute with zeros
    df[col].fillna(0, inplace=True)

    missing_after = df[col].isnull().sum()
    print(f"Пропусков после: {missing_after:,}")
    print(f"✓ Заполнено: {missing_before - missing_after:,} значений")

    return df


def validate_imputation(df, original_missing_cols):
    """
    Валидация результатов импутации
    """
    print("\n" + "=" * 80)
    print("ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # Check for remaining missing values
    total_missing = df.isnull().sum().sum()
    print(f"\nВсего пропусков в датасете: {total_missing:,}")

    if total_missing == 0:
        print("✓ Все пропуски успешно заполнены!")
    else:
        print("⚠ ВНИМАНИЕ: Остались пропуски в следующих столбцах:")
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            print(f"  - {col}: {df[col].isnull().sum():,}")

    # Check original columns
    print(f"\nПроверка импутированных столбцов:")
    for col in original_missing_cols:
        missing = df[col].isnull().sum()
        status = "✓" if missing == 0 else "✗"
        print(f"  {status} {col}: {missing:,} пропусков")

    # Check target variable preservation
    print(f"\nБаланс целевой переменной:")
    print(f"  Доля дефолтов: {df['default'].mean()*100:.2f}%")
    print(f"  Класс 0: {(df['default']==0).sum():,} ({(df['default']==0).sum()/len(df)*100:.2f}%)")
    print(f"  Класс 1: {(df['default']==1).sum():,} ({(df['default']==1).sum()/len(df)*100:.2f}%)")

    # Check data types
    print(f"\nТипы данных импутированных столбцов:")
    for col in original_missing_cols:
        print(f"  {col}: {df[col].dtype}")

    # Summary statistics
    print(f"\nСводная статистика импутированных столбцов:")
    print(df[original_missing_cols].describe().to_string())

    return total_missing == 0


def save_results(df, output_dir):
    """
    Сохранение импутированного датасета
    """
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save as CSV
    csv_path = output_dir / 'final_dataset_imputed.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSV сохранен: {csv_path}")
    print(f"  Размер: {df.shape[0]:,} строк × {df.shape[1]} столбцов")

    # Get file size
    csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  Размер файла: {csv_size_mb:.2f} МБ")

    # Save as Parquet (more efficient)
    parquet_path = output_dir / 'final_dataset_imputed.parquet'
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"\n✓ Parquet сохранен: {parquet_path}")

    parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  Размер файла: {parquet_size_mb:.2f} МБ")
    print(f"  Сжатие: {(1 - parquet_size_mb/csv_size_mb)*100:.1f}% относительно CSV")

    print(f"\nРекомендация: Используйте Parquet для более быстрой загрузки")

    return csv_path, parquet_path


def main():
    """
    Основная функция конвейера
    """
    # Configuration
    input_file = '/home/dr/cbu/final_dataset_clean.csv'
    output_dir = '/home/dr/cbu'

    try:
        # Step 1: Load data
        df = load_data(input_file)

        # Step 2: Validate missing values
        missing_cols = validate_missing_values(df)

        # Step 3: Apply imputations
        df = impute_employment_length(df)
        df = impute_revolving_balance(df)
        df = impute_num_delinquencies(df)

        # Step 4: Validate results
        success = validate_imputation(df, missing_cols)

        # Step 5: Save results
        if success:
            csv_path, parquet_path = save_results(df, output_dir)

            print("\n" + "=" * 80)
            print("ИМПУТАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
            print("=" * 80)
            print(f"\nИтоговые файлы:")
            print(f"  1. {csv_path}")
            print(f"  2. {parquet_path}")
            print(f"\nСледующие шаги:")
            print(f"  1. Feature Engineering - создание дополнительных признаков")
            print(f"  2. Feature Selection - отбор значимых признаков")
            print(f"  3. Model Training - обучение моделей ML")
            print(f"  4. Evaluation - оценка на метрике AUC")

        else:
            print("\n⚠ ОШИБКА: Импутация не завершена. Остались пропуски!")
            return 1

        return 0

    except Exception as e:
        print(f"\n✗ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
