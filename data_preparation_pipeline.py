#!/usr/bin/env python3
"""
Data Preparation Pipeline for Credit Default Prediction
Подход: Очистка каждого датасета отдельно → Объединение в единый набор данных

Автор: Data Science Specialist
Дата: 2025-11-15
"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/dr/cbu/data_cleaning.log'),
        logging.StreamHandler()
    ]
)

class DataCleaner:
    """Класс для очистки и объединения данных"""

    def __init__(self, data_dir="/home/dr/cbu/drive-download-20251115T045945Z-1-001"):
        self.data_dir = Path(data_dir)
        self.clean_data_dir = Path("/home/dr/cbu/cleaned_data")
        self.clean_data_dir.mkdir(exist_ok=True)

        # Счетчики для отчета
        self.cleaning_stats = {
            'application_metadata': {},
            'demographics': {},
            'financial_ratios': {},
            'credit_history': {},
            'loan_details': {},
            'geographic_data': {}
        }

    def clean_currency_column(self, series, column_name=""):
        """Очистка колонок с денежными значениями"""
        before_nulls = series.isnull().sum()

        # Удаляем символы валюты и запятые
        cleaned = series.astype(str).str.replace('$', '', regex=False)
        cleaned = cleaned.str.replace(',', '', regex=False)
        cleaned = cleaned.str.strip()

        # Преобразуем в числа
        cleaned = pd.to_numeric(cleaned, errors='coerce')

        after_nulls = cleaned.isnull().sum()
        cleaning_count = len(series) - before_nulls - (len(cleaned) - after_nulls)

        logging.info(f"  - Очищено {cleaning_count} значений в колонке {column_name}")
        return cleaned, cleaning_count

    def normalize_categorical(self, series, column_name=""):
        """Нормализация категориальных переменных"""
        # Подсчитываем уникальные значения до очистки
        unique_before = series.nunique()

        # Приводим к единому формату
        normalized = series.str.strip()
        normalized = normalized.str.replace('_', ' ', regex=False)
        normalized = normalized.str.replace('-', ' ', regex=False)
        normalized = normalized.str.title()  # Title Case

        unique_after = normalized.nunique()

        logging.info(f"  - Нормализовано {column_name}: {unique_before} → {unique_after} уникальных значений")
        return normalized, unique_before - unique_after

    def clean_application_metadata(self):
        """Очистка application_metadata.csv"""
        logging.info("\n1. ОЧИСТКА application_metadata.csv")

        # Загрузка данных
        df = pd.read_csv(self.data_dir / "application_metadata.csv")
        initial_shape = df.shape
        logging.info(f"  Загружено: {initial_shape[0]} строк, {initial_shape[1]} колонок")

        # Удаление шумовой колонки
        if 'random_noise_1' in df.columns:
            df = df.drop('random_noise_1', axis=1)
            logging.info("  - Удалена колонка random_noise_1")
            self.cleaning_stats['application_metadata']['removed_columns'] = ['random_noise_1']

        # Проверка и удаление дубликатов
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            logging.info(f"  - Удалено {duplicates} дубликатов")
            self.cleaning_stats['application_metadata']['duplicates_removed'] = duplicates

        # Сохранение очищенных данных
        df.to_csv(self.clean_data_dir / "application_metadata_clean.csv", index=False)
        logging.info(f"  ✓ Сохранено: {df.shape[0]} строк, {df.shape[1]} колонок")

        self.cleaning_stats['application_metadata']['final_shape'] = df.shape
        return df

    def clean_demographics(self):
        """Очистка demographics.csv"""
        logging.info("\n2. ОЧИСТКА demographics.csv")

        df = pd.read_csv(self.data_dir / "demographics.csv")
        initial_shape = df.shape
        logging.info(f"  Загружено: {initial_shape[0]} строк, {initial_shape[1]} колонок")

        # Очистка annual_income
        if 'annual_income' in df.columns:
            df['annual_income'], cleaned_count = self.clean_currency_column(
                df['annual_income'], 'annual_income'
            )
            self.cleaning_stats['demographics']['annual_income_cleaned'] = cleaned_count

        # Нормализация employment_type
        if 'employment_type' in df.columns:
            df['employment_type'], normalized_count = self.normalize_categorical(
                df['employment_type'], 'employment_type'
            )
            self.cleaning_stats['demographics']['employment_type_normalized'] = normalized_count

        # Нормализация других категориальных переменных
        for col in ['marital_status', 'education']:
            if col in df.columns:
                df[col], _ = self.normalize_categorical(df[col], col)

        # Сохранение
        df.to_csv(self.clean_data_dir / "demographics_clean.csv", index=False)
        logging.info(f"  ✓ Сохранено: {df.shape[0]} строк, {df.shape[1]} колонок")

        self.cleaning_stats['demographics']['final_shape'] = df.shape
        return df

    def clean_financial_ratios(self):
        """Очистка financial_ratios.jsonl"""
        logging.info("\n3. ОЧИСТКА financial_ratios.jsonl")

        # Загрузка JSONL файла
        records = []
        with open(self.data_dir / "financial_ratios.jsonl", 'r') as f:
            for line in f:
                records.append(json.loads(line))

        df = pd.DataFrame(records)
        initial_shape = df.shape
        logging.info(f"  Загружено: {initial_shape[0]} строк, {initial_shape[1]} колонок")

        # Очистка денежных колонок
        money_columns = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                        'revolving_balance', 'available_credit', 'total_debt_amount']

        for col in money_columns:
            if col in df.columns:
                df[col], cleaned = self.clean_currency_column(df[col], col)
                self.cleaning_stats['financial_ratios'][f'{col}_cleaned'] = cleaned

        # Проверка и пересчет финансовых коэффициентов
        logging.info("  - Проверка корректности финансовых коэффициентов...")

        # debt_to_income_ratio = existing_monthly_debt / monthly_income
        recalc_dti = df['existing_monthly_debt'] / df['monthly_income']
        dti_diff = (df['debt_to_income_ratio'] - recalc_dti).abs()
        incorrect_dti = (dti_diff > 0.01).sum()

        if incorrect_dti > 0:
            logging.warning(f"    Найдено {incorrect_dti} некорректных debt_to_income_ratio")
            df['debt_to_income_ratio'] = recalc_dti
            self.cleaning_stats['financial_ratios']['recalculated_ratios'] = incorrect_dti

        # Сохранение
        df.to_csv(self.clean_data_dir / "financial_ratios_clean.csv", index=False)
        logging.info(f"  ✓ Сохранено: {df.shape[0]} строк, {df.shape[1]} колонок")

        self.cleaning_stats['financial_ratios']['final_shape'] = df.shape
        return df

    def clean_credit_history(self):
        """Очистка credit_history.parquet"""
        logging.info("\n4. ОЧИСТКА credit_history.parquet")

        df = pd.read_parquet(self.data_dir / "credit_history.parquet")
        initial_shape = df.shape
        logging.info(f"  Загружено: {initial_shape[0]} строк, {initial_shape[1]} колонок")

        # Проверка на дубликаты
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            logging.info(f"  - Удалено {duplicates} дубликатов")
            self.cleaning_stats['credit_history']['duplicates_removed'] = duplicates

        # Проверка на пропущенные значения
        missing = df.isnull().sum()
        if missing.any():
            logging.info(f"  - Пропущенные значения: {missing[missing > 0].to_dict()}")
            self.cleaning_stats['credit_history']['missing_values'] = missing[missing > 0].to_dict()

        # Сохранение
        df.to_csv(self.clean_data_dir / "credit_history_clean.csv", index=False)
        logging.info(f"  ✓ Сохранено: {df.shape[0]} строк, {df.shape[1]} колонок")

        self.cleaning_stats['credit_history']['final_shape'] = df.shape
        return df

    def clean_loan_details(self):
        """Очистка loan_details.xlsx"""
        logging.info("\n5. ОЧИСТКА loan_details.xlsx")

        df = pd.read_excel(self.data_dir / "loan_details.xlsx")
        initial_shape = df.shape
        logging.info(f"  Загружено: {initial_shape[0]} строк, {initial_shape[1]} колонок")

        # Очистка денежных колонок, если есть
        for col in df.columns:
            if df[col].dtype == 'object':
                # Проверяем, похожа ли колонка на денежную
                sample = df[col].dropna().astype(str).head(10)
                if sample.str.contains('\\$|,', regex=True).any():
                    df[col], cleaned = self.clean_currency_column(df[col], col)
                    self.cleaning_stats['loan_details'][f'{col}_cleaned'] = cleaned

        # Сохранение
        df.to_csv(self.clean_data_dir / "loan_details_clean.csv", index=False)
        logging.info(f"  ✓ Сохранено: {df.shape[0]} строк, {df.shape[1]} колонок")

        self.cleaning_stats['loan_details']['final_shape'] = df.shape
        return df

    def clean_geographic_data(self):
        """Очистка geographic_data.xml"""
        logging.info("\n6. ОЧИСТКА geographic_data.xml")

        # Парсинг XML
        tree = ET.parse(self.data_dir / "geographic_data.xml")
        root = tree.getroot()

        records = []
        for customer in root.findall('customer'):
            record = {}
            for child in customer:
                record[child.tag] = child.text
            records.append(record)

        df = pd.DataFrame(records)
        initial_shape = df.shape
        logging.info(f"  Загружено: {initial_shape[0]} строк, {initial_shape[1]} колонок")

        # Преобразование числовых колонок
        numeric_cols = ['id', 'regional_unemployment_rate', 'regional_median_income',
                       'regional_median_rent', 'housing_price_index', 'cost_of_living_index']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Нормализация state
        if 'state' in df.columns:
            df['state'], _ = self.normalize_categorical(df['state'], 'state')

        # Сохранение
        df.to_csv(self.clean_data_dir / "geographic_data_clean.csv", index=False)
        logging.info(f"  ✓ Сохранено: {df.shape[0]} строк, {df.shape[1]} колонок")

        self.cleaning_stats['geographic_data']['final_shape'] = df.shape
        return df

    def merge_datasets(self):
        """Объединение всех очищенных датасетов"""
        logging.info("\n" + "="*60)
        logging.info("ЭТАП 2: ОБЪЕДИНЕНИЕ ОЧИЩЕННЫХ ДАТАСЕТОВ")
        logging.info("="*60)

        # Загрузка очищенных данных
        app_meta = pd.read_csv(self.clean_data_dir / "application_metadata_clean.csv")
        demographics = pd.read_csv(self.clean_data_dir / "demographics_clean.csv")
        financial = pd.read_csv(self.clean_data_dir / "financial_ratios_clean.csv")
        credit = pd.read_csv(self.clean_data_dir / "credit_history_clean.csv")
        loan = pd.read_csv(self.clean_data_dir / "loan_details_clean.csv")
        geo = pd.read_csv(self.clean_data_dir / "geographic_data_clean.csv")

        logging.info(f"\nРазмеры очищенных датасетов:")
        logging.info(f"  - application_metadata: {app_meta.shape}")
        logging.info(f"  - demographics: {demographics.shape}")
        logging.info(f"  - financial_ratios: {financial.shape}")
        logging.info(f"  - credit_history: {credit.shape}")
        logging.info(f"  - loan_details: {loan.shape}")
        logging.info(f"  - geographic_data: {geo.shape}")

        # Начинаем с application_metadata (содержит целевую переменную)
        merged = app_meta.copy()
        logging.info(f"\nБазовый датасет: {merged.shape}")

        # 1. Объединяем с demographics
        merged = pd.merge(merged, demographics,
                         left_on='customer_ref', right_on='cust_id',
                         how='left', suffixes=('', '_demo'))
        merged = merged.drop('cust_id', axis=1)
        logging.info(f"После demographics: {merged.shape}")

        # 2. Объединяем с financial_ratios
        merged = pd.merge(merged, financial,
                         left_on='customer_ref', right_on='cust_num',
                         how='left', suffixes=('', '_fin'))
        merged = merged.drop('cust_num', axis=1)
        logging.info(f"После financial_ratios: {merged.shape}")

        # 3. Объединяем с credit_history
        # Определяем ключ для credit_history
        if 'customer_id' in credit.columns:
            merge_key = 'customer_id'
        elif 'cust_ref' in credit.columns:
            merge_key = 'cust_ref'
        else:
            merge_key = credit.columns[0]  # Берем первую колонку как ключ

        merged = pd.merge(merged, credit,
                         left_on='customer_ref', right_on=merge_key,
                         how='left', suffixes=('', '_credit'))
        if merge_key != 'customer_ref' and merge_key in merged.columns:
            merged = merged.drop(merge_key, axis=1)
        logging.info(f"После credit_history: {merged.shape}")

        # 4. Объединяем с loan_details
        # Определяем ключ для loan_details
        if 'customer_ref' in loan.columns:
            merge_key = 'customer_ref'
        elif 'loan_id' in loan.columns:
            # Если есть loan_id, проверяем связь с application_id
            merge_key = 'loan_id' if 'loan_id' in app_meta.columns else loan.columns[0]
        else:
            merge_key = loan.columns[0]

        if merge_key == 'customer_ref':
            merged = pd.merge(merged, loan,
                             on='customer_ref',
                             how='left', suffixes=('', '_loan'))
        else:
            merged = pd.merge(merged, loan,
                             left_on='customer_ref', right_on=merge_key,
                             how='left', suffixes=('', '_loan'))
            if merge_key in merged.columns:
                merged = merged.drop(merge_key, axis=1)
        logging.info(f"После loan_details: {merged.shape}")

        # 5. Объединяем с geographic_data
        merged = pd.merge(merged, geo,
                         left_on='customer_ref', right_on='id',
                         how='left', suffixes=('', '_geo'))
        merged = merged.drop('id', axis=1)
        logging.info(f"После geographic_data: {merged.shape}")

        # Финальная проверка
        logging.info(f"\n✓ ФИНАЛЬНЫЙ ДАТАСЕТ: {merged.shape[0]} строк, {merged.shape[1]} колонок")

        # Проверка на пропущенные значения после объединения
        missing_summary = merged.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        if len(missing_cols) > 0:
            logging.warning(f"\nПропущенные значения после объединения:")
            for col, count in missing_cols.items():
                logging.warning(f"  - {col}: {count} ({count/len(merged)*100:.1f}%)")

        return merged

    def generate_report(self, final_df):
        """Генерация финального отчета"""
        report = []
        report.append("="*70)
        report.append("ОТЧЕТ О ПОДГОТОВКЕ ДАННЫХ")
        report.append("="*70)
        report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("ПОДХОД: Очистка каждого датасета отдельно → Объединение")
        report.append("")

        report.append("СТАТИСТИКА ОЧИСТКИ ПО ФАЙЛАМ:")
        report.append("-"*40)

        for dataset, stats in self.cleaning_stats.items():
            if stats:
                report.append(f"\n{dataset}:")
                for key, value in stats.items():
                    report.append(f"  - {key}: {value}")

        report.append("")
        report.append("ФИНАЛЬНЫЙ ДАТАСЕТ:")
        report.append("-"*40)
        report.append(f"Размер: {final_df.shape[0]} строк × {final_df.shape[1]} колонок")
        report.append(f"Целевая переменная (default): {final_df['default'].value_counts().to_dict()}")
        report.append(f"Пропущенные значения: {final_df.isnull().sum().sum()}")
        report.append(f"Дубликаты: {final_df.duplicated().sum()}")

        # Сохранение отчета
        report_text = "\n".join(report)
        with open("/home/dr/cbu/data_quality_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        print("\n" + report_text)

        return report_text

    def run_pipeline(self):
        """Запуск полного пайплайна очистки и объединения"""
        logging.info("="*60)
        logging.info("ЗАПУСК ПАЙПЛАЙНА ПОДГОТОВКИ ДАННЫХ")
        logging.info("="*60)

        # Этап 1: Очистка каждого датасета
        logging.info("\n" + "="*60)
        logging.info("ЭТАП 1: ОЧИСТКА КАЖДОГО ДАТАСЕТА ОТДЕЛЬНО")
        logging.info("="*60)

        self.clean_application_metadata()
        self.clean_demographics()
        self.clean_financial_ratios()
        self.clean_credit_history()
        self.clean_loan_details()
        self.clean_geographic_data()

        # Этап 2: Объединение
        merged_df = self.merge_datasets()

        # Сохранение финального датасета
        logging.info("\nСохранение финального датасета...")
        merged_df.to_csv("/home/dr/cbu/final_dataset_clean.csv", index=False)
        merged_df.to_parquet("/home/dr/cbu/final_dataset_clean.parquet", index=False)
        logging.info("✓ Сохранено в final_dataset_clean.csv и final_dataset_clean.parquet")

        # Генерация отчета
        self.generate_report(merged_df)

        return merged_df


if __name__ == "__main__":
    # Запуск пайплайна
    cleaner = DataCleaner()
    final_dataset = cleaner.run_pipeline()

    print("\n" + "="*60)
    print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print("="*60)
    print("\nФайлы созданы:")
    print("  • /home/dr/cbu/cleaned_data/ - очищенные датасеты")
    print("  • /home/dr/cbu/final_dataset_clean.csv - финальный CSV")
    print("  • /home/dr/cbu/final_dataset_clean.parquet - финальный Parquet")
    print("  • /home/dr/cbu/data_quality_report.txt - отчет о качестве")
    print("  • /home/dr/cbu/data_cleaning.log - лог процесса")