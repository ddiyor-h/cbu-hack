#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework для оценки и сравнения моделей
Утилиты для визуализации и анализа результатов

Автор: ML Pipeline
Дата: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
sns.set_style('whitegrid')
sns.set_palette('husl')


class ModelEvaluator:
    """
    Класс для комплексной оценки моделей машинного обучения
    """

    def __init__(self, model_name='Model'):
        self.model_name = model_name
        self.results = {}

    def evaluate(self, y_true, y_pred_proba, y_pred=None, threshold=0.5):
        """
        Полная оценка модели

        Parameters:
        -----------
        y_true : array-like
            Истинные метки классов
        y_pred_proba : array-like
            Предсказанные вероятности положительного класса
        y_pred : array-like, optional
            Предсказанные метки классов (если None, используется threshold)
        threshold : float
            Порог для преобразования вероятностей в метки
        """
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)

        # AUC-ROC
        auc = roc_auc_score(y_true, y_pred_proba)

        # Average Precision
        ap = average_precision_score(y_true, y_pred_proba)

        # Precision, Recall, F1
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.results = {
            'auc': auc,
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'confusion_matrix': cm,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'threshold': threshold
        }

        return self.results

    def print_report(self, y_true, y_pred_proba, threshold=0.5):
        """
        Печать детального отчета
        """
        results = self.evaluate(y_true, y_pred_proba, threshold=threshold)

        print("=" * 80)
        print(f"ОТЧЕТ ОЦЕНКИ МОДЕЛИ: {self.model_name}")
        print("=" * 80)

        print(f"\n1. ОСНОВНЫЕ МЕТРИКИ:")
        print(f"   AUC-ROC:     {results['auc']:.4f}")
        print(f"   AP Score:    {results['ap']:.4f}")
        print(f"   Precision:   {results['precision']:.4f}")
        print(f"   Recall:      {results['recall']:.4f}")
        print(f"   F1-Score:    {results['f1']:.4f}")
        print(f"   Specificity: {results['specificity']:.4f}")

        print(f"\n2. CONFUSION MATRIX (threshold={threshold}):")
        print(f"   TN: {results['tn']:6d}  |  FP: {results['fp']:6d}")
        print(f"   FN: {results['fn']:6d}  |  TP: {results['tp']:6d}")

        print(f"\n3. ИНТЕРПРЕТАЦИЯ:")
        total = results['tn'] + results['fp'] + results['fn'] + results['tp']
        accuracy = (results['tn'] + results['tp']) / total
        print(f"   Accuracy:           {accuracy:.4f}")
        print(f"   False Positive Rate: {results['fp'] / (results['fp'] + results['tn']):.4f}")
        print(f"   False Negative Rate: {results['fn'] / (results['fn'] + results['tp']):.4f}")

        print("=" * 80)

    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Визуализация ROC кривой
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{self.model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC кривая сохранена: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_pr_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Визуализация Precision-Recall кривой
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'{self.model_name} (AP = {ap:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR кривая сохранена: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred_proba, threshold=0.5, save_path=None):
        """
        Визуализация Confusion Matrix
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['No Default (0)', 'Default (1)'],
                    yticklabels=['No Default (0)', 'Default (1)'],
                    annot_kws={'fontsize': 14})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix - {self.model_name} (threshold={threshold})',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix сохранена: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_probability_distribution(self, y_true, y_pred_proba, save_path=None):
        """
        Распределение предсказанных вероятностей
        """
        plt.figure(figsize=(10, 6))

        plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6,
                 label='No Default (0)', density=True, color='blue')
        plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6,
                 label='Default (1)', density=True, color='red')

        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Distribution of Predicted Probabilities - {self.model_name}',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Распределение вероятностей сохранено: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_threshold_analysis(self, y_true, y_pred_proba, save_path=None):
        """
        Анализ метрик при разных порогах
        """
        thresholds = np.linspace(0, 1, 101)
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, linewidth=2, label='Precision')
        plt.plot(thresholds, recalls, linewidth=2, label='Recall')
        plt.plot(thresholds, f1_scores, linewidth=2, label='F1-Score')

        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Metrics vs Threshold - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Анализ порогов сохранен: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_all(self, y_true, y_pred_proba, threshold=0.5, save_prefix=None):
        """
        Создать все визуализации
        """
        if save_prefix:
            self.plot_roc_curve(y_true, y_pred_proba, f'{save_prefix}_roc.png')
            self.plot_pr_curve(y_true, y_pred_proba, f'{save_prefix}_pr.png')
            self.plot_confusion_matrix(y_true, y_pred_proba, threshold, f'{save_prefix}_cm.png')
            self.plot_probability_distribution(y_true, y_pred_proba, f'{save_prefix}_dist.png')
            self.plot_threshold_analysis(y_true, y_pred_proba, f'{save_prefix}_threshold.png')
        else:
            self.plot_roc_curve(y_true, y_pred_proba)
            self.plot_pr_curve(y_true, y_pred_proba)
            self.plot_confusion_matrix(y_true, y_pred_proba, threshold)
            self.plot_probability_distribution(y_true, y_pred_proba)
            self.plot_threshold_analysis(y_true, y_pred_proba)


class ModelsComparator:
    """
    Класс для сравнения нескольких моделей
    """

    def __init__(self):
        self.models = {}

    def add_model(self, name, y_true, y_pred_proba):
        """
        Добавить модель для сравнения
        """
        self.models[name] = {
            'y_true': y_true,
            'y_pred_proba': y_pred_proba,
            'auc': roc_auc_score(y_true, y_pred_proba),
            'ap': average_precision_score(y_true, y_pred_proba)
        }

    def compare_roc_curves(self, save_path=None):
        """
        Сравнение ROC кривых
        """
        plt.figure(figsize=(10, 8))

        for name, data in self.models.items():
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_pred_proba'])
            plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={data['auc']:.4f})")

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сравнение ROC кривых сохранено: {save_path}")
        else:
            plt.show()

        plt.close()

    def compare_pr_curves(self, save_path=None):
        """
        Сравнение Precision-Recall кривых
        """
        plt.figure(figsize=(10, 8))

        for name, data in self.models.items():
            precision, recall, _ = precision_recall_curve(data['y_true'], data['y_pred_proba'])
            plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={data['ap']:.4f})")

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сравнение PR кривых сохранено: {save_path}")
        else:
            plt.show()

        plt.close()

    def compare_metrics(self, save_path=None):
        """
        Сравнение метрик в виде bar chart
        """
        names = list(self.models.keys())
        aucs = [self.models[name]['auc'] for name in names]
        aps = [self.models[name]['ap'] for name in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, aucs, width, label='AUC-ROC', alpha=0.8)
        ax.bar(x + width/2, aps, width, label='AP Score', alpha=0.8)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Models Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сравнение метрик сохранено: {save_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self):
        """
        Печать сводной таблицы
        """
        print("=" * 80)
        print("СВОДНАЯ ТАБЛИЦА МОДЕЛЕЙ")
        print("=" * 80)
        print(f"{'Model Name':<25} {'AUC-ROC':<12} {'AP Score':<12}")
        print("-" * 80)

        for name, data in sorted(self.models.items(), key=lambda x: x[1]['auc'], reverse=True):
            print(f"{name:<25} {data['auc']:<12.4f} {data['ap']:<12.4f}")

        print("=" * 80)

        # Лучшая модель
        best_name = max(self.models.items(), key=lambda x: x[1]['auc'])[0]
        print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_name}")
        print(f"  AUC-ROC: {self.models[best_name]['auc']:.4f}")
        print(f"  AP Score: {self.models[best_name]['ap']:.4f}")


def main():
    """
    Пример использования
    """
    print("=" * 80)
    print("EVALUATION FRAMEWORK - ПРИМЕР ИСПОЛЬЗОВАНИЯ")
    print("=" * 80)

    # Пример: загрузка предсказаний
    # predictions = pd.read_csv('baseline_predictions.csv')
    # y_true = predictions['y_true'].values
    # y_proba = predictions['y_proba'].values

    # Использование ModelEvaluator
    # evaluator = ModelEvaluator(model_name='Logistic Regression')
    # evaluator.print_report(y_true, y_proba, threshold=0.5)
    # evaluator.plot_all(y_true, y_proba, save_prefix='model_evaluation')

    # Использование ModelsComparator
    # comparator = ModelsComparator()
    # comparator.add_model('Logistic Regression', y_true, y_proba_lr)
    # comparator.add_model('LightGBM', y_true, y_proba_lgbm)
    # comparator.add_model('CatBoost', y_true, y_proba_catboost)
    # comparator.compare_roc_curves(save_path='models_comparison_roc.png')
    # comparator.print_summary()

    print("\nДля использования импортируйте классы:")
    print("  from evaluation_framework import ModelEvaluator, ModelsComparator")


if __name__ == '__main__':
    main()
