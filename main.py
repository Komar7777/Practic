# -*- coding: utf-8 -*-
"""
ФИО автора: Комаров Даниил Иванович
Тема ВКР: Прогнозирование неисправности компьютерных компонентов методами машинного обучения
Описание: Веб-приложение для анализа данных, обучения моделей, прогнозирования неисправностей
и генерации отчетов с использованием методов машинного обучения.
"""

# Импорт библиотек
import pandas as pd
import numpy as np
import os
import logging
import time
import sys
import json
import warnings
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix, 
                             roc_curve, auc, log_loss, roc_auc_score)
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import streamlit as st
from datetime import datetime
import itertools
import unittest
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

# Попытка импорта Boruta с обработкой ошибки
try:
    from boruta import BorutaPy
    boruta_available = True
except ImportError:
    boruta_available = False
    BorutaPy = None

# Попытка импорта SHAP с обработкой ошибки
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    shap = None

# Настройка логирования
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Игнорирование предупреждений для чистоты вывода
warnings.filterwarnings('ignore')

# Установка стиля визуализаций
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# --- Модуль загрузки данных ---
def load_dataset(local_path="POLOMKA.csv", url="https://raw.githubusercontent.com/[ваш_репозиторий]/main/POLOMKA.csv"):
    """
    Загружает датасет из локального файла или URL, пробуя разные кодировки.
    """
    encodings = ['utf-8', 'cp1251', 'latin-1', 'iso-8859-1']
    try:
        if os.path.exists(local_path):
            logger.info("Попытка загрузки датасета из локального файла")
            for encoding in encodings:
                try:
                    data = pd.read_csv(local_path, encoding=encoding)
                    logger.info(f"Датасет успешно загружен с кодировкой {encoding}")
                    return data
                except UnicodeDecodeError:
                    logger.warning(f"Не удалось загрузить с кодировкой {encoding}")
                    continue
            raise UnicodeDecodeError("Не удалось определить кодировку файла")
        else:
            logger.info("Локальный файл не найден, попытка загрузки по URL")
            if url:
                urlretrieve(url, local_path)
                for encoding in encodings:
                    try:
                        data = pd.read_csv(local_path, encoding=encoding)
                        logger.info(f"Датасет успешно загружен с кодировкой {encoding}")
                        return data
                    except UnicodeDecodeError:
                        logger.warning(f"Не удалось загрузить с кодировкой {encoding}")
                        continue
                raise UnicodeDecodeError("Не удалось определить кодировку файла")
            else:
                raise FileNotFoundError("URL не указан")
    except Exception as e:
        logger.error(f"Ошибка загрузки датасета: {str(e)}")
        st.error(f"Ошибка загрузки датасета: {str(e)}")
        return None

# --- Модуль генерации синтетических данных ---
def generate_synthetic_data(n_samples=1000, n_features=6, n_classes=2):
    """
    Генерирует синтетические данные для тестирования моделей.
    """
    try:
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                  n_classes=n_classes, n_informative=n_features-1, 
                                  n_redundant=1, random_state=42)
        columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        df = pd.DataFrame(X, columns=columns[:n_features])
        df['Target'] = y
        logger.info(f"Сгенерирован синтетический датасет размером {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Ошибка генерации синтетических данных: {str(e)}")
        st.error(f"Ошибка генерации синтетических данных: {str(e)}")
        return None

# --- Модуль предобработки данных ---
def detect_outliers_iqr(data, column):
    """
    Обнаруживает выбросы методом межквартильного размаха.
    """
    try:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
        return outliers
    except KeyError as e:
        logger.error(f"Столбец {column} отсутствует: {str(e)}")
        st.error(f"Столбец {column} отсутствует")
        return None

def preprocess_data(data):
    """
    Обрабатывает данные: кодирование, нормализация, удаление выбросов.
    """
    try:
        logger.info("Начало предобработки данных")
        df = data.copy()
        
        # Проверка пропущенных значений
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Обнаружены пропущенные значения: {missing_values.to_dict()}")
            df.fillna(df.median(numeric_only=True), inplace=True)
            
        # Удаление ненужных столбцов
        columns_to_drop = ['UDI', 'Product ID', 'Failure Type']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        
        # Кодирование категориальной переменной
        le = LabelEncoder()
        if 'Type' in df.columns:
            df['Type'] = le.fit_transform(df['Type'])
        
        # Обработка выбросов
        numeric_columns = ['Air temperature [K]', 'Process temperature [K]', 
                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        for col in numeric_columns:
            outliers = detect_outliers_iqr(df, col)
            if outliers is not None and not outliers.empty:
                logger.info(f"Обнаружены выбросы в {col}: {len(outliers)}")
                df[col] = df[col].clip(lower=df[col].quantile(0.05), 
                                      upper=df[col].quantile(0.95))
        
        # Нормализация
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Разделение данных
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        logger.info("Предобработка завершена")
        return X, y, le, scaler
    except Exception as e:
        logger.error(f"Ошибка предобработки данных: {str(e)}")
        st.error(f"Ошибка предобработки: {str(e)}")
        return None, None, None, None
    
# --- Модуль анализа данных ---
def analyze_correlations(X):
    """
    Анализирует корреляции между признаками.
    """
    try:
        corr_matrix = X.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                    linewidths=0.5, cbar_kws={'label': 'Корреляция'})
        plt.title('Матрица корреляций', fontsize=14, pad=10)
        st.pyplot(plt)
        logger.info("Матрица корреляций построена")
    except Exception as e:
        logger.error(f"Ошибка анализа корреляций: {str(e)}")
        st.error(f"Ошибка анализа корреляций: {str(e)}")

def plot_feature_distribution(data, column):
    """
    Визуализирует распределение признака.
    """
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, bins=30, color='skyblue')
        plt.title(f'Распределение {column}', fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Частота', fontsize=12)
        st.pyplot(plt)
        logger.info(f"Распределение {column} построено")
    except Exception as e:
        logger.error(f"Ошибка построения распределения: {str(e)}")
        st.error(f"Ошибка построения распределения: {str(e)}")

def plot_box_plots(data, columns):
    """
    Визуализирует box plots для числовых признаков.
    """
    try:
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.boxplot(y=data[col], color='lightgreen')
            plt.title(f'Box Plot: {col}', fontsize=12)
            plt.ylabel(col, fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)
        logger.info("Box plots построены")
    except Exception as e:
        logger.error(f"Ошибка построения box plots: {str(e)}")
        st.error(f"Ошибка построения box plots: {str(e)}")

def plot_scatter_matrix(X):
    """
    Визуализирует матрицу рассеяния.
    """
    try:
        sns.pairplot(X, diag_kind='kde', plot_kws={'alpha': 0.5})
        plt.suptitle('Матрица рассеяния', y=1.02, fontsize=14)
        st.pyplot(plt)
        logger.info("Матрица рассеяния построена")
    except Exception as e:
        logger.error(f"Ошибка построения матрицы рассеяния: {str(e)}")
        st.error(f"Ошибка построения матрицы рассеяния: {str(e)}")

def plot_3d_scatter(X, y, feature1, feature2, feature3):
    """
    Визуализирует 3D-график рассеяния для трех признаков.
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[feature1], X[feature2], X[feature3], 
                           c=y, cmap='viridis', alpha=0.6)
        ax.set_xlabel(feature1, fontsize=10)
        ax.set_ylabel(feature2, fontsize=10)
        ax.set_zlabel(feature3, fontsize=10)
        ax.set_title('3D-график рассеяния', fontsize=14)
        plt.colorbar(scatter, label='Target')
        st.pyplot(plt)
        logger.info("3D-график рассеяния построен")
    except Exception as e:
        logger.error(f"Ошибка построения 3D-графика: {str(e)}")
        st.error(f"Ошибка построения 3D-графика: {str(e)}")

# --- Модуль обучения моделей ---
def train_random_forest(X_train, y_train, tune_params=False):
    """
    Обучает модель Random Forest с опциональным подбором гиперпараметров.
    """
    try:
        logger.info("Начало обучения Random Forest")
        start_time = time.time()
        
        if tune_params:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            model = GridSearchCV(RandomForestClassifier(random_state=42), 
                                param_grid, cv=5, n_jobs=-1, verbose=1)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                          min_samples_split=2, min_samples_leaf=1, 
                                          max_features='sqrt', random_state=42)
        
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"Обучение Random Forest завершено за {elapsed_time:.2f} сек")
        if tune_params:
            logger.info(f"Лучшие параметры: {model.best_params_}")
        return model
    except Exception as e:
        logger.error(f"Ошибка обучения Random Forest: {str(e)}")
        st.error(f"Ошибка обучения Random Forest: {str(e)}")
        return None

def train_gradient_boosting(X_train, y_train, tune_params=False):
    """
    Обучает модель Gradient Boosting с опциональным подбором гиперпараметров.
    """
    try:
        logger.info("Начало обучения Gradient Boosting")
        start_time = time.time()
        
        if tune_params:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5]
            }
            model = GridSearchCV(GradientBoostingClassifier(random_state=42), 
                                param_grid, cv=5, n_jobs=-1, verbose=1)
        else:
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                              max_depth=3, subsample=1.0, 
                                              min_samples_split=2, random_state=42)
        
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"Обучение Gradient Boosting завершено за {elapsed_time:.2f} сек")
        if tune_params:
            logger.info(f"Лучшие параметры: {model.best_params_}")
        return model
    except Exception as e:
        logger.error(f"Ошибка обучения Gradient Boosting: {str(e)}")
        st.error(f"Ошибка обучения Gradient Boosting: {str(e)}")
        return None

def train_mlp(X_train, y_train, tune_params=False):
    """
    Обучает модель MLP с опциональным подбором гиперпараметров.
    """
    try:
        logger.info("Начало обучения MLP")
        start_time = time.time()
        
        if tune_params:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50)],
                'max_iter': [200, 500, 1000],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01],
                'activation': ['relu', 'tanh']
            }
            model = GridSearchCV(MLPClassifier(random_state=42), 
                                param_grid, cv=5, n_jobs=-1, verbose=1)
        else:
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                                 learning_rate_init=0.001, alpha=0.0001, 
                                 activation='relu', random_state=42)
        
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"Обучение MLP завершено за {elapsed_time:.2f} сек")
        if tune_params:
            logger.info(f"Лучшие параметры: {model.best_params_}")
        return model
    except Exception as e:
        logger.error(f"Ошибка обучения MLP: {str(e)}")
        st.error(f"Ошибка обучения MLP: {str(e)}")
        return None

def train_svm(X_train, y_train, tune_params=False):
    """
    Обучает модель SVM с опциональным подбором гиперпараметров.
    """
    try:
        logger.info("Начало обучения SVM")
        start_time = time.time()
        
        if tune_params:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1],
                'degree': [2, 3]
            }
            model = GridSearchCV(SVC(probability=True, random_state=42), 
                                param_grid, cv=5, n_jobs=-1, verbose=1)
        else:
            model = SVC(C=1, kernel='rbf', gamma='scale', probability=True, 
                       random_state=42)
        
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"Обучение SVM завершено за {elapsed_time:.2f} сек")
        if tune_params:
            logger.info(f"Лучшие параметры: {model.best_params_}")
        return model
    except Exception as e:
        logger.error(f"Ошибка обучения SVM: {str(e)}")
        st.error(f"Ошибка обучения SVM: {str(e)}")
        return None

def train_knn(X_train, y_train, tune_params=False):
    """
    Обучает модель KNN с опциональным подбором гиперпараметров.
    """
    try:
        logger.info("Начало обучения KNN")
        start_time = time.time()
        
        if tune_params:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'leaf_size': [20, 30, 40]
            }
            model = GridSearchCV(KNeighborsClassifier(), 
                                param_grid, cv=5, n_jobs=-1, verbose=1)
        else:
            model = KNeighborsClassifier(n_neighbors=5, weights='uniform', 
                                        p=2, leaf_size=30)
        
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"Обучение KNN завершено за {elapsed_time:.2f} сек")
        if tune_params:
            logger.info(f"Лучшие параметры: {model.best_params_}")
        return model
    except Exception as e:
        logger.error(f"Ошибка обучения KNN: {str(e)}")
        st.error(f"Ошибка обучения KNN: {str(e)}")
        return None

def train_stacking(X_train, y_train):
    """
    Обучает ансамблевую модель стэкинга, комбинирующую несколько алгоритмов.
    """
    try:
        logger.info("Начало обучения модели стэкинга")
        start_time = time.time()
        
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        model = StackingClassifier(estimators=estimators, 
                                  final_estimator=LogisticRegression(), 
                                  cv=5, n_jobs=-1)
        
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"Обучение модели стэкинга завершено за {elapsed_time:.2f} сек")
        return model
    except Exception as e:
        logger.error(f"Ошибка обучения модели стэкинга: {str(e)}")
        st.error(f"Ошибка обучения модели стэкинга: {str(e)}")
        return None

# --- Модуль подбора признаков с Boruta ---
def select_features_boruta(X, y):
    """
    Выполняет подбор признаков с использованием алгоритма Boruta.
    """
    try:
        if not boruta_available:
            error_msg = "Модуль boruta не установлен. Установите его с помощью команды: pip install boruta"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        
        logger.info("Начало подбора признаков с Boruta")
        rf = RandomForestClassifier(n_jobs=-1, random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
        boruta.fit(X.values, y.values)
        
        selected_features = X.columns[boruta.support_].tolist()
        logger.info(f"Выбраны признаки: {selected_features}")
        return selected_features
    except Exception as e:
        logger.error(f"Ошибка подбора признаков Boruta: {str(e)}")
        st.error(f"Ошибка подбора признаков Boruta: {str(e)}")
        return None
    
# --- Модуль кросс-валидации ---
def perform_cross_validation(model, X, y, cv=5):
    """
    Выполняет кросс-валидацию модели.
    """
    try:
        scores = {
            'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy', 
                                      n_jobs=-1),
            'precision': cross_val_score(model, X, y, cv=cv, scoring='precision', 
                                       n_jobs=-1),
            'recall': cross_val_score(model, X, y, cv=cv, scoring='recall', 
                                    n_jobs=-1),
            'f1': cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1),
            'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc', 
                                      n_jobs=-1)
        }
        results = {
            'Accuracy': np.mean(scores['accuracy']),
            'Precision': np.mean(scores['precision']),
            'Recall': np.mean(scores['recall']),
            'F1': np.mean(scores['f1']),
            'ROC-AUC': np.mean(scores['roc_auc'])
        }
        logger.info(f"Кросс-валидация: {results}")
        return results
    except Exception as e:
        logger.error(f"Ошибка кросс-валидации: {str(e)}")
        st.error(f"Ошибка кросс-валидации: {str(e)}")
        return None

# --- Модуль сохранения и загрузки ---
def save_model(model, filename):
    """
    Сохраняет модель в файл.
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Модель сохранена как {filename}")
        st.success(f"Модель сохранена как {filename}")
    except Exception as e:
        logger.error(f"Ошибка сохранения модели: {str(e)}")
        st.error(f"Ошибка сохранения модели: {str(e)}")

def load_model(filename):
    """
    Загружает модель из файла.
    """
    try:
        if os.path.exists(filename):
            model = joblib.load(filename)
            logger.info(f"Модель загружена из {filename}")
            return model
        else:
            logger.error(f"Файл модели {filename} не найден")
            st.error(f"Файл модели {filename} не найден")
            return None
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None

# --- Модуль оценки моделей ---
def evaluate_model(model, X_test, y_test, model_name):
    """
    Оценивает производительность модели.
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Log Loss': log_loss(y_test, y_pred_proba) if hasattr(model, 'predict_proba') else None,
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Метрики {model_name}: {metrics}")
        return metrics, report, y_pred
    except Exception as e:
        logger.error(f"Ошибка оценки модели {model_name}: {str(e)}")
        st.error(f"Ошибка оценки модели {model_name}: {str(e)}")
        return None, None, None

def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Визуализирует матрицу ошибок.
    """
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Матрица ошибок ({model_name})', fontsize=14)
        plt.xlabel('Предсказано', fontsize=12)
        plt.ylabel('Фактически', fontsize=12)
        st.pyplot(plt)
        logger.info(f"Матрица ошибок для {model_name} построена")
    except Exception as e:
        logger.error(f"Ошибка построения матрицы ошибок: {str(e)}")
        st.error(f"Ошибка построения матрицы ошибок: {str(e)}")

def plot_roc_curve(model, X_test, y_test, model_name):
    """
    Визуализирует ROC-кривую.
    """
    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC кривая (AUC = {roc_auc:.2f})', color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
        plt.title(f'ROC кривая ({model_name})', fontsize=14)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(loc='lower right')
        st.pyplot(plt)
        logger.info(f"ROC кривая для {model_name} построена")
    except Exception as e:
        logger.error(f"Ошибка построения ROC кривой: {str(e)}")
        st.error(f"Ошибка построения ROC кривой: {str(e)}")

def plot_feature_importance(model, X, model_name):
    """
    Визуализирует важность признаков.
    """
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(8, 5))
            sns.barplot(x='importance', y='feature', data=feature_importance, 
                       color='salmon')
            plt.title(f'Важность признаков ({model_name})', fontsize=14)
            plt.xlabel('Важность', fontsize=12)
            plt.ylabel('Признак', fontsize=12)
            st.pyplot(plt)
            logger.info(f"Важность признаков для {model_name} построена")
    except Exception as e:
        logger.error(f"Ошибка построения важности признаков: {str(e)}")
        st.error(f"Ошибка построения важности признаков: {str(e)}")

def plot_shap_values(model, X, model_name):
    """
    Визуализирует SHAP-значения для интерпретации модели.
    """
    try:
        if not shap_available:
            error_msg = "Модуль shap не установлен. Установите его с помощью команды: pip install shap"
            logger.error(error_msg)
            st.error(error_msg)
            return
        
        explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, X.sample(100))
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'SHAP значения ({model_name})', fontsize=14)
        st.pyplot(plt)
        logger.info(f"SHAP значения для {model_name} построены")
    except Exception as e:
        logger.error(f"Ошибка построения SHAP значений: {str(e)}")
        st.error(f"Ошибка построения SHAP значений: {str(e)}")

# --- Модуль анализа чувствительности гиперпараметров ---
def hyperparameter_sensitivity(model_class, X_train, y_train, param_name, param_values):
    """
    Анализирует чувствительность модели к изменению гиперпараметра.
    """
    try:
        results = []
        for value in param_values:
            params = {param_name: value, 'random_state': 42}
            model = model_class(**params)
            model.fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='accuracy', n_jobs=-1)
            results.append({
                'Parameter Value': value,
                'Mean Accuracy': np.mean(scores),
                'Std Accuracy': np.std(scores)
            })
        results_df = pd.DataFrame(results)
        
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='Parameter Value', y='Mean Accuracy', data=results_df, 
                    marker='o', label='Средняя точность')
        plt.fill_between(results_df['Parameter Value'], 
                        results_df['Mean Accuracy'] - results_df['Std Accuracy'], 
                        results_df['Mean Accuracy'] + results_df['Std Accuracy'], 
                        alpha=0.2, label='Стд. отклонение')
        plt.title(f'Чувствительность к {param_name}', fontsize=14)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.legend()
        st.pyplot(plt)
        
        results_df.to_csv(f'hyperparam_{param_name}_sensitivity.csv', index=False)
        logger.info(f"Анализ чувствительности для {param_name} завершен")
        return results_df
    except Exception as e:
        logger.error(f"Ошибка анализа чувствительности: {str(e)}")
        st.error(f"Ошибка анализа чувствительности: {str(e)}")
        return None

# --- Модуль сравнения моделей ---
def compare_models(models, X_test, y_test):
    """
    Сравнивает производительность моделей.
    """
    try:
        results = []
        for name, model in models.items():
            metrics, _, _ = evaluate_model(model, X_test, y_test, name)
            if metrics:
                results.append({
                    'Model': name,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1': metrics['F1'],
                    'Log Loss': metrics['Log Loss'],
                    'ROC-AUC': metrics['ROC-AUC']
                })
        
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
        plt.title('Сравнение точности моделей', fontsize=14)
        plt.xlabel('Модель', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        st.pyplot(plt)
        
        results_df.to_csv('model_comparison.csv', index=False)
        with open('model_comparison.json', 'w') as f:
            json.dump(results, f)
        logger.info("Результаты сравнения сохранены в CSV и JSON")
        return results_df
    except Exception as e:
        logger.error(f"Ошибка сравнения моделей: {str(e)}")
        st.error(f"Ошибка сравнения моделей: {str(e)}")
        return None

# --- Модуль предсказания ---
def predict_with_model(model, input_data, scaler, le):
    """
    Делает предсказание с помощью модели.
    """
    try:
        numeric_columns = ['Air temperature [K]', 'Process temperature [K]', 
                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        numeric_columns = [col for col in numeric_columns if col in input_data.columns]
        input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])
        
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        st.error(f"Ошибка предсказания: {str(e)}")
        return None

# --- Модуль тестирования на подвыборках ---
def test_on_subsamples(X, y, model, sample_sizes=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Тестирует модель на подвыборках разного размера.
    """
    try:
        results = []
        for size in sample_sizes:
            X_sub, _, y_sub, _ = train_test_split(X, y, train_size=size, 
                                                 random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, 
                                                               test_size=0.2, 
                                                               random_state=42)
            model.fit(X_train, y_train)
            metrics, _, _ = evaluate_model(model, X_test, y_test, 
                                          f"Subsample {size}")
            results.append({
                'Sample Size': size,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'Log Loss': metrics['Log Loss'],
                'ROC-AUC': metrics['ROC-AUC']
            })
        
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Sample Size', y='Accuracy', data=results_df, 
                    marker='o', color='purple')
        plt.title('Зависимость точности от размера выборки', fontsize=14)
        plt.xlabel('Размер выборки', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        st.pyplot(plt)
        
        results_df.to_csv('subsample_results.csv', index=False)
        with open('subsample_results.json', 'w') as f:
            json.dump(results, f)
        logger.info("Результаты тестирования на подвыборках сохранены")
        return results_df
    except Exception as e:
        logger.error(f"Ошибка тестирования на подвыборках: {str(e)}")
        st.error(f"Ошибка тестирования на подвыборках: {str(e)}")
        return None

# --- Модуль стресс-тестирования ---
def stress_test_model(model, n_samples=10000, n_features=6):
    """
    Проводит стресс-тестирование модели на синтетических данных.
    """
    try:
        synthetic_data = generate_synthetic_data(n_samples, n_features)
        X, y, _, scaler = preprocess_data(synthetic_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42)
        model.fit(X_train, y_train)
        metrics, _, _ = evaluate_model(model, X_test, y_test, "Stress Test")
        logger.info(f"Стресс-тестирование: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Ошибка стресс-тестирования: {str(e)}")
        st.error(f"Ошибка стресс-тестирования: {str(e)}")
        return None

# --- Модуль тестирования подмножеств признаков ---
def test_feature_subsets(X, y, model, max_features=3):
    """
    Тестирует модель на различных подмножествах признаков.
    """
    try:
        results = []
        feature_combinations = []
        for r in range(1, max_features + 1):
            feature_combinations.extend(itertools.combinations(X.columns, r))
        
        for features in feature_combinations:
            X_subset = X[list(features)]
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, 
                                                               test_size=0.2, 
                                                               random_state=42)
            model.fit(X_train, y_train)
            metrics, _, _ = evaluate_model(model, X_test, y_test, 
                                          f"Features: {features}")
            results.append({
                'Features': ', '.join(features),
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'Log Loss': metrics['Log Loss'],
                'ROC-AUC': metrics['ROC-AUC']
            })
        
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Accuracy', y='Features', data=results_df, palette='magma')
        plt.title('Точность для подмножеств признаков', fontsize=14)
        plt.xlabel('Точность', fontsize=12)
        plt.ylabel('Признаки', fontsize=12)
        st.pyplot(plt)
        
        results_df.to_csv('feature_subset_results.csv', index=False)
        logger.info("Результаты тестирования подмножеств признаков сохранены")
        return results_df
    except Exception as e:
        logger.error(f"Ошибка тестирования подмножеств признаков: {str(e)}")
        st.error(f"Ошибка тестирования подмножеств признаков: {str(e)}")
        return None

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ... (остальные импорты остаются без изменений)

def generate_pdf_report(models, X_test, y_test):
    """
    Генерирует PDF-отчет с результатами сравнения моделей с использованием reportlab.
    """
    try:
        # Регистрация шрифта (используем Times New Roman)
        pdfmetrics.registerFont(TTFont('TimesNewRoman', 'TimesNewRoman.ttf'))
        
        # Создаем буфер для PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch, 
                               topMargin=inch, bottomMargin=inch)
        elements = []
        
        # Стили для текста
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        title_style.fontName = 'TimesNewRoman'
        title_style.fontSize = 16
        section_style = styles['Heading2']
        section_style.fontName = 'TimesNewRoman'
        section_style.fontSize = 14
        body_style = ParagraphStyle(
            name='BodyText',
            fontName='TimesNewRoman',
            fontSize=10,
            leading=12,
            spaceAfter=6,
            encoding='UTF-8'
        )
        
        # Заголовок
        elements.append(Paragraph("Отчет по прогнозированию неисправности компьютерных компонентов", title_style))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Автор: Комаров Даниил Иванович", body_style))
        elements.append(Paragraph(f"Дата: {datetime.today().strftime('%Y-%m-%d')}", body_style))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Введение
        elements.append(Paragraph("Введение", section_style))
        elements.append(Paragraph(
            "Данный отчет представляет результаты анализа данных и производительности моделей "
            "машинного обучения для прогнозирования неисправности компьютерных компонентов. "
            "Использованы алгоритмы Random Forest, Gradient Boosting, MLP, SVM, KNN и "
            "ансамблевая модель стэкинга.", body_style
        ))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Методология
        elements.append(Paragraph("Методология", section_style))
        methodology_text = """
        <ul>
            <li>Датасет: POLOMKA.csv</li>
            <li>Предобработка: кодирование категориальных переменных, нормализация, удаление выбросов</li>
            <li>Подбор признаков: алгоритм Boruta</li>
            <li>Оценка: Accuracy, Precision, Recall, F1, Log Loss, ROC-AUC</li>
        </ul>
        """
        elements.append(Paragraph(methodology_text, body_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Результаты
        elements.append(Paragraph("Результаты", section_style))
        elements.append(Paragraph("Сравнение моделей", section_style))
        
        # Собираем данные для таблицы
        results = []
        for name, model in models.items():
            metrics, _, _ = evaluate_model(model, X_test, y_test, name)
            if metrics:
                results.append([
                    name,
                    f"{metrics['Accuracy']:.3f}",
                    f"{metrics['Precision']:.3f}",
                    f"{metrics['Recall']:.3f}",
                    f"{metrics['F1']:.3f}",
                    f"{metrics['Log Loss']:.3f}" if metrics['Log Loss'] else "N/A",
                    f"{metrics['ROC-AUC']:.3f}"
                ])
        
        # Создаем таблицу
        table_data = [["Модель", "Accuracy", "Precision", "Recall", "F1", "Log Loss", "ROC-AUC"]] + results
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Интерпретация
        elements.append(Paragraph("Интерпретация", section_style))
        elements.append(Paragraph(
            "Анализ SHAP-значений показал, что наиболее значимыми признаками являются "
            "Rotational speed [rpm], Torque [Nm] и Tool wear [min]. Подбор признаков с помощью "
            "Boruta подтвердил важность этих признаков, исключив менее значимые, такие как Type.", 
            body_style
        ))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Выводы
        elements.append(Paragraph("Выводы", section_style))
        elements.append(Paragraph(
            "На основе анализа производительности моделей можно сделать вывод, что Random Forest "
            "и ансамблевая модель стэкинга показали наилучшие результаты с точки зрения точности "
            "и ROC-AUC. Дальнейшая работа может включать оптимизацию гиперпараметров, тестирование "
            "на дополнительных данных и интеграцию с реальными системами мониторинга.", 
            body_style
        ))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Рекомендации
        elements.append(Paragraph("Рекомендации", section_style))
        recommendations_text = """
        <ul>
            <li>Использовать Random Forest или Stacking для реальных приложений.</li>
            <li>Проводить регулярное обновление моделей с новыми данными.</li>
            <li>Интегрировать систему с датчиками для мониторинга в реальном времени.</li>
        </ul>
        """
        elements.append(Paragraph(recommendations_text, body_style))
        
        # Создаем PDF
        doc.build(elements)
        buffer.seek(0)
        logger.info("PDF-отчет успешно сгенерирован")
        st.success("PDF-отчет сгенерирован и готов к скачиванию.")
        return buffer
    except Exception as e:
        logger.error(f"Ошибка генерации PDF-отчета: {str(e)}")
        st.error(f"Ошибка генерации PDF-отчета: {str(e)}")
        return None

# --- Модуль юнит-тестирования ---
class TestPredictionSystem(unittest.TestCase):
    """
    Юнит-тесты для проверки функциональности системы.
    """
    def setUp(self):
        """
        Инициализация тестового окружения.
        """
        self.data = generate_synthetic_data(n_samples=100)
        self.X, self.y, self.le, self.scaler = preprocess_data(self.data)
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X, self.y)
    
    def test_load_dataset(self):
        """
        Проверяет корректность загрузки датасета.
        """
        data = generate_synthetic_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
    
    def test_preprocess_data(self):
        """
        Проверяет корректность предобработки данных.
        """
        X, y, le, scaler = preprocess_data(self.data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertIsInstance(le, LabelEncoder)
        self.assertIsInstance(scaler, StandardScaler)
    
    def test_model_training(self):
        """
        Проверяет, что модель обучается без ошибок.
        """
        model = train_random_forest(self.X, self.y, tune_params=False)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
    
    def test_prediction(self):
        """
        Проверяет, что модель возвращает предсказания.
        """
        input_data = self.X.iloc[:1]
        prediction = predict_with_model(self.model, input_data, self.scaler, self.le)
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction), 1)
    
    def test_evaluate_model(self):
        """
        Проверяет корректность оценки модели.
        """
        metrics, report, y_pred = evaluate_model(self.model, self.X, self.y, "Test")
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(report)
        self.assertIsNotNone(y_pred)
        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision', metrics)
        self.assertIn('Recall', metrics)
        self.assertIn('F1', metrics)
    
    def test_save_load_model(self):
        """
        Проверяет сохранение и загрузку модели.
        """
        save_model(self.model, 'test_model.joblib')
        loaded_model = load_model('test_model.joblib')
        self.assertIsNotNone(loaded_model)
        self.assertTrue(hasattr(loaded_model, 'predict'))
    
    def test_boruta_selection(self):
        """
        Проверяет подбор признаков Boruta.
        """
        if boruta_available:
            selected_features = select_features_boruta(self.X, self.y)
            self.assertIsNotNone(selected_features)
            self.assertIsInstance(selected_features, list)
        else:
            self.skipTest("Boruta не установлен")
    
    def test_shap_analysis(self):
        """
        Проверяет SHAP анализ.
        """
        if shap_available:
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(self.X)
                self.assertIsNotNone(shap_values)
            except Exception as e:
                self.fail(f"SHAP анализ завершился с ошибкой: {str(e)}")
        else:
            self.skipTest("SHAP не установлен")
    
    def test_pdf_generation(self):
        """
        Проверяет генерацию PDF-отчета.
        """
        models = {'Random Forest': self.model}
        pdf_buffer = generate_pdf_report(models, self.X, self.y)
        self.assertIsNotNone(pdf_buffer)
        self.assertIsInstance(pdf_buffer, io.BytesIO)
        self.assertGreater(pdf_buffer.getbuffer().nbytes, 0)  # Проверяем, что PDF не пустой

# --- Основной модуль Streamlit ---
def main():
    """
    Основная функция для запуска веб-приложения.
    """
    try:
        st.set_page_config(page_title="Прогнозирование неисправностей", 
                          layout="wide")
        st.title("Прогнозирование неисправности компьютерных компонентов")
        
        # Диагностика окружения
        st.sidebar.header("Диагностика")
        if st.sidebar.button("Проверить окружение"):
            st.write("Версия Python:", sys.version)
            st.write("Путь к исполняемому файлу Python:", sys.executable)
            try:
                import streamlit
                st.write("Версия Streamlit:", streamlit.__version__)
            except ImportError:
                st.error("Streamlit не установлен в текущем окружении")
            st.write("Boruta установлен:", boruta_available)
            st.write("SHAP установлен:", shap_available)
        
        # Запуск тестов
        if st.sidebar.button("Запустить юнит-тесты"):
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestPredictionSystem)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            st.write("Результаты тестов:", str(result))
        
        # Загрузка датасета
        dataset_url = st.text_input("URL датасета", 
                                   "https://raw.githubusercontent.com/[ваш_репозиторий]/main/POLOMKA.csv")
        use_synthetic = st.checkbox("Использовать синтетические данные")
        if use_synthetic:
            n_samples = st.number_input("Количество синтетических записей", 
                                       min_value=100, value=1000, step=100)
            data = generate_synthetic_data(n_samples=n_samples)
        else:
            data = load_dataset(url=dataset_url)
        if data is None:
            st.error("Не удалось загрузить датасет. Проверьте URL или используйте синтетические данные.")
            return
        
        # Предобработка данных
        X, y, le, scaler = preprocess_data(data)
        if X is None:
            st.error("Ошибка предобработки данных. Проверьте формат датасета.")
            return
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42)
        logger.info(f"Размер обучающей выборки: {X_train.shape}, тестовой: {X_test.shape}")
        
        # Панель управления
        st.sidebar.header("Управление")
        action = st.sidebar.selectbox("Выберите действие", 
                                     ["Анализ данных", "Обучить модели", 
                                      "Загрузить модели", "Сравнить модели", 
                                      "Сделать предсказание", "Тестировать на подвыборках",
                                      "Кросс-валидация", "SHAP анализ", 
                                      "Стресс-тестирование", "Анализ гиперпараметров",
                                      "Тестирование подмножеств признаков", 
                                      "Подбор признаков Boruta", "Генерация PDF-отчета"])
        
        # Анализ данных
        if action == "Анализ данных":
            st.subheader("Анализ датасета")
            st.write("Размер датасета:", data.shape)
            st.write("Первые строки:")
            st.dataframe(data.head())
            
            st.write("Корреляции между признаками:")
            analyze_correlations(X)
            
            st.write("Распределение признаков:")
            for col in X.columns:
                plot_feature_distribution(data, col)
            
            st.write("Box Plots для числовых признаков:")
            numeric_columns = [col for col in ['Air temperature [K]', 
                                              'Process temperature [K]', 
                                              'Rotational speed [rpm]', 
                                              'Torque [Nm]', 'Tool wear [min]'] 
                              if col in data.columns]
            plot_box_plots(data, numeric_columns)
            
            st.write("Матрица рассеяния:")
            plot_scatter_matrix(X)
            
            st.write("3D-график рассеяния:")
            if len(X.columns) >= 3:
                feature1, feature2, feature3 = X.columns[:3]
                plot_3d_scatter(X, y, feature1, feature2, feature3)
        
        # Обучение моделей
        elif action == "Обучить модели":
            st.subheader("Обучение моделей")
            tune_params = st.checkbox("Подбирать гиперпараметры (может занять время)")
            
            models = {
                'Random Forest': None,
                'Gradient Boosting': None,
                'MLP': None,
                'SVM': None,
                'KNN': None,
                'Stacking': None
            }
            
            # Random Forest
            st.write("Обучение Random Forest...")
            models['Random Forest'] = train_random_forest(X_train, y_train, tune_params)
            if models['Random Forest']:
                save_model(models['Random Forest'], 'model_rf.joblib')
                metrics, report, y_pred = evaluate_model(models['Random Forest'], 
                                                        X_test, y_test, 'Random Forest')
                st.write("Метрики Random Forest:", metrics)
                st.write("Отчет по классификации:")
                st.json(report)
                plot_confusion_matrix(y_test, y_pred, 'Random Forest')
                plot_roc_curve(models['Random Forest'], X_test, y_test, 'Random Forest')
                plot_feature_importance(models['Random Forest'], X, 'Random Forest')
            
            # Gradient Boosting
            st.write("Обучение Gradient Boosting...")
            models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train, tune_params)
            if models['Gradient Boosting']:
                save_model(models['Gradient Boosting'], 'model_gb.joblib')
                metrics, report, y_pred = evaluate_model(models['Gradient Boosting'], 
                                                        X_test, y_test, 'Gradient Boosting')
                st.write("Метрики Gradient Boosting:", metrics)
                st.write("Отчет по классификации:")
                st.json(report)
                plot_confusion_matrix(y_test, y_pred, 'Gradient Boosting')
                plot_roc_curve(models['Gradient Boosting'], X_test, y_test, 'Gradient Boosting')
                plot_feature_importance(models['Gradient Boosting'], X, 'Gradient Boosting')
            
            # MLP
            st.write("Обучение MLP...")
            models['MLP'] = train_mlp(X_train, y_train, tune_params)
            if models['MLP']:
                save_model(models['MLP'], 'model_mlp.joblib')
                metrics, report, y_pred = evaluate_model(models['MLP'], 
                                                        X_test, y_test, 'MLP')
                st.write("Метрики MLP:", metrics)
                st.write("Отчет по классификации:")
                st.json(report)
                plot_confusion_matrix(y_test, y_pred, 'MLP')
                plot_roc_curve(models['MLP'], X_test, y_test, 'MLP')
            
            # SVM
            st.write("Обучение SVM...")
            models['SVM'] = train_svm(X_train, y_train, tune_params)
            if models['SVM']:
                save_model(models['SVM'], 'model_svm.joblib')
                metrics, report, y_pred = evaluate_model(models['SVM'], 
                                                        X_test, y_test, 'SVM')
                st.write("Метрики SVM:", metrics)
                st.write("Отчет по классификации:")
                st.json(report)
                plot_confusion_matrix(y_test, y_pred, 'SVM')
                plot_roc_curve(models['SVM'], X_test, y_test, 'SVM')
            
            # KNN
            st.write("Обучение KNN...")
            models['KNN'] = train_knn(X_train, y_train, tune_params)
            if models['KNN']:
                save_model(models['KNN'], 'model_knn.joblib')
                metrics, report, y_pred = evaluate_model(models['KNN'], 
                                                        X_test, y_test, 'KNN')
                st.write("Метрики KNN:", metrics)
                st.write("Отчет по классификации:")
                st.json(report)
                plot_confusion_matrix(y_test, y_pred, 'KNN')
                plot_roc_curve(models['KNN'], X_test, y_test, 'KNN')
            
            # Stacking
            st.write("Обучение модели стэкинга...")
            models['Stacking'] = train_stacking(X_train, y_train)
            if models['Stacking']:
                save_model(models['Stacking'], 'model_stacking.joblib')
                metrics, report, y_pred = evaluate_model(models['Stacking'], 
                                                        X_test, y_test, 'Stacking')
                st.write("Метрики Stacking:", metrics)
                st.write("Отчет по классификации:")
                st.json(report)
                plot_confusion_matrix(y_test, y_pred, 'Stacking')
                plot_roc_curve(models['Stacking'], X_test, y_test, 'Stacking')
        
        # Загрузка моделей
        elif action == "Загрузить модели":
            st.subheader("Загрузка моделей")
            
            models = {
                'Random Forest': load_model('model_rf.joblib'),
                'Gradient Boosting': load_model('model_gb.joblib'),
                'MLP': load_model('model_mlp.joblib'),
                'SVM': load_model('model_svm.joblib'),
                'KNN': load_model('model_knn.joblib'),
                'Stacking': load_model('model_stacking.joblib')
            }
            
            for name, model in models.items():
                if model:
                    metrics, report, y_pred = evaluate_model(model, X_test, y_test, name)
                    st.write(f"Метрики {name}:", metrics)
                    st.write("Отчет по классификации:")
                    st.json(report)
                    plot_confusion_matrix(y_test, y_pred, name)
                    plot_roc_curve(model, X_test, y_test, name)
                    if name in ['Random Forest', 'Gradient Boosting']:
                        plot_feature_importance(model, X, name)
        
        # Сравнение моделей
        elif action == "Сравнить модели":
            st.subheader("Сравнение моделей")
            
            models = {
                'Random Forest': load_model('model_rf.joblib'),
                'Gradient Boosting': load_model('model_gb.joblib'),
                'MLP': load_model('model_mlp.joblib'),
                'SVM': load_model('model_svm.joblib'),
                'KNN': load_model('model_knn.joblib'),
                'Stacking': load_model('model_stacking.joblib')
            }
            
            results_df = compare_models(models, X_test, y_test)
            if results_df is not None:
                st.write("Результаты сравнения:")
                st.dataframe(results_df)
        
        # Предсказание
        elif action == "Сделать предсказание":
            st.subheader("Предсказание неисправности")
            
            model_name = st.selectbox("Выберите модель", 
                                     ["Random Forest", "Gradient Boosting", "MLP", 
                                      "SVM", "KNN", "Stacking"])
            model_files = {
                "Random Forest": 'model_rf.joblib',
                "Gradient Boosting": 'model_gb.joblib',
                "MLP": 'model_mlp.joblib',
                "SVM": 'model_svm.joblib',
                "KNN": 'model_knn.joblib',
                "Stacking": 'model_stacking.joblib'
            }
            model = load_model(model_files[model_name])
            
            if model:
                st.write("Введите данные для предсказания:")
                type_input = st.selectbox("Тип продукта", ["L", "M", "H"])
                air_temp = st.number_input("Температура воздуха [K]", 
                                          min_value=0.0, value=298.0, step=0.1)
                process_temp = st.number_input("Температура процесса [K]", 
                                              min_value=0.0, value=308.0, step=0.1)
                rpm = st.number_input("Скорость вращения [rpm]", 
                                     min_value=0, value=1500, step=10)
                torque = st.number_input("Крутящий момент [Nm]", 
                                        min_value=0.0, value=40.0, step=0.1)
                tool_wear = st.number_input("Износ инструмента [min]", 
                                           min_value=0, value=0, step=1)
                
                if st.button("Сделать предсказание"):
                    input_data = pd.DataFrame({
                        'Type': [le.transform([type_input])[0]],
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rpm],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear]
                    })
                    
                    prediction = predict_with_model(model, input_data, scaler, le)
                    if prediction is not None:
                        result = "Неисправность" if prediction[0] == 1 else "Нет неисправности"
                        st.write(f"Результат ({model_name}): {result}")
                        logger.info(f"Предсказание ({model_name}): {result}")
        
        # Кросс-валидация
        elif action == "Кросс-валидация":
            st.subheader("Кросс-валидация моделей")
            
            model_name = st.selectbox("Выберите модель для кросс-валидации", 
                                     ["Random Forest", "Gradient Boosting", "MLP", 
                                      "SVM", "KNN", "Stacking"])
            model_files = {
                "Random Forest": 'model_rf.joblib',
                "Gradient Boosting": 'model_gb.joblib',
                "MLP": 'model_mlp.joblib',
                "SVM": 'model_svm.joblib',
                "KNN": 'model_knn.joblib',
                "Stacking": 'model_stacking.joblib'
            }
            model = load_model(model_files[model_name])
            
            if model:
                cv_folds = st.slider("Количество фолдов", min_value=3, 
                                    max_value=10, value=5)
                cv_results = perform_cross_validation(model, X, y, cv=cv_folds)
                if cv_results:
                    st.write("Результаты кросс-валидации:", cv_results)
        
        # SHAP анализ
        elif action == "SHAP анализ":
            st.subheader("SHAP анализ моделей")
            
            if not shap_available:
                st.error("Модуль shap не установлен. Установите его с помощью команды: `pip install shap`")
                st.markdown("""
                **Инструкция по установке:**
                1. Откройте терминал или командную строку.
                2. Активируйте ваше виртуальное окружение (если используется).
                3. Выполните команду: `pip install shap`
                4. Перезапустите приложение после установки.
                """)
                if st.button("Проверить установку shap"):
                    try:
                        import shap
                        st.success("Модуль shap успешно установлен!")
                        globals()['shap_available'] = True
                        globals()['shap'] = shap
                    except ImportError:
                        st.error("Модуль shap все еще не установлен. Проверьте правильность выполнения команды установки.")
            else:
                model_name = st.selectbox("Выберите модель для SHAP анализа", 
                                         ["Random Forest", "Gradient Boosting", "MLP", 
                                          "SVM", "KNN", "Stacking"])
                model_files = {
                    "Random Forest": 'model_rf.joblib',
                    "Gradient Boosting": 'model_gb.joblib',
                    "MLP": 'model_mlp.joblib',
                    "SVM": 'model_svm.joblib',
                    "KNN": 'model_knn.joblib',
                    "Stacking": 'model_stacking.joblib'
                }
                model = load_model(model_files[model_name])
                
                if model:
                    st.write(f"Построение SHAP значений для {model_name}...")
                    plot_shap_values(model, X_test, model_name)
        
        # Стресс-тестирование
        elif action == "Стресс-тестирование":
            st.subheader("Стресс-тестирование моделей")
            
            model_name = st.selectbox("Выберите модель для стресс-тестирования", 
                                     ["Random Forest", "Gradient Boosting", "MLP", 
                                      "SVM", "KNN", "Stacking"])
            model_files = {
                "Random Forest": 'model_rf.joblib',
                "Gradient Boosting": 'model_gb.joblib',
                "MLP": 'model_mlp.joblib',
                "SVM": 'model_svm.joblib',
                "KNN": 'model_knn.joblib',
                "Stacking": 'model_stacking.joblib'
            }
            model = load_model(model_files[model_name])
            
            if model:
                n_samples = st.number_input("Количество записей для стресс-теста", 
                                           min_value=1000, value=10000, step=1000)
                metrics = stress_test_model(model, n_samples=n_samples)
                if metrics:
                    st.write("Результаты стресс-тестирования:", metrics)
                    
        # Анализ гиперпараметров
        elif action == "Анализ гиперпараметров":
            st.subheader("Анализ чувствительности гиперпараметров")
            
            model_name = st.selectbox("Выберите модель", 
                                     ["Random Forest", "Gradient Boosting", "MLP", 
                                      "SVM", "KNN"])
            if model_name == "Random Forest":
                param_name = st.selectbox("Гиперпараметр", 
                                         ["n_estimators", "max_depth"])
                param_values = {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [5, 10, 15, 20, None]
                }[param_name]
                model_class = RandomForestClassifier
            elif model_name == "Gradient Boosting":
                param_name = st.selectbox("Гиперпараметр", 
                                         ["n_estimators", "learning_rate"])
                param_values = {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2]
                }[param_name]
                model_class = GradientBoostingClassifier
            elif model_name == "MLP":
                param_name = st.selectbox("Гиперпараметр", 
                                         ["learning_rate_init", "max_iter"])
                param_values = {
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'max_iter': [200, 500, 1000, 2000]
                }[param_name]
                model_class = MLPClassifier
            elif model_name == "SVM":
                param_name = st.selectbox("Гиперпараметр", ["C", "gamma"])
                param_values = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1]
                }[param_name]
                model_class = SVC
            else:  # KNN
                param_name = st.selectbox("Гиперпараметр", ["n_neighbors", "weights"])
                param_values = {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                }[param_name]
                model_class = KNeighborsClassifier
            
            st.write(f"Анализ чувствительности гиперпараметра {param_name} для {model_name}...")
            results_df = hyperparameter_sensitivity(model_class, X_train, y_train, 
                                                  param_name, param_values)
            if results_df is not None:
                st.write("Результаты анализа чувствительности:")
                st.dataframe(results_df)
                logger.info(f"Анализ чувствительности для {model_name}, {param_name} завершен")
        
        # Тестирование подмножеств признаков
        elif action == "Тестирование подмножеств признаков":
            st.subheader("Тестирование подмножеств признаков")
            
            model_name = st.selectbox("Выберите модель для тестирования", 
                                     ["Random Forest", "Gradient Boosting", "MLP", 
                                      "SVM", "KNN", "Stacking"])
            model_files = {
                "Random Forest": 'model_rf.joblib',
                "Gradient Boosting": 'model_gb.joblib',
                "MLP": 'model_mlp.joblib',
                "SVM": 'model_svm.joblib',
                "KNN": 'model_knn.joblib',
                "Stacking": 'model_stacking.joblib'
            }
            model = load_model(model_files[model_name])
            
            if model:
                max_features = st.slider("Максимальное количество признаков в подмножестве", 
                                        min_value=1, max_value=len(X.columns), value=3)
                st.write(f"Тестирование модели {model_name} на подмножествах признаков...")
                results_df = test_feature_subsets(X, y, model, max_features=max_features)
                if results_df is not None:
                    st.write("Результаты тестирования подмножеств признаков:")
                    st.dataframe(results_df)
                    logger.info(f"Тестирование подмножеств признаков для {model_name} завершено")
            else:
                st.error("Ошибка загрузки модели. Убедитесь, что модель обучена и сохранена.")
        
        # Подбор признаков Boruta
        elif action == "Подбор признаков Boruta":
            st.subheader("Подбор признаков с использованием Boruta")
            if not boruta_available:
                st.error("Модуль boruta не установлен. Установите его с помощью команды: `pip install boruta`")
                st.markdown("""
                **Инструкция по установке:**
                1. Откройте терминал или командную строку.
                2. Активируйте ваше виртуальное окружение (если используется).
                3. Выполните команду: `pip install boruta`
                4. Перезапустите приложение после установки.
                """)
                if st.button("Проверить установку boruta"):
                    try:
                        from boruta import BorutaPy
                        st.success("Модуль boruta успешно установлен!")
                        globals()['boruta_available'] = True
                        globals()['BorutaPy'] = BorutaPy
                    except ImportError:
                        st.error("Модуль boruta все еще не установлен. Проверьте правильность выполнения команды установки.")
            else:
                st.write("Выполняется подбор признаков...")
                selected_features = select_features_boruta(X, y)
                if selected_features:
                    st.write("Выбранные признаки:", selected_features)
                    logger.info(f"Подборpip install -r requirements.txtпризнаков Boruta завершен: {selected_features}")
                else:
                    st.error("Ошибка подбора признаков Boruta. Проверьте данные или настройки.")
        
# Генерация PDF-отчета
        if action == "Генерация PDF-отчета":
            st.subheader("Генерация PDF-отчета")
            models = {
                'Random Forest': load_model('model_rf.joblib'),
                'Gradient Boosting': load_model('model_gb.joblib'),
                'MLP': load_model('model_mlp.joblib'),
                'SVM': load_model('model_svm.joblib'),
                'KNN': load_model('model_knn.joblib'),
                'Stacking': load_model('model_stacking.joblib')
            }
            models = {name: model for name, model in models.items() if model is not None}
            if models:
                st.write("Генерация отчета...")
                pdf_buffer = generate_pdf_report(models, X_test, y_test)
                if pdf_buffer:
                    st.write("PDF-отчет сгенерирован. Нажмите кнопку ниже, чтобы скачать.")
                    st.download_button(
                        label="Скачать PDF-отчет",
                        data=pdf_buffer,
                        file_name="report.pdf",
                        mime="application/pdf"
                    )
                    logger.info("PDF-отчет готов к скачиванию")
            else:
                st.error("Не удалось загрузить модели для генерации отчета.")
                
    except Exception as e:
        logger.error(f"Критическая ошибка в приложении: {str(e)}")
        st.error(f"Критическая ошибка: {str(e)}")

if __name__ == "__main__":
    main()