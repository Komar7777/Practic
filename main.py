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
