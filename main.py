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
