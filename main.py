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
