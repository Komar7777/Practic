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