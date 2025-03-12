#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для предобработки данных для нейросетевых моделей.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Класс для подготовки данных для обучения нейросетевых моделей.
    """
    
    def __init__(self, sequence_length: int = 60, forecast_horizon: int = 1, 
                 test_size: float = 0.2, validation_size: float = 0.1,
                 scaling_method: str = 'minmax'):
        """
        Инициализация препроцессора данных.
        
        Args:
            sequence_length (int): Длина последовательности для входных данных (окно)
            forecast_horizon (int): Горизонт прогнозирования (на сколько шагов вперед)
            test_size (float): Доля данных для тестирования (от 0 до 1)
            validation_size (float): Доля данных для валидации (от 0 до 1)
            scaling_method (str): Метод масштабирования данных ('minmax' или 'standard')
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_columns = []
        
        logger.info(f"Инициализирован препроцессор данных с параметрами: "
                   f"sequence_length={sequence_length}, forecast_horizon={forecast_horizon}, "
                   f"test_size={test_size}, validation_size={validation_size}, "
                   f"scaling_method={scaling_method}")
    
    def create_scaler(self, scaling_method: str = None) -> None:
        """
        Создание скейлера для нормализации данных.
        
        Args:
            scaling_method (str, optional): Метод масштабирования ('minmax' или 'standard')
        """
        if scaling_method is None:
            scaling_method = self.scaling_method
        
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            logger.info("Создан MinMaxScaler")
        elif scaling_method == 'standard':
            self.scaler = StandardScaler()
            logger.info("Создан StandardScaler")
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            logger.warning(f"Неизвестный метод масштабирования '{scaling_method}'. Используется MinMaxScaler")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close', 
                    feature_columns: List[str] = None, label_type: str = 'regression',
                    n_classes: int = 3) -> Dict[str, Any]:
        """
        Подготовка данных для обучения и тестирования нейросетевых моделей.
        
        Args:
            df (pd.DataFrame): DataFrame с данными
            target_column (str): Целевая колонка для прогнозирования
            feature_columns (List[str], optional): Список колонок-признаков. 
                Если None, используются все колонки кроме date и volume
            label_type (str): Тип задачи - 'regression' или 'classification'
            n_classes (int): Количество классов для классификации
            
        Returns:
            Dict[str, Any]: Словарь с подготовленными данными
        """
        if df.empty:
            logger.error("Невозможно подготовить данные: пустой DataFrame")
            return {}
        
        # Копируем DataFrame, чтобы не изменять оригинал
        data = df.copy()
        
        # Определяем признаки
        if feature_columns is None:
            feature_columns = [col for col in data.columns 
                              if col not in ['date', 'volume', 'timestamp']]
        
        self.feature_columns = feature_columns
        logger.info(f"Выбранные признаки: {feature_columns}")
        
        # Проверяем наличие целевой колонки
        if target_column not in data.columns:
            logger.error(f"Целевая колонка '{target_column}' отсутствует в данных")
            return {}
        
        # Создаем скейлер, если он еще не создан
        if self.scaler is None:
            self.create_scaler()
        
        # Масштабируем данные
        data_scaled = data[feature_columns].copy()
        data_scaled[feature_columns] = self.scaler.fit_transform(data[feature_columns])
        
        # Если задача классификации, подготавливаем метки
        if label_type == 'classification':
            # Создаем метки классов на основе изменения цены
            if n_classes == 3:  # 3 класса: падение, неизменно, рост
                data['target'] = 1  # неизменно (по умолчанию)
                price_change = data[target_column].pct_change(self.forecast_horizon)
                threshold = 0.005  # порог изменения (0.5%)
                data.loc[price_change > threshold, 'target'] = 2  # рост
                data.loc[price_change < -threshold, 'target'] = 0  # падение
            elif n_classes == 2:  # 2 класса: падение, рост
                data['target'] = (data[target_column].pct_change(self.forecast_horizon) > 0).astype(int)
            
            # Создаем one-hot encoding для меток
            target_values = data['target'].values[self.sequence_length:]
            target_encoded = to_categorical(target_values, num_classes=n_classes)
        else:  # регрессия
            # Для регрессии целевая переменная - значение цены в будущем
            data['target'] = data[target_column].shift(-self.forecast_horizon)
            # Масштабируем целевую переменную
            target_values = self.scaler.transform(data[[target_column]])[:-self.forecast_horizon]
            target_encoded = target_values
        
        # Создаем последовательности для обучения
        X, y = self._create_sequences(data_scaled, target_encoded)
        
        # Разделяем данные на обучающую, валидационную и тестовую выборки
        return self._split_data(X, y, label_type)
    
    def _create_sequences(self, data: pd.DataFrame, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей для входных данных.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
            targets (np.ndarray): Целевые значения
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж (X, y) с входными и целевыми данными
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Создаем последовательность из sequence_length точек
            X.append(data.iloc[i:i+self.sequence_length][self.feature_columns].values)
            # Соответствующее целевое значение
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, label_type: str) -> Dict[str, Any]:
        """
        Разделение данных на обучающую, валидационную и тестовую выборки.
        
        Args:
            X (np.ndarray): Входные данные
            y (np.ndarray): Целевые данные
            label_type (str): Тип задачи ('regression' или 'classification')
            
        Returns:
            Dict[str, Any]: Словарь с разделенными данными
        """
        # Определяем границы для разделения
        test_boundary = int(len(X) * (1 - self.test_size))
        val_boundary = int(test_boundary * (1 - self.validation_size))
        
        # Разделяем данные
        X_train, y_train = X[:val_boundary], y[:val_boundary]
        X_val, y_val = X[val_boundary:test_boundary], y[val_boundary:test_boundary]
        X_test, y_test = X[test_boundary:], y[test_boundary:]
        
        logger.info(f"Разделение данных: обучающая выборка - {len(X_train)}, "
                   f"валидационная - {len(X_val)}, тестовая - {len(X_test)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'label_type': label_type,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
    
    def inverse_transform(self, scaled_data: np.ndarray, feature_index: int = 0) -> np.ndarray:
        """
        Обратное преобразование масштабированных данных.
        
        Args:
            scaled_data (np.ndarray): Масштабированные данные
            feature_index (int): Индекс признака для обратного преобразования
            
        Returns:
            np.ndarray: Данные в исходном масштабе
        """
        if self.scaler is None:
            logger.error("Скейлер не инициализирован")
            return scaled_data
        
        # Создаем фиктивный массив для обратного преобразования
        dummy = np.zeros((len(scaled_data), len(self.feature_columns)))
        dummy[:, feature_index] = scaled_data.flatten()
        
        # Обратное преобразование
        dummy = self.scaler.inverse_transform(dummy)
        
        return dummy[:, feature_index]
    
    def prepare_single_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """
        Подготовка данных для одиночного предсказания.
        
        Args:
            df (pd.DataFrame): DataFrame с последними данными
            
        Returns:
            np.ndarray: Подготовленные данные для предсказания
        """
        if len(df) < self.sequence_length:
            logger.error(f"Недостаточно данных для предсказания. Нужно минимум {self.sequence_length} точек.")
            return np.array([])
        
        if self.scaler is None:
            logger.error("Скейлер не инициализирован")
            return np.array([])
        
        # Берем последние sequence_length точек
        recent_data = df[-self.sequence_length:][self.feature_columns].copy()
        
        # Масштабируем данные
        recent_data_scaled = self.scaler.transform(recent_data)
        
        # Формируем входные данные для модели
        X_pred = recent_data_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return X_pred