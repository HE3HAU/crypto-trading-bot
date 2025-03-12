#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль с LSTM моделью для прогнозирования временных рядов.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    Класс для создания, обучения и использования LSTM модели.
    """
    
    def __init__(self, input_shape: Tuple[int, int], output_size: int,
                output_activation: str = 'linear', lstm_units: List[int] = [50, 50],
                dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Инициализация LSTM модели.
        
        Args:
            input_shape (Tuple[int, int]): Форма входных данных (sequence_length, n_features)
            output_size (int): Размер выходного слоя (1 для регрессии, n_classes для классификации)
            output_activation (str): Функция активации выходного слоя
            lstm_units (List[int]): Список с количеством юнитов в каждом слое LSTM
            dropout_rate (float): Коэффициент отсева для слоев Dropout
            learning_rate (float): Скорость обучения оптимизатора
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_activation = output_activation
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
        logger.info(f"Инициализирована LSTM модель с параметрами: "
                   f"input_shape={input_shape}, output_size={output_size}, "
                   f"lstm_units={lstm_units}, dropout_rate={dropout_rate}")
    
    def build_model(self) -> None:
        """
        Построение архитектуры модели.
        """
        model = Sequential()
        
        # Первый слой LSTM и регуляризация
        model.add(LSTM(self.lstm_units[0], return_sequences=len(self.lstm_units) > 1,
                      input_shape=self.input_shape, activation='tanh',
                      recurrent_activation='sigmoid'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Дополнительные слои LSTM (если указаны)
        for i in range(1, len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(self.lstm_units[i], return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Выходной слой
        model.add(Dense(self.output_size, activation=self.output_activation))
        
        # Компиляция модели
        if self.output_activation == 'softmax':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mean_squared_error'
            metrics = ['mean_absolute_error']
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        logger.info(f"Построена модель: {model.summary()}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32,
             model_path: str = 'models/lstm_model.h5',
             patience: int = 10) -> Dict[str, Any]:
        """
        Обучение модели.
        
        Args:
            X_train (np.ndarray): Обучающие данные
            y_train (np.ndarray): Целевые значения для обучения
            X_val (np.ndarray): Валидационные данные
            y_val (np.ndarray): Целевые значения для валидации
            epochs (int): Количество эпох обучения
            batch_size (int): Размер батча
            model_path (str): Путь для сохранения модели
            patience (int): Терпение для early stopping
            
        Returns:
            Dict[str, Any]: История обучения
        """
        if self.model is None:
            self.build_model()
        
        # Создаем колбэки
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6)
        ]
        
        # Обучаем модель
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Обучение завершено: {len(history.history['loss'])} эпох")
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание с помощью модели.
        
        Args:
            X (np.ndarray): Входные данные
            
        Returns:
            np.ndarray: Предсказания модели
        """
        if self.model is None:
            logger.error("Модель не построена")
            return np.array([])
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Сохранение модели.
        
        Args:
            filepath (str): Путь для сохранения модели
        """
        if self.model is None:
            logger.error("Нет модели для сохранения")
            return
        
        self.model.save(filepath)
        logger.info(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """
        Загрузка модели.
        
        Args:
            filepath (str): Путь к сохраненной модели
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            self.model = load_model(filepath)
            logger.info(f"Модель загружена из {filepath}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return False
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Оценка модели на тестовых данных.
        
        Args:
            X_test (np.ndarray): Тестовые данные
            y_test (np.ndarray): Целевые значения для теста
            
        Returns:
            Dict[str, float]: Метрики оценки
        """
        if self.model is None:
            logger.error("Модель не построена")
            return {}
        
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {}
        
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = evaluation[i]
        
        logger.info(f"Результаты оценки: {metrics}")
        
        return metrics