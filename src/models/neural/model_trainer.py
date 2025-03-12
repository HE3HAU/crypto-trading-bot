#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для обучения нейросетевых моделей.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from src.models.neural.data_preprocessor import DataPreprocessor
from src.models.neural.lstm_model import LSTMModel
from src.models.neural.gru_model import GRUModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Класс для подготовки данных, обучения и оценки моделей.
    """
    
    def __init__(self, models_dir: str = 'models', sequence_length: int = 60,
                forecast_horizon: int = 1, test_size: float = 0.2,
                validation_size: float = 0.1, scaling_method: str = 'minmax'):
        """
        Инициализация тренера моделей.
        
        Args:
            models_dir (str): Директория для сохранения моделей
            sequence_length (int): Длина последовательности для входных данных
            forecast_horizon (int): Горизонт прогнозирования
            test_size (float): Доля данных для тестирования
            validation_size (float): Доля данных для валидации
            scaling_method (str): Метод масштабирования данных
        """
        self.models_dir = models_dir
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaling_method = scaling_method
        
        # Создаем директорию для моделей, если она не существует
        os.makedirs(models_dir, exist_ok=True)
        
        # Инициализируем препроцессор
        self.preprocessor = DataPreprocessor(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            test_size=test_size,
            validation_size=validation_size,
            scaling_method=scaling_method
        )
        
        logger.info(f"Инициализирован ModelTrainer с параметрами: "
                   f"sequence_length={sequence_length}, forecast_horizon={forecast_horizon}")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close', 
                    feature_columns: List[str] = None, label_type: str = 'regression',
                    n_classes: int = 3) -> Dict[str, Any]:
        """
        Подготовка данных для обучения моделей.
        
        Args:
            df (pd.DataFrame): DataFrame с данными
            target_column (str): Целевая колонка для прогнозирования
            feature_columns (List[str], optional): Список колонок-признаков
            label_type (str): Тип задачи ('regression' или 'classification')
            n_classes (int): Количество классов для классификации
            
        Returns:
            Dict[str, Any]: Словарь с подготовленными данными
        """
        logger.info(f"Подготовка данных для {label_type} с целевой колонкой {target_column}")
        
        prepared_data = self.preprocessor.prepare_data(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            label_type=label_type,
            n_classes=n_classes
        )
        
        if not prepared_data:
            logger.error("Не удалось подготовить данные")
            return {}
        
        return prepared_data
    
    def train_lstm_model(self, prepared_data: Dict[str, Any], 
                         lstm_units: List[int] = [50, 50],
                         dropout_rate: float = 0.2, 
                         learning_rate: float = 0.001,
                         epochs: int = 50, 
                         batch_size: int = 32,
                         model_name: str = 'lstm_model') -> Tuple[LSTMModel, Dict[str, Any]]:
        """
        Обучение LSTM модели.
        
        Args:
            prepared_data (Dict[str, Any]): Подготовленные данные
            lstm_units (List[int]): Список с количеством юнитов в каждом слое LSTM
            dropout_rate (float): Коэффициент отсева для слоев Dropout
            learning_rate (float): Скорость обучения оптимизатора
            epochs (int): Количество эпох обучения
            batch_size (int): Размер батча
            model_name (str): Имя модели для сохранения
            
        Returns:
            Tuple[LSTMModel, Dict[str, Any]]: Модель и история обучения
        """
        if not prepared_data:
            logger.error("Не предоставлены подготовленные данные")
            return None, {}
        
        # Извлекаем данные
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        label_type = prepared_data['label_type']
        
        # Определяем параметры модели
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        output_activation = 'softmax' if label_type == 'classification' else 'linear'
        
        # Создаем модель
        model = LSTMModel(
            input_shape=input_shape,
            output_size=output_size,
            output_activation=output_activation,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Строим модель
        model.build_model()
        
        # Путь для сохранения модели
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.h5")
        
        # Обучаем модель
        logger.info(f"Начало обучения LSTM модели: {epochs} эпох, batch_size={batch_size}")
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path,
            patience=10
        )
        
        return model, history
    
    def train_gru_model(self, prepared_data: Dict[str, Any], 
                        gru_units: List[int] = [50, 50],
                        dropout_rate: float = 0.2, 
                        learning_rate: float = 0.001,
                        epochs: int = 50, 
                        batch_size: int = 32,
                        model_name: str = 'gru_model') -> Tuple[GRUModel, Dict[str, Any]]:
        """
        Обучение GRU модели.
        
        Args:
            prepared_data (Dict[str, Any]): Подготовленные данные
            gru_units (List[int]): Список с количеством юнитов в каждом слое GRU
            dropout_rate (float): Коэффициент отсева для слоев Dropout
            learning_rate (float): Скорость обучения оптимизатора
            epochs (int): Количество эпох обучения
            batch_size (int): Размер батча
            model_name (str): Имя модели для сохранения
            
        Returns:
            Tuple[GRUModel, Dict[str, Any]]: Модель и история обучения
        """
        if not prepared_data:
            logger.error("Не предоставлены подготовленные данные")
            return None, {}
        
        # Извлекаем данные
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        label_type = prepared_data['label_type']
        
        # Определяем параметры модели
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        output_activation = 'softmax' if label_type == 'classification' else 'linear'
        
        # Создаем модель
        model = GRUModel(
            input_shape=input_shape,
            output_size=output_size,
            output_activation=output_activation,
            gru_units=gru_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Строим модель
        model.build_model()
        
        # Путь для сохранения модели
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.h5")
        
        # Обучаем модель
        logger.info(f"Начало обучения GRU модели: {epochs} эпох, batch_size={batch_size}")
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path,
            patience=10
        )
        
        return model, history
    
    def evaluate_model(self, model: Any, prepared_data: Dict[str, Any], 
                      plot_results: bool = True, save_dir: str = 'visualizations') -> Dict[str, float]:
        """
        Оценка модели на тестовых данных.
        
        Args:
            model (Any): Модель (LSTM или GRU)
            prepared_data (Dict[str, Any]): Подготовленные данные
            plot_results (bool): Создавать ли визуализации результатов
            save_dir (str): Директория для сохранения визуализаций
            
        Returns:
            Dict[str, float]: Метрики оценки
        """
        if not prepared_data:
            logger.error("Не предоставлены подготовленные данные")
            return {}
        
        # Извлекаем данные
        X_test = prepared_data['X_test']
        y_test = prepared_data['y_test']
        label_type = prepared_data['label_type']
        scaler = prepared_data['scaler']
        
        # Оцениваем модель
        metrics = model.evaluate(X_test, y_test)
        
        if plot_results and label_type == 'regression':
            # Создаем директорию для визуализаций
            os.makedirs(save_dir, exist_ok=True)
            
            # Получаем предсказания
            y_pred = model.predict(X_test)
            
            # Обратное преобразование предсказаний и фактических значений
            y_test_inv = self.preprocessor.inverse_transform(y_test)
            y_pred_inv = self.preprocessor.inverse_transform(y_pred)
            
            # Создаем график
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_inv, label='Actual')
            plt.plot(y_pred_inv, label='Predicted')
            plt.title('Model Predictions vs Actual')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Сохраняем график
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(save_dir, f"prediction_results_{timestamp}.png")
            plt.savefig(filepath, dpi=150)
            plt.close()
            
            logger.info(f"График результатов сохранен в {filepath}")
        
        return metrics
    
    def predict_next_values(self, model: Any, df: pd.DataFrame, 
                           n_steps: int = 5) -> List[float]:
        """
        Прогнозирование следующих значений временного ряда.
        
        Args:
            model (Any): Модель (LSTM или GRU)
            df (pd.DataFrame): DataFrame с последними данными
            n_steps (int): Количество шагов для прогноза
            
        Returns:
            List[float]: Список с прогнозными значениями
        """
        # Проверяем наличие модели и данных
        if model is None or df.empty:
            logger.error("Не предоставлены модель или данные")
            return []
        
        # Подготавливаем данные для первого предсказания
        X_pred = self.preprocessor.prepare_single_prediction(df)
        
        if X_pred.size == 0:
            logger.error("Не удалось подготовить данные для предсказания")
            return []
        
        predictions = []
        latest_data = df.copy()
        
        # Выполняем предсказания на n_steps вперед
        for i in range(n_steps):
            # Получаем предсказание
            pred = model.predict(X_pred)[0]
            
            # Для регрессии преобразуем предсказание обратно
            if len(pred.shape) == 0 or pred.shape[0] == 1:
                pred_value = self.preprocessor.inverse_transform(np.array([pred]))[0]
            else:
                # Для классификации берем индекс максимального значения
                pred_class = np.argmax(pred)
                pred_value = pred_class
            
            predictions.append(pred_value)
            
            # Добавляем предсказание в данные для следующего шага
            if i < n_steps - 1:
                # Создаем новую строку с предсказанной ценой
                new_row = latest_data.iloc[-1:].copy()
                new_row.index = [latest_data.index[-1] + pd.Timedelta(minutes=5)]  # для 5-минутного таймфрейма
                new_row['close'] = pred_value
                
                # Добавляем новую строку к данным
                latest_data = pd.concat([latest_data, new_row])
                
                # Подготавливаем данные для следующего предсказания
                X_pred = self.preprocessor.prepare_single_prediction(latest_data)
        
        logger.info(f"Получены прогнозы на {n_steps} шагов вперед: {predictions}")
        
        return predictions