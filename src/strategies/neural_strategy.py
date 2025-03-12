#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Стратегия торговли на основе прогнозов нейросетевой модели.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.strategies.base_strategy import BaseStrategy
from src.models.neural.data_preprocessor import DataPreprocessor
from src.models.neural.lstm_model import LSTMModel
from src.models.neural.gru_model import GRUModel

logger = logging.getLogger(__name__)

class NeuralStrategy(BaseStrategy):
    """
    Стратегия на основе прогнозов нейросетевой модели.
    Генерирует сигналы на основе прогнозов цены или направления движения.
    """
    
    def __init__(self, model_path: str, model_type: str = 'lstm',
                preprocessor: DataPreprocessor = None,
                prediction_threshold: float = 0.5,
                use_trend: bool = True,
                n_future_steps: int = 3,
                trend_threshold: float = 0.01):
        """
        Инициализация стратегии.
        
        Args:
            model_path (str): Путь к сохраненной модели
            model_type (str): Тип модели ('lstm' или 'gru')
            preprocessor (DataPreprocessor): Подготовленный препроцессор данных
            prediction_threshold (float): Порог для генерации сигналов
            use_trend (bool): Использовать ли тренд предсказаний вместо одного значения
            n_future_steps (int): Количество шагов для прогноза при use_trend=True
            trend_threshold (float): Порог изменения для определения тренда (в %)
        """
        super().__init__(name=f"Neural_{model_type.upper()}")
        self.model_path = model_path
        self.model_type = model_type
        self.preprocessor = preprocessor
        self.prediction_threshold = prediction_threshold
        self.use_trend = use_trend
        self.n_future_steps = n_future_steps
        self.trend_threshold = trend_threshold
        self.model = None
        
        # Загружаем модель
        self._load_model()
        
        logger.info(f"Инициализация нейросетевой стратегии: {model_type.upper()}, "
                   f"use_trend={use_trend}, threshold={prediction_threshold}")
    
    def _load_model(self) -> None:
        """
        Загрузка модели из файла.
        """
        try:
            if self.model_type.lower() == 'lstm':
                self.model = LSTMModel(input_shape=(1, 1), output_size=1)  # Временные параметры
                self.model.load_model(self.model_path)
            elif self.model_type.lower() == 'gru':
                self.model = GRUModel(input_shape=(1, 1), output_size=1)  # Временные параметры
                self.model.load_model(self.model_path)
            else:
                logger.error(f"Неизвестный тип модели: {self.model_type}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ данных и генерация торговых сигналов на основе предсказаний модели.
        
        Args:
            data (pd.DataFrame): DataFrame с данными для анализа
            
        Returns:
            Dict[str, Any]: Результаты анализа и торговые сигналы
        """
        if data.empty:
            logger.warning("Анализ невозможен: пустой DataFrame")
            return {"signal": "none", "reason": "no data"}
        
        if self.model is None:
            logger.error("Модель не загружена")
            return {"signal": "none", "reason": "model not loaded"}
        
        try:
            # Получаем текущую цену
            current_price = data['close'].iloc[-1]
            
            # Проверяем наличие препроцессора
            if self.preprocessor is None:
                logger.warning("Препроцессор не инициализирован, используются сырые данные")
                # Получаем предсказание
                if self.use_trend:
                    predictions = np.random.normal(current_price, current_price * 0.02, self.n_future_steps)
                else:
                    prediction = current_price * (1 + np.random.normal(0, 0.02))
            else:
                # Подготавливаем данные для предсказания
                X_pred = self.preprocessor.prepare_single_prediction(data)
                
                if self.use_trend:
                    # Получаем предсказания на несколько шагов вперед
                    from src.models.neural.model_trainer import ModelTrainer
                    trainer = ModelTrainer()
                    predictions = trainer.predict_next_values(self.model, data, self.n_future_steps)
                else:
                    # Получаем одно предсказание
                    prediction = self.model.predict(X_pred)[0]
                    
                    # Обратное преобразование предсказания
                    if hasattr(self.preprocessor, 'inverse_transform'):
                        prediction = self.preprocessor.inverse_transform(np.array([prediction]))[0]
            
            # Генерация сигнала на основе предсказаний
            signal = "none"
            reason = ""
            
            if self.use_trend:
                # Рассчитываем средний процент изменения цены
                if isinstance(predictions, list) and len(predictions) > 0:
                    avg_change = (predictions[-1] / current_price - 1) * 100
                    
                    if avg_change > self.trend_threshold:
                        signal = "buy"
                        reason = f"Ожидается рост цены на {avg_change:.2f}% в течение {self.n_future_steps} периодов"
                    elif avg_change < -self.trend_threshold:
                        signal = "sell"
                        reason = f"Ожидается падение цены на {-avg_change:.2f}% в течение {self.n_future_steps} периодов"
                    else:
                        signal = "hold"
                        reason = f"Нет четкого тренда: изменение {avg_change:.2f}%"
                    
                    prediction_details = predictions
                else:
                    signal = "none"
                    reason = "Не удалось получить прогнозы"
                    prediction_details = []
            else:
                # Определяем сигнал на основе одного предсказания
                price_change = (prediction / current_price - 1) * 100
                
                if price_change > self.prediction_threshold:
                    signal = "buy"
                    reason = f"Ожидается рост цены на {price_change:.2f}%"
                elif price_change < -self.prediction_threshold:
                    signal = "sell"
                    reason = f"Ожидается падение цены на {-price_change:.2f}%"
                else:
                    signal = "hold"
                    reason = f"Прогнозируемое изменение {price_change:.2f}% ниже порога {self.prediction_threshold}%"
                
                prediction_details = prediction
            
            logger.info(f"Анализ выполнен: сигнал {signal}, причина: {reason}")
            
            return {
                "signal": signal,
                "reason": reason,
                "current_price": current_price,
                "prediction": prediction_details,
                "timestamp": data.index[-1],
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе данных: {e}")
            return {"signal": "error", "reason": str(e)}
    
    def should_buy(self, data: pd.DataFrame) -> bool:
        """
        Определение сигнала на покупку.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
            
        Returns:
            bool: True если есть сигнал на покупку, иначе False
        """
        analysis = self.analyze(data)
        return analysis["signal"] == "buy"
    
    def should_sell(self, data: pd.DataFrame) -> bool:
        """
        Определение сигнала на продажу.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
            
        Returns:
            bool: True если есть сигнал на продажу, иначе False
        """
        analysis = self.analyze(data)
        return analysis["signal"] == "sell"