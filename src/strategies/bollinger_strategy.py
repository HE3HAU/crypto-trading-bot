#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Стратегия торговли на основе полос Боллинджера (Bollinger Bands).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BollingerStrategy(BaseStrategy):
    """
    Стратегия на основе полос Боллинджера.
    Покупка: когда цена пересекает нижнюю полосу снизу вверх.
    Продажа: когда цена пересекает верхнюю полосу снизу вверх.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        """
        Инициализация стратегии.
        
        Args:
            window (int): Период для расчета полос Боллинджера
            num_std (float): Количество стандартных отклонений для полос
        """
        super().__init__(name=f"Bollinger_{window}_{num_std}")
        self.window = window
        self.num_std = num_std
        logger.info(f"Инициализация Bollinger стратегии с параметрами: window={window}, num_std={num_std}")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ данных и генерация торговых сигналов.
        
        Args:
            data (pd.DataFrame): DataFrame с данными для анализа
            
        Returns:
            Dict[str, Any]: Результаты анализа и торговые сигналы
        """
        if data.empty:
            logger.warning("Анализ невозможен: пустой DataFrame")
            return {"signal": "none", "reason": "no data"}
        
        try:
            # Создаем копию данных
            df = data.copy()
            
            # Проверяем наличие полос Боллинджера или рассчитываем их если нет
            bb_mid_col = f'bb_mid{self.window}'
            bb_upper_col = f'bb_upper{self.window}'
            bb_lower_col = f'bb_lower{self.window}'
            
            if bb_mid_col not in df.columns or bb_upper_col not in df.columns or bb_lower_col not in df.columns:
                # Рассчитываем среднюю и стандартное отклонение
                df[bb_mid_col] = df['close'].rolling(window=self.window).mean()
                rolling_std = df['close'].rolling(window=self.window).std()
                
                # Рассчитываем верхнюю и нижнюю полосы
                df[bb_upper_col] = df[bb_mid_col] + (rolling_std * self.num_std)
                df[bb_lower_col] = df[bb_mid_col] - (rolling_std * self.num_std)
            
            # Получаем последние два значения цены и полос
            last_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            last_upper = df[bb_upper_col].iloc[-1]
            last_lower = df[bb_lower_col].iloc[-1]
            prev_upper = df[bb_upper_col].iloc[-2]
            prev_lower = df[bb_lower_col].iloc[-2]
            
            # Определяем сигнал на основе пересечения полос
            signal = "none"
            reason = ""
            
            # Покупка: цена пересекает нижнюю полосу снизу вверх
            if prev_close < prev_lower and last_close > last_lower:
                signal = "buy"
                reason = "Цена пересекла нижнюю полосу Боллинджера снизу вверх"
            
            # Продажа: цена пересекает верхнюю полосу сверху вниз
            elif prev_close > prev_upper and last_close < last_upper:
                signal = "sell"
                reason = "Цена пересекла верхнюю полосу Боллинджера сверху вниз"
            
            # Дополнительные сигналы для более тонкого анализа
            
            # Приближение к верхней полосе
            elif last_close > df[bb_mid_col].iloc[-1] and last_close > 0.95 * last_upper:
                signal = "hold_near_sell"
                reason = "Цена приближается к верхней полосе Боллинджера"
            
            # Приближение к нижней полосе
            elif last_close < df[bb_mid_col].iloc[-1] and last_close < 1.05 * last_lower:
                signal = "hold_near_buy"
                reason = "Цена приближается к нижней полосе Боллинджера"
            
            # Нейтральная зона - цена между полосами
            else:
                if last_close > df[bb_mid_col].iloc[-1]:
                    signal = "hold_upper_zone"
                    reason = "Цена в верхней зоне Боллинджера"
                else:
                    signal = "hold_lower_zone"
                    reason = "Цена в нижней зоне Боллинджера"
            
            logger.info(f"Анализ выполнен: сигнал {signal}, причина: {reason}")
            
            # Рассчитываем процент ширины полосы Боллинджера
            bandwidth = (last_upper - last_lower) / df[bb_mid_col].iloc[-1] * 100
            
            return {
                "signal": signal,
                "reason": reason,
                "current_price": last_close,
                "upper_band": last_upper,
                "middle_band": df[bb_mid_col].iloc[-1],
                "lower_band": last_lower,
                "bandwidth": bandwidth,
                "timestamp": df.index[-1],
                "data": df
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