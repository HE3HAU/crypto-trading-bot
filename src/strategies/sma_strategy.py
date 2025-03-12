#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Стратегия торговли на основе скользящих средних (Simple Moving Average).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SmaStrategy(BaseStrategy):
    """
    Стратегия на основе пересечения скользящих средних.
    Покупка: когда короткая SMA пересекает длинную SMA снизу вверх.
    Продажа: когда короткая SMA пересекает длинную SMA сверху вниз.
    """
    
    def __init__(self, short_window: int = 7, long_window: int = 25):
        """
        Инициализация стратегии.
        
        Args:
            short_window (int): Период короткой скользящей средней
            long_window (int): Период длинной скользящей средней
        """
        super().__init__(name=f"SMA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        logger.info(f"Инициализация SMA стратегии с параметрами: short_window={short_window}, long_window={long_window}")
    
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
            
            # Если индикаторы еще не добавлены, добавляем их
            if f'sma{self.short_window}' not in df.columns:
                df[f'sma{self.short_window}'] = df['close'].rolling(window=self.short_window).mean()
            
            if f'sma{self.long_window}' not in df.columns:
                df[f'sma{self.long_window}'] = df['close'].rolling(window=self.long_window).mean()
            
            # Создаем колонку с сигналами
            # 1: покупка, -1: продажа, 0: удержание
            df['signal'] = 0
            
            # Генерация сигналов на основе пересечения скользящих средних
            df['signal'] = np.where(
                df[f'sma{self.short_window}'] > df[f'sma{self.long_window}'], 1, 0)
            
            # Находим точки пересечения (изменения сигнала)
            df['position_change'] = df['signal'].diff()
            
            # Последняя точка данных
            latest = df.iloc[-1]
            
            # Проверяем, есть ли сигнал на последней свече
            if latest['position_change'] > 0:
                signal = "buy"
                reason = f"Короткая SMA ({self.short_window}) пересекла длинную SMA ({self.long_window}) снизу вверх"
            elif latest['position_change'] < 0:
                signal = "sell"
                reason = f"Короткая SMA ({self.short_window}) пересекла длинную SMA ({self.long_window}) сверху вниз"
            else:
                if latest['signal'] == 1:
                    signal = "hold_long"
                    reason = f"Короткая SMA ({self.short_window}) выше длинной SMA ({self.long_window})"
                else:
                    signal = "hold_short"
                    reason = f"Короткая SMA ({self.short_window}) ниже длинной SMA ({self.long_window})"
            
            # Дополнительные данные для принятия решения
            current_price = latest['close']
            short_sma = latest[f'sma{self.short_window}']
            long_sma = latest[f'sma{self.long_window}']
            
            logger.info(f"Анализ выполнен: сигнал {signal}, причина: {reason}")
            
            return {
                "signal": signal,
                "reason": reason,
                "current_price": current_price,
                "short_sma": short_sma,
                "long_sma": long_sma,
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