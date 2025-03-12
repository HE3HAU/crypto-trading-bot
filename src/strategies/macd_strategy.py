#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Стратегия торговли на основе индикатора MACD (Moving Average Convergence Divergence).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MacdStrategy(BaseStrategy):
    """
    Стратегия на основе индикатора MACD.
    Покупка: когда MACD пересекает сигнальную линию снизу вверх.
    Продажа: когда MACD пересекает сигнальную линию сверху вниз.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Инициализация стратегии.
        
        Args:
            fast_period (int): Период быстрой EMA
            slow_period (int): Период медленной EMA
            signal_period (int): Период сигнальной линии MACD
        """
        super().__init__(name=f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        logger.info(f"Инициализация MACD стратегии с параметрами: fast_period={fast_period}, slow_period={slow_period}, signal_period={signal_period}")
    
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
            
            # Проверяем наличие MACD или рассчитываем его если нет
            if 'macd' not in df.columns or 'macd_signal' not in df.columns:
                # Рассчитываем EMA
                fast_ema = df['close'].ewm(span=self.fast_period, adjust=False).mean()
                slow_ema = df['close'].ewm(span=self.slow_period, adjust=False).mean()
                
                # Рассчитываем MACD и сигнальную линию
                df['macd'] = fast_ema - slow_ema
                df['macd_signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Получаем последние два значения MACD и сигнальной линии
            last_macd = df['macd'].iloc[-1]
            last_signal = df['macd_signal'].iloc[-1]
            prev_macd = df['macd'].iloc[-2]
            prev_signal = df['macd_signal'].iloc[-2]
            
            # Определяем сигнал на основе пересечения
            signal = "none"
            reason = ""
            
            # Покупка: MACD пересекает сигнальную линию снизу вверх
            if prev_macd < prev_signal and last_macd > last_signal:
                signal = "buy"
                reason = "MACD пересек сигнальную линию снизу вверх"
            
            # Продажа: MACD пересекает сигнальную линию сверху вниз
            elif prev_macd > prev_signal and last_macd < last_signal:
                signal = "sell"
                reason = "MACD пересек сигнальную линию сверху вниз"
            
            # Удержание: MACD выше сигнальной линии (восходящий тренд)
            elif last_macd > last_signal:
                signal = "hold_long"
                reason = "MACD выше сигнальной линии (восходящий тренд)"
            
            # Удержание: MACD ниже сигнальной линии (нисходящий тренд)
            else:
                signal = "hold_short"
                reason = "MACD ниже сигнальной линии (нисходящий тренд)"
            
            logger.info(f"Анализ выполнен: сигнал {signal}, причина: {reason}")
            
            return {
                "signal": signal,
                "reason": reason,
                "current_price": df['close'].iloc[-1],
                "current_macd": last_macd,
                "current_signal": last_signal,
                "macd_hist": df['macd_hist'].iloc[-1],
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