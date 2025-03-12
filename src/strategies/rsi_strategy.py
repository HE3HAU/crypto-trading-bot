#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Стратегия торговли на основе индикатора RSI (Relative Strength Index).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RsiStrategy(BaseStrategy):
    """
    Стратегия на основе индикатора RSI.
    Покупка: когда RSI пересекает уровень перепроданности снизу вверх.
    Продажа: когда RSI пересекает уровень перекупленности сверху вниз.
    """
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        """
        Инициализация стратегии.
        
        Args:
            rsi_period (int): Период для расчета RSI
            oversold (int): Уровень перепроданности (обычно 30)
            overbought (int): Уровень перекупленности (обычно 70)
        """
        super().__init__(name=f"RSI_{rsi_period}_{oversold}_{overbought}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        logger.info(f"Инициализация RSI стратегии с параметрами: period={rsi_period}, oversold={oversold}, overbought={overbought}")
    
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
            
            # Проверяем наличие RSI или рассчитываем его если нет
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # Получаем последние два значения RSI для определения пересечения
            last_rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            
            # Определяем сигнал на основе пересечения уровней
            signal = "none"
            reason = ""
            
            # Покупка: RSI пересекает уровень перепроданности снизу вверх
            if prev_rsi < self.oversold and last_rsi > self.oversold:
                signal = "buy"
                reason = f"RSI ({last_rsi:.2f}) пересек уровень перепроданности ({self.oversold}) снизу вверх"
            
            # Продажа: RSI пересекает уровень перекупленности сверху вниз
            elif prev_rsi > self.overbought and last_rsi < self.overbought:
                signal = "sell"
                reason = f"RSI ({last_rsi:.2f}) пересек уровень перекупленности ({self.overbought}) сверху вниз"
            
            # Удержание: RSI находится в зоне перепроданности
            elif last_rsi < self.oversold:
                signal = "hold_wait_buy"
                reason = f"RSI ({last_rsi:.2f}) находится в зоне перепроданности (< {self.oversold})"
            
            # Удержание: RSI находится в зоне перекупленности
            elif last_rsi > self.overbought:
                signal = "hold_wait_sell"
                reason = f"RSI ({last_rsi:.2f}) находится в зоне перекупленности (> {self.overbought})"
            
            # Удержание: RSI находится в нейтральной зоне
            else:
                signal = "hold_neutral"
                reason = f"RSI ({last_rsi:.2f}) находится в нейтральной зоне"
            
            logger.info(f"Анализ выполнен: сигнал {signal}, причина: {reason}")
            
            return {
                "signal": signal,
                "reason": reason,
                "current_price": df['close'].iloc[-1],
                "current_rsi": last_rsi,
                "previous_rsi": prev_rsi,
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