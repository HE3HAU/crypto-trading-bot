#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Комбинированная стратегия торговли, использующая несколько индикаторов.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.strategies.base_strategy import BaseStrategy
from src.strategies.sma_strategy import SmaStrategy
from src.strategies.rsi_strategy import RsiStrategy
from src.strategies.macd_strategy import MacdStrategy

logger = logging.getLogger(__name__)

class CombinedStrategy(BaseStrategy):
    """
    Комбинированная стратегия, объединяющая сигналы нескольких стратегий.
    Генерирует сигналы на основе голосования или усиленных сигналов.
    """
    
    def __init__(self, use_sma: bool = True, use_rsi: bool = True, use_macd: bool = True, 
                 voting_threshold: float = 0.5):
        """
        Инициализация стратегии.
        
        Args:
            use_sma (bool): Использовать SMA стратегию
            use_rsi (bool): Использовать RSI стратегию
            use_macd (bool): Использовать MACD стратегию
            voting_threshold (float): Порог для голосования (от 0 до 1)
        """
        super().__init__(name="Combined_Strategy")
        self.use_sma = use_sma
        self.use_rsi = use_rsi
        self.use_macd = use_macd
        self.voting_threshold = voting_threshold
        
        # Инициализация базовых стратегий
        self.strategies = []
        
        if use_sma:
            self.strategies.append(SmaStrategy())
        if use_rsi:
            self.strategies.append(RsiStrategy())
        if use_macd:
            self.strategies.append(MacdStrategy())
        
        strategy_names = [s.name for s in self.strategies]
        logger.info(f"Инициализация комбинированной стратегии с компонентами: {', '.join(strategy_names)}, порог: {voting_threshold}")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ данных и генерация торговых сигналов на основе комбинации стратегий.
        
        Args:
            data (pd.DataFrame): DataFrame с данными для анализа
            
        Returns:
            Dict[str, Any]: Результаты анализа и торговые сигналы
        """
        if data.empty:
            logger.warning("Анализ невозможен: пустой DataFrame")
            return {"signal": "none", "reason": "no data"}
        
        if not self.strategies:
            logger.warning("Анализ невозможен: не включена ни одна стратегия")
            return {"signal": "none", "reason": "no strategies"}
        
        try:
            # Собираем результаты анализа от всех стратегий
            strategy_results = []
            for strategy in self.strategies:
                result = strategy.analyze(data)
                strategy_results.append(result)
            
            # Считаем голоса
            votes = {"buy": 0, "sell": 0, "hold": 0}
            reasons = []
            
            for result in strategy_results:
                signal = result["signal"]
                
                if signal == "buy":
                    votes["buy"] += 1
                elif signal == "sell":
                    votes["sell"] += 1
                else:
                    votes["hold"] += 1
                
                reasons.append(f"{result['signal']} ({result['reason']})")
            
            # Определяем итоговый сигнал на основе голосования
            total_votes = len(self.strategies)
            buy_ratio = votes["buy"] / total_votes
            sell_ratio = votes["sell"] / total_votes
            
            final_signal = "hold"
            
            if buy_ratio >= self.voting_threshold:
                final_signal = "buy"
            elif sell_ratio >= self.voting_threshold:
                final_signal = "sell"
            
            # Формируем объяснение решения
            reason = f"Голосование: Buy - {votes['buy']}, Sell - {votes['sell']}, Hold - {votes['hold']}. Использовано {len(self.strategies)} стратегий."
            
            logger.info(f"Анализ выполнен: сигнал {final_signal}, причина: {reason}")
            
            return {
                "signal": final_signal,
                "reason": reason,
                "votes": votes,
                "strategy_results": strategy_results,
                "current_price": data['close'].iloc[-1],
                "timestamp": data.index[-1],
                "detailed_reasons": reasons,
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