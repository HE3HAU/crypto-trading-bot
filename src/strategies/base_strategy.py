#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Базовый класс для торговых стратегий.
"""

from abc import ABC, abstractmethod
import logging
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Абстрактный базовый класс для торговых стратегий.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Инициализация базовой стратегии.
        
        Args:
            name (str): Имя стратегии
        """
        self.name = name
        logger.info(f"Инициализация стратегии: {name}")
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ данных и генерация торговых сигналов.
        
        Args:
            data (pd.DataFrame): DataFrame с данными для анализа
            
        Returns:
            Dict[str, Any]: Результаты анализа и торговые сигналы
        """
        pass
    
    @abstractmethod
    def should_buy(self, data: pd.DataFrame) -> bool:
        """
        Определение сигнала на покупку.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
            
        Returns:
            bool: True если есть сигнал на покупку, иначе False
        """
        pass
    
    @abstractmethod
    def should_sell(self, data: pd.DataFrame) -> bool:
        """
        Определение сигнала на продажу.
        
        Args:
            data (pd.DataFrame): DataFrame с данными
            
        Returns:
            bool: True если есть сигнал на продажу, иначе False
        """
        pass
    
    def calculate_position_size(self, available_balance: float, risk_per_trade: float = 0.02) -> float:
        """
        Расчет размера позиции на основе управления капиталом.
        
        Args:
            available_balance (float): Доступный баланс
            risk_per_trade (float): Риск на одну сделку (по умолчанию 2%)
            
        Returns:
            float: Размер позиции
        """
        position_size = available_balance * risk_per_trade
        logger.info(f"Рассчитан размер позиции: {position_size} (баланс: {available_balance}, риск: {risk_per_trade*100}%)")
        return position_size