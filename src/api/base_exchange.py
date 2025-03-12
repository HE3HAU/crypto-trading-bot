#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Абстрактный базовый класс для работы с криптовалютными биржами.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class BaseExchange(ABC):
    """
    Абстрактный базовый класс для реализации API бирж.
    Все конкретные реализации бирж должны наследоваться от этого класса.
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", password: str = "", sandbox: bool = True):
        """
        Инициализация базового класса биржи.
        
        Args:
            api_key (str): API ключ для аутентификации на бирже
            secret_key (str): Секретный ключ для аутентификации
            password (str, optional): Пароль (для бирж, которые его требуют)
            sandbox (bool): Использовать тестовый режим (песочницу)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.password = password
        self.sandbox = sandbox
        self.exchange_name = self.__class__.__name__
        logger.info(f"Инициализация биржи {self.exchange_name}, sandbox: {sandbox}")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Установка соединения с биржей.
        
        Returns:
            bool: Успешность соединения
        """
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """
        Получение баланса аккаунта.
        
        Returns:
            Dict[str, float]: Словарь с балансами валют
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Получение текущей цены и связанной информации по торговой паре.
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC/USDT')
            
        Returns:
            Dict[str, Any]: Информация о торговой паре
        """
        pass
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List[float]]:
        """
        Получение OHLCV данных (Open, High, Low, Close, Volume).
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Временной интервал ('1m', '5m', '1h', '1d', и т.д.)
            limit (int): Количество свечей
            
        Returns:
            List[List[float]]: Список OHLCV свечей
        """
        pass
    
    @abstractmethod
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                     price: Optional[float] = None) -> Dict[str, Any]:
        """
        Создание ордера на бирже.
        
        Args:
            symbol (str): Символ торговой пары
            order_type (str): Тип ордера ('limit', 'market')
            side (str): Сторона ('buy', 'sell')
            amount (float): Количество 
            price (Optional[float]): Цена для лимитного ордера
            
        Returns:
            Dict[str, Any]: Информация о созданном ордере
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Отмена ордера.
        
        Args:
            order_id (str): ID ордера
            symbol (str): Символ торговой пары
            
        Returns:
            bool: Успешность отмены ордера
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Получение информации об ордере.
        
        Args:
            order_id (str): ID ордера
            symbol (str): Символ торговой пары
            
        Returns:
            Dict[str, Any]: Информация об ордере
        """
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение списка открытых ордеров.
        
        Args:
            symbol (Optional[str]): Символ торговой пары (если None, то все пары)
            
        Returns:
            List[Dict[str, Any]]: Список открытых ордеров
        """
        pass