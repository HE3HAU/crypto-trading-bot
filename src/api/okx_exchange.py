#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Реализация API для биржи OKX.
"""

import ccxt
import logging
from typing import Dict, List, Optional, Any, Tuple
import time

from api.base_exchange import BaseExchange

logger = logging.getLogger(__name__)

class OkxExchange(BaseExchange):
    """
    Реализация API для биржи OKX с использованием библиотеки CCXT.
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", password: str = "", sandbox: bool = True):
        """
        Инициализация соединения с OKX.
        
        Args:
            api_key (str): API ключ
            secret_key (str): Секретный ключ
            password (str): Пароль
            sandbox (bool): Использовать тестовый режим
        """
        super().__init__(api_key, secret_key, password, sandbox)
        self.exchange = None
        
    def connect(self) -> bool:
        """
        Установка соединения с OKX.
        
        Returns:
            bool: Успешность соединения
        """
        try:
            # Инициализация клиента CCXT для OKX
            self.exchange = ccxt.okx({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'password': self.password,
                'enableRateLimit': True,  # Включение ограничения запросов
            })
            
            # Настройка режима песочницы
            if self.sandbox:
                self.exchange.set_sandbox_mode(True)
                logger.info("Установлен режим песочницы для OKX")
            
            # Проверка соединения
            self.exchange.load_markets()
            logger.info(f"Успешное соединение с OKX. Доступно {len(self.exchange.markets)} торговых пар.")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при соединении с OKX: {e}")
            return False
    
    def get_balance(self) -> Dict[str, float]:
        """
        Получение баланса аккаунта на OKX.
        
        Returns:
            Dict[str, float]: Словарь с балансами валют
        """
        try:
            balance = self.exchange.fetch_balance()
            # Оставляем только ненулевые балансы
            result = {currency: float(data['total']) 
                     for currency, data in balance['total'].items() 
                     if data and float(data) > 0}
            
            logger.debug(f"Получен баланс: {result}")
            return result
        except Exception as e:
            logger.error(f"Ошибка при получении баланса: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Получение текущей цены и связанной информации по торговой паре.
        
        Args:
            symbol (str): Символ торговой пары (например, 'BTC/USDT')
            
        Returns:
            Dict[str, Any]: Информация о торговой паре
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(f"Получен тикер для {symbol}: последняя цена {ticker['last']}")
            return ticker
        except Exception as e:
            logger.error(f"Ошибка при получении тикера для {symbol}: {e}")
            return {}
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List[float]]:
        """
        Получение OHLCV данных (Open, High, Low, Close, Volume).
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Временной интервал ('1m', '5m', '1h', '1d', и т.д.)
            limit (int): Количество свечей
            
        Returns:
            List[List[float]]: Список OHLCV свечей [[timestamp, open, high, low, close, volume], ...]
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.debug(f"Получены OHLCV данные для {symbol} ({timeframe}): {len(ohlcv)} свечей")
            return ohlcv
        except Exception as e:
            logger.error(f"Ошибка при получении OHLCV данных для {symbol}: {e}")
            return []
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                     price: Optional[float] = None) -> Dict[str, Any]:
        """
        Создание ордера на OKX.
        
        Args:
            symbol (str): Символ торговой пары
            order_type (str): Тип ордера ('limit', 'market')
            side (str): Сторона ('buy', 'sell')
            amount (float): Количество 
            price (Optional[float]): Цена для лимитного ордера
            
        Returns:
            Dict[str, Any]: Информация о созданном ордере
        """
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            logger.info(f"Создан ордер: {order['id']} ({symbol}, {side}, {amount})")
            return order
        except Exception as e:
            logger.error(f"Ошибка при создании ордера для {symbol}: {e}")
            return {}
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Отмена ордера на OKX.
        
        Args:
            order_id (str): ID ордера
            symbol (str): Символ торговой пары
            
        Returns:
            bool: Успешность отмены ордера
        """
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Отменен ордер {order_id} для {symbol}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при отмене ордера {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Получение информации об ордере на OKX.
        
        Args:
            order_id (str): ID ордера
            symbol (str): Символ торговой пары
            
        Returns:
            Dict[str, Any]: Информация об ордере
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Ошибка при получении информации о ордере {order_id}: {e}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение списка открытых ордеров на OKX.
        
        Args:
            symbol (Optional[str]): Символ торговой пары (если None, то все пары)
            
        Returns:
            List[Dict[str, Any]]: Список открытых ордеров
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            logger.debug(f"Получены открытые ордера: {len(orders)}")
            return orders
        except Exception as e:
            logger.error(f"Ошибка при получении открытых ордеров: {e}")
            return []