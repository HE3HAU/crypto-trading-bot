#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для управления ордерами и позициями.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from src.api.base_exchange import BaseExchange
from src.risk.risk_manager import StopLossCalculator, TakeProfitCalculator

logger = logging.getLogger(__name__)

class Order:
    """
    Класс для представления торгового ордера.
    """
    
    def __init__(self, symbol: str, order_type: str, side: str, amount: float, 
                price: Optional[float] = None, stop_loss: Optional[float] = None, 
                take_profit: Optional[float] = None, order_id: Optional[str] = None):
        """
        Инициализация ордера.
        
        Args:
            symbol (str): Символ торгового инструмента
            order_type (str): Тип ордера ('market', 'limit')
            side (str): Сторона ордера ('buy', 'sell')
            amount (float): Количество актива
            price (Optional[float]): Цена для лимитного ордера
            stop_loss (Optional[float]): Уровень стоп-лосса
            take_profit (Optional[float]): Уровень тейк-профита
            order_id (Optional[str]): ID ордера (если известен)
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.amount = amount
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_id = order_id
        self.created_at = datetime.now()
        self.executed_at = None
        self.status = "created"
        
        logger.info(f"Создан ордер: {self.side} {self.amount} {self.symbol} "
                   f"по {'market' if self.price is None else self.price}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование ордера в словарь.
        
        Returns:
            Dict[str, Any]: Словарь с информацией об ордере
        """
        return {
            'symbol': self.symbol,
            'order_type': self.order_type,
            'side': self.side,
            'amount': self.amount,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'order_id': self.order_id,
            'created_at': self.created_at,
            'executed_at': self.executed_at,
            'status': self.status
        }


class Position:
    """
    Класс для представления открытой позиции.
    """
    
    def __init__(self, symbol: str, side: str, amount: float, entry_price: float, 
                entry_time: datetime, stop_loss: Optional[float] = None, 
                take_profit: Optional[float] = None, trailing_stop: bool = False,
                strategy_name: str = "unknown"):
        """
        Инициализация позиции.
        
        Args:
            symbol (str): Символ торгового инструмента
            side (str): Сторона позиции ('buy', 'sell')
            amount (float): Количество актива
            entry_price (float): Цена входа
            entry_time (datetime): Время входа
            stop_loss (Optional[float]): Уровень стоп-лосса
            take_profit (Optional[float]): Уровень тейк-профита
            trailing_stop (bool): Использовать трейлинг-стоп
            strategy_name (str): Название стратегии, создавшей позицию
        """
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.strategy_name = strategy_name
        self.exit_price = None
        self.exit_time = None
        self.profit = None
        self.profit_percent = None
        self.status = "open"
        
        logger.info(f"Открыта позиция: {self.side} {self.amount} {self.symbol} по {self.entry_price} "
                   f"(SL: {self.stop_loss}, TP: {self.take_profit})")
    
    def close(self, exit_price: float, exit_time: datetime):
        """
        Закрытие позиции.
        
        Args:
            exit_price (float): Цена выхода
            exit_time (datetime): Время выхода
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        # Расчет прибыли
        if self.side == 'buy':
            self.profit = (exit_price - self.entry_price) * self.amount
            self.profit_percent = (exit_price / self.entry_price - 1) * 100
        else:  # 'sell'
            self.profit = (self.entry_price - exit_price) * self.amount
            self.profit_percent = (self.entry_price / exit_price - 1) * 100
        
        self.status = "closed"
        
        logger.info(f"Закрыта позиция: {self.side} {self.amount} {self.symbol} "
                   f"по {self.exit_price}, прибыль: {self.profit:.2f} ({self.profit_percent:.2f}%)")
    
    def update_stop_loss(self, new_stop: float):
        """
        Обновление уровня стоп-лосса.
        
        Args:
            new_stop (float): Новый уровень стоп-лосса
        """
        if self.side == 'buy' and new_stop > self.stop_loss:
            self.stop_loss = new_stop
            logger.info(f"Обновлен стоп-лосс для {self.symbol}: {self.stop_loss}")
        elif self.side == 'sell' and new_stop < self.stop_loss:
            self.stop_loss = new_stop
            logger.info(f"Обновлен стоп-лосс для {self.symbol}: {self.stop_loss}")
    
    def check_stop_loss(self, current_price: float) -> bool:
        """
        Проверка срабатывания стоп-лосса.
        
        Args:
            current_price (float): Текущая цена
            
        Returns:
            bool: True, если стоп-лосс сработал, иначе False
        """
        if self.stop_loss is None:
            return False
        
        if self.side == 'buy' and current_price <= self.stop_loss:
            logger.info(f"Сработал стоп-лосс для {self.symbol}: {current_price} <= {self.stop_loss}")
            return True
        elif self.side == 'sell' and current_price >= self.stop_loss:
            logger.info(f"Сработал стоп-лосс для {self.symbol}: {current_price} >= {self.stop_loss}")
            return True
        
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """
        Проверка срабатывания тейк-профита.
        
        Args:
            current_price (float): Текущая цена
            
        Returns:
            bool: True, если тейк-профит сработал, иначе False
        """
        if self.take_profit is None:
            return False
        
        if self.side == 'buy' and current_price >= self.take_profit:
            logger.info(f"Сработал тейк-профит для {self.symbol}: {current_price} >= {self.take_profit}")
            return True
        elif self.side == 'sell' and current_price <= self.take_profit:
            logger.info(f"Сработал тейк-профит для {self.symbol}: {current_price} <= {self.take_profit}")
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование позиции в словарь.
        
        Returns:
            Dict[str, Any]: Словарь с информацией о позиции
        """
        return {
            'symbol': self.symbol,
            'side': self.side,
            'amount': self.amount,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'strategy_name': self.strategy_name,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'profit': self.profit,
            'profit_percent': self.profit_percent,
            'status': self.status
        }


class OrderManager:
    """
    Класс для управления ордерами и позициями.
    """
    
    def __init__(self, exchange: BaseExchange):
        """
        Инициализация менеджера ордеров.
        
        Args:
            exchange (BaseExchange): Экземпляр биржи
        """
        self.exchange = exchange
        self.orders = []  # Список всех ордеров
        self.positions = {}  # Словарь открытых позиций {symbol: Position}
        self.closed_positions = []  # Список закрытых позиций
        self.stop_loss_calculator = StopLossCalculator()
        self.take_profit_calculator = TakeProfitCalculator()
        
        logger.info(f"Инициализирован OrderManager для {exchange.exchange_name}")
    
    def create_market_order(self, symbol: str, side: str, amount: float, 
                           stop_loss: Optional[float] = None, 
                           take_profit: Optional[float] = None,
                           strategy_name: str = "unknown") -> Optional[Order]:
        """
        Создание рыночного ордера.
        
        Args:
            symbol (str): Символ торгового инструмента
            side (str): Сторона ордера ('buy', 'sell')
            amount (float): Количество актива
            stop_loss (Optional[float]): Уровень стоп-лосса
            take_profit (Optional[float]): Уровень тейк-профита
            strategy_name (str): Название стратегии
            
        Returns:
            Optional[Order]: Созданный ордер или None в случае ошибки
        """
        try:
            # Создаем ордер
            order = Order(symbol, 'market', side, amount, None, stop_loss, take_profit)
            
            # Выполняем ордер на бирже
            exchange_order = self.exchange.create_order(symbol, 'market', side, amount)
            
            if not exchange_order:
                logger.error(f"Не удалось создать рыночный ордер: {symbol}, {side}, {amount}")
                return None
            
            # Обновляем информацию об ордере
            order.order_id = exchange_order.get('id')
            order.price = exchange_order.get('price') or exchange_order.get('average')
            order.executed_at = datetime.now()
            order.status = "filled"
            
            self.orders.append(order)
            
            # Создаем позицию
            if order.price:
                position = Position(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    entry_price=order.price,
                    entry_time=order.executed_at,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_name=strategy_name
                )
                
                self.positions[symbol] = position
            
            logger.info(f"Создан и выполнен рыночный ордер: {side} {amount} {symbol} по {order.price}")
            
            return order
            
        except Exception as e:
            logger.error(f"Ошибка при создании рыночного ордера: {e}")
            return None
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float,
                          stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None,
                          strategy_name: str = "unknown") -> Optional[Order]:
        """
        Создание лимитного ордера.
        
        Args:
            symbol (str): Символ торгового инструмента
            side (str): Сторона ордера ('buy', 'sell')
            amount (float): Количество актива
            price (float): Цена для ордера
            stop_loss (Optional[float]): Уровень стоп-лосса
            take_profit (Optional[float]): Уровень тейк-профита
            strategy_name (str): Название стратегии
            
        Returns:
            Optional[Order]: Созданный ордер или None в случае ошибки
        """
        try:
            # Создаем ордер
            order = Order(symbol, 'limit', side, amount, price, stop_loss, take_profit)
            
            # Выполняем ордер на бирже
            exchange_order = self.exchange.create_order(symbol, 'limit', side, amount, price)
            
            if not exchange_order:
                logger.error(f"Не удалось создать лимитный ордер: {symbol}, {side}, {amount}, {price}")
                return None
            
            # Обновляем информацию об ордере
            order.order_id = exchange_order.get('id')
            order.status = "open"
            
            self.orders.append(order)
            
            logger.info(f"Создан лимитный ордер: {side} {amount} {symbol} по {price}")
            
            return order
            
        except Exception as e:
            logger.error(f"Ошибка при создании лимитного ордера: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Отмена ордера.
        
        Args:
            order_id (str): ID ордера
            symbol (str): Символ торгового инструмента
            
        Returns:
            bool: True, если ордер успешно отменен, иначе False
        """
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            
            if result:
                # Обновляем статус ордера в списке
                for order in self.orders:
                    if order.order_id == order_id:
                        order.status = "canceled"
                        logger.info(f"Ордер {order_id} отменен")
                        return True
            
            logger.error(f"Не удалось отменить ордер {order_id}")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при отмене ордера {order_id}: {e}")
            return False
    
    def close_position(self, symbol: str) -> bool:
        """
        Закрытие позиции.
        
        Args:
            symbol (str): Символ торгового инструмента
            
        Returns:
            bool: True, если позиция успешно закрыта, иначе False
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"Нет открытой позиции для {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Создаем ордер для закрытия позиции
            close_side = 'sell' if position.side == 'buy' else 'buy'
            
            # Получаем текущую цену
            ticker = self.exchange.get_ticker(symbol)
            current_price = ticker.get('last')
            
            if not current_price:
                logger.error(f"Не удалось получить текущую цену для {symbol}")
                return False
            
            # Создаем рыночный ордер для закрытия
            order = self.create_market_order(symbol, close_side, position.amount)
            
            if order and order.price:
                # Закрываем позицию
                position.close(order.price, order.executed_at)
                
                # Перемещаем позицию в список закрытых
                self.closed_positions.append(position)
                del self.positions[symbol]
                
                logger.info(f"Позиция для {symbol} закрыта по {order.price}")
                return True
            
            logger.error(f"Не удалось закрыть позицию для {symbol}")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при закрытии позиции для {symbol}: {e}")
            return False
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Обновление позиций на основе текущих цен и проверка на срабатывание стоп-лосса/тейк-профита.
        
        Args:
            current_prices (Dict[str, float]): Словарь текущих цен {symbol: price}
        """
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Проверка на срабатывание стоп-лосса
            if position.check_stop_loss(current_price):
                logger.info(f"Закрытие позиции {symbol} по стоп-лоссу")
                self.close_position(symbol)
                continue
            
            # Проверка на срабатывание тейк-профита
            if position.check_take_profit(current_price):
                logger.info(f"Закрытие позиции {symbol} по тейк-профиту")
                self.close_position(symbol)
                continue
            
            # Обновление трейлинг-стопа, если он включен
            if position.trailing_stop and position.stop_loss is not None:
                if position.side == 'buy' and current_price > position.entry_price:
                    # Расчет нового стоп-лосса для длинной позиции
                    trailing_distance = position.entry_price - position.stop_loss
                    new_stop = current_price - trailing_distance
                    
                    if new_stop > position.stop_loss:
                        position.update_stop_loss(new_stop)
                        logger.info(f"Обновлен трейлинг-стоп для {symbol}: {new_stop}")
                
                elif position.side == 'sell' and current_price < position.entry_price:
                    # Расчет нового стоп-лосса для короткой позиции
                    trailing_distance = position.stop_loss - position.entry_price
                    new_stop = current_price + trailing_distance
                    
                    if new_stop < position.stop_loss:
                        position.update_stop_loss(new_stop)
                        logger.info(f"Обновлен трейлинг-стоп для {symbol}: {new_stop}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Получение информации о позиции по символу.
        
        Args:
            symbol (str): Символ торгового инструмента
            
        Returns:
            Optional[Position]: Позиция или None, если нет открытой позиции
        """
        return self.positions.get(symbol)
    
    def get_positions_df(self) -> pd.DataFrame:
        """
        Получение DataFrame с открытыми позициями.
        
        Returns:
            pd.DataFrame: DataFrame с открытыми позициями
        """
        positions_data = [position.to_dict() for position in self.positions.values()]
        if not positions_data:
            return pd.DataFrame()
        return pd.DataFrame(positions_data)
    
    def get_closed_positions_df(self) -> pd.DataFrame:
        """
        Получение DataFrame с закрытыми позициями.
        
        Returns:
            pd.DataFrame: DataFrame с закрытыми позициями
        """
        positions_data = [position.to_dict() for position in self.closed_positions]
        if not positions_data:
            return pd.DataFrame()
        return pd.DataFrame(positions_data)
    
    def get_orders_df(self) -> pd.DataFrame:
        """
        Получение DataFrame с ордерами.
        
        Returns:
            pd.DataFrame: DataFrame с ордерами
        """
        orders_data = [order.to_dict() for order in self.orders]
        if not orders_data:
            return pd.DataFrame()
        return pd.DataFrame(orders_data)