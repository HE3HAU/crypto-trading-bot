#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль антикризисного управления для торгового бота.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from src.risk.risk_manager import VolatilityGuard, DrawdownProtection
from src.risk.order_manager import OrderManager

logger = logging.getLogger(__name__)

class CrisisManager:
    """
    Класс для антикризисного управления торговым ботом.
    """
    
    def __init__(self, order_manager: OrderManager, initial_capital: float = 10000.0,
                max_daily_trades: int = 10, max_daily_loss_percent: float = 5.0):
        """
        Инициализация менеджера кризисов.
        
        Args:
            order_manager (OrderManager): Менеджер ордеров
            initial_capital (float): Начальный капитал
            max_daily_trades (int): Максимальное количество сделок в день
            max_daily_loss_percent (float): Максимальный процент потерь в день
        """
        self.order_manager = order_manager
        self.initial_capital = initial_capital
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_percent = max_daily_loss_percent
        
        # Инициализация защитных механизмов
        self.volatility_guard = VolatilityGuard()
        self.drawdown_protection = DrawdownProtection(initial_capital)
        
        # Статистика торговли
        self.daily_trades = {}  # {date: count}
        self.daily_pnl = {}  # {date: pnl}
        self.trading_restricted = False
        self.trading_pause_until = None
        
        logger.info(f"Инициализирован CrisisManager с ограничениями: "
                   f"макс. {max_daily_trades} сделок в день, "
                   f"макс. {max_daily_loss_percent}% потерь в день")
    
    def update_statistics(self, current_capital: float):
        """
        Обновление статистики торговли.
        
        Args:
            current_capital (float): Текущий капитал
        """
        today = datetime.now().date()
        today_str = today.isoformat()
        
        # Получаем закрытые сегодня позиции
        closed_positions_df = self.order_manager.get_closed_positions_df()
        
        if not closed_positions_df.empty:
            today_positions = closed_positions_df[
                pd.to_datetime(closed_positions_df['exit_time']).dt.date == today
            ]
            
            # Обновляем количество сделок за день
            self.daily_trades[today_str] = len(today_positions)
            
            # Обновляем P&L за день
            self.daily_pnl[today_str] = today_positions['profit'].sum()
            
            # Проверяем ограничения
            self._check_trading_limits(today_str, current_capital)
        
        # Обновляем статус защиты от просадки
        self.drawdown_protection.update_capital(current_capital)
    
    def _check_trading_limits(self, date: str, current_capital: float):
        """
        Проверка ограничений на торговлю.
        
        Args:
            date (str): Дата в формате ISO
            current_capital (float): Текущий капитал
        """
        # Проверка на максимальное количество сделок в день
        if self.daily_trades.get(date, 0) >= self.max_daily_trades:
            logger.warning(f"Достигнуто максимальное количество сделок за день ({self.max_daily_trades})")
            self.trading_restricted = True
            self.trading_pause_until = datetime.now() + timedelta(hours=12)
            return
        
        # Проверка на максимальные потери за день
        daily_loss_percent = -self.daily_pnl.get(date, 0) / self.initial_capital * 100
        if daily_loss_percent >= self.max_daily_loss_percent:
            logger.warning(f"Достигнут максимальный процент потерь за день ({daily_loss_percent:.2f}%)")
            self.trading_restricted = True
            self.trading_pause_until = datetime.now() + timedelta(hours=24)
            return
        
        # Проверка, не истек ли срок паузы
        if self.trading_restricted and self.trading_pause_until:
            if datetime.now() > self.trading_pause_until:
                logger.info(f"Пауза в торговле закончилась")
                self.trading_restricted = False
                self.trading_pause_until = None
    
    def should_open_position(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Проверка, следует ли открывать новую позицию.
        
        Args:
            symbol (str): Символ торгового инструмента
            data (pd.DataFrame): DataFrame с рыночными данными
            
        Returns:
            bool: True, если можно открывать позицию, иначе False
        """
        # Проверка на ограничения торговли
        if self.trading_restricted:
            logger.warning(f"Торговля ограничена до {self.trading_pause_until}")
            return False
        
        # Проверка на просадку
        if not self.drawdown_protection.should_trade():
            logger.warning(f"Торговля приостановлена из-за большой просадки")
            return False
        
        # Проверка на экстремальную волатильность
        extreme_volatility, _, _ = self.volatility_guard.is_extreme_volatility(data)
        if extreme_volatility:
            logger.warning(f"Торговля приостановлена из-за экстремальной волатильности")
            return False
        
        # Проверка, что у нас нет уже открытой позиции по этому инструменту
        if self.order_manager.get_position(symbol):
            logger.warning(f"Уже есть открытая позиция для {symbol}")
            return False
        
        return True
    
    def adjust_position_for_risk(self, symbol: str, base_amount: float, 
                               price: float, data: pd.DataFrame) -> float:
        """
        Корректировка размера позиции с учетом рисков.
        
        Args:
            symbol (str): Символ торгового инструмента
            base_amount (float): Базовый размер позиции
            price (float): Текущая цена
            data (pd.DataFrame): DataFrame с рыночными данными
            
        Returns:
            float: Скорректированный размер позиции
        """
        # Корректировка на основе просадки
        amount = self.drawdown_protection.adjust_position_size(base_amount)
        
        # Корректировка на основе волатильности
        amount = self.volatility_guard.adjust_position_size_for_volatility(amount, data)
        
        logger.info(f"Скорректирован размер позиции для {symbol}: {base_amount} -> {amount}")
        
        return amount
    
    def calculate_crisis_adjusted_stops(self, symbol: str, entry_price: float, 
                                      side: str, base_stop_percent: float, 
                                      data: pd.DataFrame) -> Tuple[float, float]:
        """
        Расчет стоп-лосса и тейк-профита с учетом рыночных условий.
        
        Args:
            symbol (str): Символ торгового инструмента
            entry_price (float): Цена входа
            side (str): Сторона сделки ('buy', 'sell')
            base_stop_percent (float): Базовый процент для стоп-лосса
            data (pd.DataFrame): DataFrame с рыночными данными
            
        Returns:
            Tuple[float, float]: (стоп-лосс, тейк-профит)
        """
        # Корректировка стоп-лосса на основе волатильности
        stop_loss = self.volatility_guard.adjust_stop_loss_for_volatility(
            entry_price, base_stop_percent, data, side
        )
        
        # Расчет тейк-профита с R/R = 2
        risk = abs(entry_price - stop_loss)
        if side == 'buy':
            take_profit = entry_price + risk * 2
        else:  # 'sell'
            take_profit = entry_price - risk * 2
        
        logger.info(f"Рассчитаны кризисные уровни для {symbol}: "
                   f"SL: {stop_loss}, TP: {take_profit}")
        
        return stop_loss, take_profit
    
    def close_positions_in_crisis(self, crisis_detected: bool, reason: str):
        """
        Закрытие всех позиций в случае кризиса.
        
        Args:
            crisis_detected (bool): Флаг обнаружения кризиса
            reason (str): Причина кризиса
        """
        if not crisis_detected:
            return
        
        logger.warning(f"Обнаружен кризис: {reason}. Закрытие всех позиций.")
        
        # Закрываем все открытые позиции
        for symbol in list(self.order_manager.positions.keys()):
            logger.info(f"Экстренное закрытие позиции {symbol} из-за кризиса")
            self.order_manager.close_position(symbol)
        
        # Устанавливаем паузу в торговле
        self.trading_restricted = True
        self.trading_pause_until = datetime.now() + timedelta(hours=24)
    
    def detect_market_crash(self, data: pd.DataFrame, threshold: float = -10.0) -> Tuple[bool, str]:
        """
        Обнаружение краха рынка.
        
        Args:
            data (pd.DataFrame): DataFrame с рыночными данными
            threshold (float): Пороговое значение для обнаружения краха (отрицательный процент)
            
        Returns:
            Tuple[bool, str]: (флаг обнаружения, причина)
        """
        # Проверяем падение цены за короткий период
        if len(data) < 2:
            return False, ""
        
        # Расчет процентного изменения цены
        price_change = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        
        if price_change <= threshold:
            reason = f"Крах рынка: падение цены на {price_change:.2f}% за короткий период"
            logger.warning(reason)
            return True, reason
        
        return False, ""
    
    def detect_extreme_conditions(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Обнаружение экстремальных рыночных условий.
        
        Args:
            data (pd.DataFrame): DataFrame с рыночными данными
            
        Returns:
            Tuple[bool, str]: (флаг обнаружения, причина)
        """
        # Проверка на экстремальную волатильность
        extreme_volatility, current_volatility, avg_volatility = self.volatility_guard.is_extreme_volatility(data)
        
        if extreme_volatility:
            reason = f"Экстремальная волатильность: {current_volatility:.2f}% (в {current_volatility/avg_volatility:.2f} раз выше средней)"
            return True, reason
        
        # Проверка на крах рынка
        market_crash, crash_reason = self.detect_market_crash(data)
        if market_crash:
            return True, crash_reason
        
        # Проверка на необычно высокий объем торгов
        if 'volume' in data.columns:
            avg_volume = data['volume'].mean()
            last_volume = data['volume'].iloc[-1]
            
            if last_volume > avg_volume * 5:
                reason = f"Необычно высокий объем торгов: {last_volume:.2f} (в {last_volume/avg_volume:.2f} раз выше среднего)"
                logger.warning(reason)
                return True, reason
        
        return False, ""