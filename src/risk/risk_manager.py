#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для управления капиталом и рисками при торговле.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Класс для расчета размера позиции на основе различных стратегий управления капиталом.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Инициализация класса управления размером позиции.
        
        Args:
            initial_capital (float): Начальный капитал
        """
        self.initial_capital = initial_capital
        logger.info(f"Инициализирован PositionSizer с капиталом {initial_capital}")
    
    def fixed_size(self, available_capital: float, price: float, 
                  amount: float) -> Tuple[float, float]:
        """
        Стратегия с фиксированным размером позиции.
        
        Args:
            available_capital (float): Доступный капитал
            price (float): Текущая цена актива
            amount (float): Количество актива для покупки/продажи
            
        Returns:
            Tuple[float, float]: (объем позиции в единицах актива, объем позиции в деньгах)
        """
        cost = price * amount
        
        # Проверка, что хватает капитала
        if cost > available_capital:
            # Уменьшаем размер позиции до доступного капитала
            amount = available_capital / price
            cost = available_capital
            logger.warning(f"Недостаточно капитала. Размер позиции уменьшен до {amount}")
        
        return amount, cost
    
    def fixed_risk(self, available_capital: float, price: float, 
                  risk_per_trade: float, stop_loss_percent: float) -> Tuple[float, float]:
        """
        Стратегия с фиксированным риском на сделку.
        
        Args:
            available_capital (float): Доступный капитал
            price (float): Текущая цена актива
            risk_per_trade (float): Риск на сделку в процентах от капитала
            stop_loss_percent (float): Процент стоп-лосса от цены входа
            
        Returns:
            Tuple[float, float]: (объем позиции в единицах актива, объем позиции в деньгах)
        """
        # Максимальный риск в валюте
        max_risk = available_capital * risk_per_trade / 100
        
        # Расчет размера позиции
        # Если стоп-лосс 2% и мы хотим рисковать 1% капитала,
        # то размер позиции = (капитал * риск%) / стоп-лосс%
        position_size_currency = max_risk / (stop_loss_percent / 100)
        position_size_units = position_size_currency / price
        
        # Проверка, что хватает капитала
        if position_size_currency > available_capital:
            position_size_currency = available_capital
            position_size_units = position_size_currency / price
            logger.warning(f"Недостаточно капитала. Размер позиции уменьшен до {position_size_units}")
        
        logger.info(f"Рассчитан размер позиции с фиксированным риском: {position_size_units} единиц, "
                   f"{position_size_currency} в валюте (риск: {max_risk})")
        
        return position_size_units, position_size_currency
    
    def kelly_criterion(self, available_capital: float, price: float,
                        win_rate: float, win_loss_ratio: float) -> Tuple[float, float]:
        """
        Стратегия размера позиции на основе критерия Келли.
        
        Args:
            available_capital (float): Доступный капитал
            price (float): Текущая цена актива
            win_rate (float): Процент успешных сделок (от 0 до 1)
            win_loss_ratio (float): Отношение средней прибыли к среднему убытку
            
        Returns:
            Tuple[float, float]: (объем позиции в единицах актива, объем позиции в деньгах)
        """
        # Расчет по формуле Келли
        # f* = (p * b - q) / b
        # где p - вероятность успеха (win_rate),
        # q - вероятность провала (1 - win_rate),
        # b - отношение среднего выигрыша к среднему проигрышу (win_loss_ratio)
        
        # Защита от некорректных входных данных
        if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
            logger.warning("Некорректные входные данные для критерия Келли")
            return 0, 0
        
        q = 1 - win_rate
        kelly_fraction = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Обычно используется половина или треть от полного размера Келли
        # для более консервативного управления капиталом
        conservative_fraction = kelly_fraction / 2
        
        # Ограничиваем размер позиции, если критерий Келли дает отрицательное значение
        # или слишком большое значение
        if conservative_fraction <= 0:
            logger.warning(f"Критерий Келли дал отрицательное значение: {kelly_fraction}. Позиция не открывается.")
            return 0, 0
        elif conservative_fraction > 0.5:
            conservative_fraction = 0.5
            logger.warning(f"Критерий Келли ограничен до 50% капитала")
        
        position_size_currency = available_capital * conservative_fraction
        position_size_units = position_size_currency / price
        
        logger.info(f"Рассчитан размер позиции по критерию Келли: {position_size_units} единиц, "
                   f"{position_size_currency} в валюте (фракция Келли: {kelly_fraction:.4f})")
        
        return position_size_units, position_size_currency
    
    def martingale(self, available_capital: float, price: float, 
                  base_amount: float, consecutive_losses: int = 0, 
                  max_multiplier: int = 8) -> Tuple[float, float]:
        """
        Стратегия Мартингейл - увеличивает размер позиции после убыточной сделки.
        
        Args:
            available_capital (float): Доступный капитал
            price (float): Текущая цена актива
            base_amount (float): Базовый размер позиции
            consecutive_losses (int): Количество последовательных убыточных сделок
            max_multiplier (int): Максимальный множитель для увеличения позиции
            
        Returns:
            Tuple[float, float]: (объем позиции в единицах актива, объем позиции в деньгах)
        """
        # Расчет множителя: каждый проигрыш удваивает ставку
        multiplier = min(2 ** consecutive_losses, max_multiplier)
        amount = base_amount * multiplier
        cost = price * amount
        
        # Проверка, что хватает капитала
        if cost > available_capital:
            amount = available_capital / price
            cost = available_capital
            logger.warning(f"Недостаточно капитала для полного Мартингейла. Размер позиции ограничен до {amount}")
        
        logger.info(f"Рассчитан размер позиции по Мартингейлу: {amount} единиц, "
                   f"{cost} в валюте (множитель: {multiplier})")
        
        return amount, cost
    
    def anti_martingale(self, available_capital: float, price: float, 
                       base_amount: float, consecutive_wins: int = 0, 
                       max_multiplier: int = 8) -> Tuple[float, float]:
        """
        Стратегия Анти-Мартингейл - увеличивает размер позиции после прибыльной сделки.
        
        Args:
            available_capital (float): Доступный капитал
            price (float): Текущая цена актива
            base_amount (float): Базовый размер позиции
            consecutive_wins (int): Количество последовательных выигрышных сделок
            max_multiplier (int): Максимальный множитель для увеличения позиции
            
        Returns:
            Tuple[float, float]: (объем позиции в единицах актива, объем позиции в деньгах)
        """
        # Расчет множителя: каждый выигрыш увеличивает ставку
        multiplier = min(1 + consecutive_wins * 0.5, max_multiplier)
        amount = base_amount * multiplier
        cost = price * amount
        
        # Проверка, что хватает капитала
        if cost > available_capital:
            amount = available_capital / price
            cost = available_capital
            logger.warning(f"Недостаточно капитала для полного Анти-Мартингейла. Размер позиции ограничен до {amount}")
        
        logger.info(f"Рассчитан размер позиции по Анти-Мартингейлу: {amount} единиц, "
                   f"{cost} в валюте (множитель: {multiplier})")
        
        return amount, cost
    
    def percent_of_equity(self, available_capital: float, price: float, 
                         percent: float = 5.0) -> Tuple[float, float]:
        """
        Стратегия с фиксированным процентом от капитала.
        
        Args:
            available_capital (float): Доступный капитал
            price (float): Текущая цена актива
            percent (float): Процент от капитала на сделку
            
        Returns:
            Tuple[float, float]: (объем позиции в единицах актива, объем позиции в деньгах)
        """
        cost = available_capital * percent / 100
        amount = cost / price
        
        logger.info(f"Рассчитан размер позиции как {percent}% от капитала: {amount} единиц, {cost} в валюте")
        
        return amount, cost


class StopLossCalculator:
    """
    Класс для расчета уровней стоп-лосса на основе различных методик.
    """
    
    def __init__(self):
        """
        Инициализация калькулятора стоп-лоссов.
        """
        logger.info(f"Инициализирован StopLossCalculator")
    
    def fixed_percent(self, entry_price: float, side: str, 
                     percent: float = 2.0) -> float:
        """
        Расчет стоп-лосса как фиксированного процента от цены входа.
        
        Args:
            entry_price (float): Цена входа в позицию
            side (str): Сторона сделки ('buy' или 'sell')
            percent (float): Процент для стоп-лосса
            
        Returns:
            float: Уровень стоп-лосса
        """
        if side == 'buy':
            stop_level = entry_price * (1 - percent / 100)
        else:  # 'sell'
            stop_level = entry_price * (1 + percent / 100)
        
        logger.info(f"Рассчитан стоп-лосс ({percent}%): {stop_level} для {side} по цене {entry_price}")
        
        return stop_level
    
    def atr_based(self, entry_price: float, side: str, atr: float, 
                 multiplier: float = 2.0) -> float:
        """
        Расчет стоп-лосса на основе ATR (Average True Range).
        
        Args:
            entry_price (float): Цена входа в позицию
            side (str): Сторона сделки ('buy' или 'sell')
            atr (float): Значение ATR
            multiplier (float): Множитель ATR
            
        Returns:
            float: Уровень стоп-лосса
        """
        if side == 'buy':
            stop_level = entry_price - atr * multiplier
        else:  # 'sell'
            stop_level = entry_price + atr * multiplier
        
        logger.info(f"Рассчитан стоп-лосс на основе ATR: {stop_level} для {side} по цене {entry_price} (ATR: {atr})")
        
        return stop_level
    
    def support_resistance(self, entry_price: float, side: str, 
                          support_level: float, resistance_level: float) -> float:
        """
        Расчет стоп-лосса на основе уровней поддержки и сопротивления.
        
        Args:
            entry_price (float): Цена входа в позицию
            side (str): Сторона сделки ('buy' или 'sell')
            support_level (float): Уровень поддержки
            resistance_level (float): Уровень сопротивления
            
        Returns:
            float: Уровень стоп-лосса
        """
        if side == 'buy':
            stop_level = support_level
            # Проверяем, что стоп не слишком далеко (не более 5% от цены входа)
            max_distance = entry_price * 0.05
            if entry_price - stop_level > max_distance:
                stop_level = entry_price - max_distance
                logger.warning(f"Уровень поддержки слишком далеко, стоп-лосс ограничен до {stop_level}")
        else:  # 'sell'
            stop_level = resistance_level
            # Проверяем, что стоп не слишком далеко (не более 5% от цены входа)
            max_distance = entry_price * 0.05
            if stop_level - entry_price > max_distance:
                stop_level = entry_price + max_distance
                logger.warning(f"Уровень сопротивления слишком далеко, стоп-лосс ограничен до {stop_level}")
        
        logger.info(f"Рассчитан стоп-лосс на основе уровней: {stop_level} для {side} по цене {entry_price}")
        
        return stop_level
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                               current_stop: float, side: str, activation_percent: float = 1.0,
                               trail_percent: float = 0.5) -> float:
        """
        Расчет трейлинг-стопа.
        
        Args:
            entry_price (float): Цена входа в позицию
            current_price (float): Текущая цена
            current_stop (float): Текущий уровень стоп-лосса
            side (str): Сторона сделки ('buy' или 'sell')
            activation_percent (float): Процент активации трейлинг-стопа
            trail_percent (float): Процент следования за ценой
            
        Returns:
            float: Новый уровень стоп-лосса
        """
        if side == 'buy':
            # Проверяем, активирован ли трейлинг-стоп
            if current_price >= entry_price * (1 + activation_percent / 100):
                # Рассчитываем новый стоп
                new_stop = current_price * (1 - trail_percent / 100)
                # Перемещаем стоп только если он выше текущего
                if new_stop > current_stop:
                    logger.info(f"Обновлен трейлинг-стоп: {new_stop} (был: {current_stop}) при цене {current_price}")
                    return new_stop
        else:  # 'sell'
            # Проверяем, активирован ли трейлинг-стоп
            if current_price <= entry_price * (1 - activation_percent / 100):
                # Рассчитываем новый стоп
                new_stop = current_price * (1 + trail_percent / 100)
                # Перемещаем стоп только если он ниже текущего
                if new_stop < current_stop:
                    logger.info(f"Обновлен трейлинг-стоп: {new_stop} (был: {current_stop}) при цене {current_price}")
                    return new_stop
        
        # Если трейлинг не активирован или новый стоп не лучше текущего, возвращаем текущий
        return current_stop


class TakeProfitCalculator:
    """
    Класс для расчета уровней тейк-профита на основе различных методик.
    """
    
    def __init__(self):
        """
        Инициализация калькулятора тейк-профитов.
        """
        logger.info(f"Инициализирован TakeProfitCalculator")
    
    def fixed_percent(self, entry_price: float, side: str, 
                     percent: float = 4.0) -> float:
        """
        Расчет тейк-профита как фиксированного процента от цены входа.
        
        Args:
            entry_price (float): Цена входа в позицию
            side (str): Сторона сделки ('buy' или 'sell')
            percent (float): Процент для тейк-профита
            
        Returns:
            float: Уровень тейк-профита
        """
        if side == 'buy':
            take_level = entry_price * (1 + percent / 100)
        else:  # 'sell'
            take_level = entry_price * (1 - percent / 100)
        
        logger.info(f"Рассчитан тейк-профит ({percent}%): {take_level} для {side} по цене {entry_price}")
        
        return take_level
    
    def risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                         side: str, ratio: float = 2.0) -> float:
        """
        Расчет тейк-профита на основе соотношения риск/прибыль.
        
        Args:
            entry_price (float): Цена входа в позицию
            stop_loss (float): Уровень стоп-лосса
            side (str): Сторона сделки ('buy' или 'sell')
            ratio (float): Соотношение риск/прибыль
            
        Returns:
            float: Уровень тейк-профита
        """
        if side == 'buy':
            risk = entry_price - stop_loss
            take_level = entry_price + (risk * ratio)
        else:  # 'sell'
            risk = stop_loss - entry_price
            take_level = entry_price - (risk * ratio)
        
        logger.info(f"Рассчитан тейк-профит на основе R/R {ratio}: {take_level} для {side} по цене {entry_price}")
        
        return take_level
    
    def fibonacci_extension(self, entry_price: float, recent_low: float, 
                           recent_high: float, side: str, level: float = 1.618) -> float:
        """
        Расчет тейк-профита на основе уровней расширения Фибоначчи.
        
        Args:
            entry_price (float): Цена входа в позицию
            recent_low (float): Недавний минимум
            recent_high (float): Недавний максимум
            side (str): Сторона сделки ('buy' или 'sell')
            level (float): Уровень расширения Фибоначчи
            
        Returns:
            float: Уровень тейк-профита
        """
        range_size = recent_high - recent_low
        
        if side == 'buy':
            take_level = recent_high + (range_size * level)
        else:  # 'sell'
            take_level = recent_low - (range_size * level)
        
        logger.info(f"Рассчитан тейк-профит на основе Фибоначчи {level}: {take_level} для {side} по цене {entry_price}")
        
        return take_level
    
    def multiple_targets(self, entry_price: float, stop_loss: float, 
                        side: str, ratios: List[float] = [1.0, 2.0, 3.0]) -> List[float]:
        """
        Расчет нескольких уровней тейк-профита.
        
        Args:
            entry_price (float): Цена входа в позицию
            stop_loss (float): Уровень стоп-лосса
            side (str): Сторона сделки ('buy' или 'sell')
            ratios (List[float]): Список соотношений риск/прибыль для разных уровней
            
        Returns:
            List[float]: Список уровней тейк-профита
        """
        targets = []
        
        if side == 'buy':
            risk = entry_price - stop_loss
            for ratio in ratios:
                take_level = entry_price + (risk * ratio)
                targets.append(take_level)
        else:  # 'sell'
            risk = stop_loss - entry_price
            for ratio in ratios:
                take_level = entry_price - (risk * ratio)
                targets.append(take_level)
        
        logger.info(f"Рассчитаны несколько тейк-профитов: {targets} для {side} по цене {entry_price}")
        
        return targets


class DrawdownProtection:
    """
    Класс для защиты от чрезмерных просадок.
    """
    
    def __init__(self, initial_capital: float, max_drawdown_percent: float = 10.0,
                reduce_position_at: float = 5.0, stop_trading_at: float = 15.0):
        """
        Инициализация защиты от просадок.
        
        Args:
            initial_capital (float): Начальный капитал
            max_drawdown_percent (float): Максимально допустимый процент просадки
            reduce_position_at (float): Процент просадки, при котором уменьшать размер позиции
            stop_trading_at (float): Процент просадки, при котором полностью останавливать торговлю
        """
        self.initial_capital = initial_capital
        self.max_drawdown_percent = max_drawdown_percent
        self.reduce_position_at = reduce_position_at
        self.stop_trading_at = stop_trading_at
        self.peak_capital = initial_capital
        self.current_drawdown_percent = 0.0
        self.position_size_multiplier = 1.0
        
        logger.info(f"Инициализирована защита от просадок: "
                   f"макс. просадка {max_drawdown_percent}%, "
                   f"уменьшение позиции при {reduce_position_at}%, "
                   f"стоп торговли при {stop_trading_at}%")
    
    def update_capital(self, current_capital: float) -> float:
        """
        Обновление текущего капитала и расчет просадки.
        
        Args:
            current_capital (float): Текущий капитал
            
        Returns:
            float: Текущий процент просадки
        """
        # Обновляем пиковый капитал, если текущий капитал выше
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Расчет текущей просадки в процентах
        self.current_drawdown_percent = (1 - current_capital / self.peak_capital) * 100
        
        # Расчет множителя размера позиции на основе текущей просадки
        if self.current_drawdown_percent >= self.stop_trading_at:
            self.position_size_multiplier = 0.0  # Полная остановка торговли
            logger.warning(f"Торговля остановлена: просадка {self.current_drawdown_percent:.2f}% "
                          f"превышает лимит {self.stop_trading_at}%")
        elif self.current_drawdown_percent >= self.reduce_position_at:
            # Линейное уменьшение размера позиции от 100% до 0%
            reduction_range = self.stop_trading_at - self.reduce_position_at
            reduction_factor = (self.current_drawdown_percent - self.reduce_position_at) / reduction_range
            self.position_size_multiplier = 1.0 - reduction_factor
            logger.warning(f"Размер позиции уменьшен до {self.position_size_multiplier:.2f} "
                          f"из-за просадки {self.current_drawdown_percent:.2f}%")
        else:
            self.position_size_multiplier = 1.0  # Нормальный режим торговли
        
        return self.current_drawdown_percent
    
    def adjust_position_size(self, position_size: float) -> float:
        """
        Корректировка размера позиции с учетом текущей просадки.
        
        Args:
            position_size (float): Изначальный размер позиции
            
        Returns:
            float: Скорректированный размер позиции
        """
        adjusted_size = position_size * self.position_size_multiplier
        
        if adjusted_size < position_size:
            logger.info(f"Размер позиции скорректирован с {position_size} до {adjusted_size} "
                       f"из-за просадки {self.current_drawdown_percent:.2f}%")
        
        return adjusted_size
    
    def should_trade(self) -> bool:
        """
        Проверка, следует ли продолжать торговлю с учетом текущей просадки.
        
        Returns:
            bool: True, если торговля разрешена, False в противном случае
        """
        return self.position_size_multiplier > 0.0


class VolatilityGuard:
    """
    Класс для управления торговлей в условиях высокой волатильности.
    """
    
    def __init__(self, lookback_period: int = 20, volatility_threshold: float = 2.5,
                extreme_volatility_threshold: float = 4.0):
        """
        Инициализация защиты от высокой волатильности.
        
        Args:
            lookback_period (int): Период для расчета волатильности
            volatility_threshold (float): Порог высокой волатильности (множитель от средней)
            extreme_volatility_threshold (float): Порог экстремальной волатильности (множитель от средней)
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.extreme_volatility_threshold = extreme_volatility_threshold
        
        logger.info(f"Инициализирована защита от волатильности: "
                   f"период {lookback_period}, "
                   f"порог высокой волатильности x{volatility_threshold}, "
                   f"порог экстремальной волатильности x{extreme_volatility_threshold}")
    
    def calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """
        Расчет текущей волатильности.
        
        Args:
            price_data (pd.DataFrame): DataFrame с ценовыми данными
            
        Returns:
            float: Текущее значение волатильности
        """
        # Расчет дневных изменений цены в процентах
        returns = price_data['close'].pct_change().dropna()
        
        # Расчет стандартного отклонения (волатильности)
        volatility = returns.std() * 100
        
        logger.debug(f"Рассчитана волатильность: {volatility:.2f}%")
        
        return volatility
    
    def is_high_volatility(self, price_data: pd.DataFrame) -> Tuple[bool, float, float]:
        """
        Проверка на высокую волатильность.
        
        Args:
            price_data (pd.DataFrame): DataFrame с ценовыми данными
            
        Returns:
            Tuple[bool, float, float]: (флаг высокой волатильности, текущая волатильность, средняя волатильность)
        """
        if len(price_data) < self.lookback_period * 2:
            logger.warning(f"Недостаточно данных для анализа волатильности")
            return False, 0.0, 0.0
        
        # Расчет текущей волатильности
        current_volatility = self.calculate_volatility(price_data.iloc[-self.lookback_period:])
        
        # Расчет средней волатильности за предыдущий период
        previous_data = price_data.iloc[-(self.lookback_period*2):-self.lookback_period]
        avg_volatility = self.calculate_volatility(previous_data)
        
        # Проверка на высокую волатильность
        if current_volatility > avg_volatility * self.volatility_threshold:
            logger.warning(f"Обнаружена высокая волатильность: {current_volatility:.2f}% "
                          f"(в {current_volatility/avg_volatility:.2f} раз выше средней)")
            return True, current_volatility, avg_volatility
        
        return False, current_volatility, avg_volatility
    
    def is_extreme_volatility(self, price_data: pd.DataFrame) -> Tuple[bool, float, float]:
        """
        Проверка на экстремальную волатильность.
        
        Args:
            price_data (pd.DataFrame): DataFrame с ценовыми данными
            
        Returns:
            Tuple[bool, float, float]: (флаг экстремальной волатильности, текущая волатильность, средняя волатильность)
        """
        high_volatility, current_volatility, avg_volatility = self.is_high_volatility(price_data)
        
        # Проверка на экстремальную волатильность
        if current_volatility > avg_volatility * self.extreme_volatility_threshold:
            logger.warning(f"Обнаружена экстремальная волатильность: {current_volatility:.2f}% "
                          f"(в {current_volatility/avg_volatility:.2f} раз выше средней)")
            return True, current_volatility, avg_volatility
        
        return False, current_volatility, avg_volatility
    
    def adjust_position_size_for_volatility(self, base_position_size: float, 
                                          price_data: pd.DataFrame) -> float:
        """
        Корректировка размера позиции с учетом текущей волатильности.
        
        Args:
            base_position_size (float): Базовый размер позиции
            price_data (pd.DataFrame): DataFrame с ценовыми данными
            
        Returns:
            float: Скорректированный размер позиции
        """
        high_volatility, current_volatility, avg_volatility = self.is_high_volatility(price_data)
        extreme_volatility, _, _ = self.is_extreme_volatility(price_data)
        
        if extreme_volatility:
            # При экстремальной волатильности уменьшаем позицию на 75%
            adjusted_size = base_position_size * 0.25
            logger.warning(f"Размер позиции уменьшен до 25% ({adjusted_size}) из-за экстремальной волатильности")
        elif high_volatility:
            # При высокой волатильности уменьшаем позицию на 50%
            adjusted_size = base_position_size * 0.5
            logger.warning(f"Размер позиции уменьшен до 50% ({adjusted_size}) из-за высокой волатильности")
        else:
            adjusted_size = base_position_size
        
        return adjusted_size
    
    def adjust_stop_loss_for_volatility(self, entry_price: float, base_stop_percent: float, 
                                      price_data: pd.DataFrame, side: str) -> float:
        """
        Корректировка уровня стоп-лосса с учетом текущей волатильности.
        
        Args:
            entry_price (float): Цена входа в позицию
            base_stop_percent (float): Базовый процент для стоп-лосса
            price_data (pd.DataFrame): DataFrame с ценовыми данными
            side (str): Сторона сделки ('buy' или 'sell')
            
        Returns:
            float: Скорректированный уровень стоп-лосса
        """
        high_volatility, current_volatility, avg_volatility = self.is_high_volatility(price_data)
        extreme_volatility, _, _ = self.is_extreme_volatility(price_data)
        
        volatility_factor = 1.0
        
        if extreme_volatility:
            # При экстремальной волатильности расширяем стоп-лосс в 2 раза
            volatility_factor = 2.0
            logger.warning(f"Стоп-лосс расширен в 2 раза из-за экстремальной волатильности")
        elif high_volatility:
            # При высокой волатильности расширяем стоп-лосс в 1.5 раза
            volatility_factor = 1.5
            logger.warning(f"Стоп-лосс расширен в 1.5 раза из-за высокой волатильности")
        
        adjusted_stop_percent = base_stop_percent * volatility_factor
        
        if side == 'buy':
            stop_level = entry_price * (1 - adjusted_stop_percent / 100)
        else:  # 'sell'
            stop_level = entry_price * (1 + adjusted_stop_percent / 100)
        
        logger.info(f"Рассчитан стоп-лосс с учетом волатильности: {stop_level} ({adjusted_stop_percent:.2f}%) для {side}")
        
        return stop_level