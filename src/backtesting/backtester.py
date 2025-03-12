#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Система бэктестинга для оценки эффективности торговых стратегий.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import os

from src.strategies.base_strategy import BaseStrategy
from src.data.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)

class Trade:
    """
    Класс для представления отдельной торговой сделки.
    """
    
    def __init__(self, symbol: str, entry_time: datetime, entry_price: float, 
                 amount: float, side: str, strategy_name: str):
        """
        Инициализация сделки.
        
        Args:
            symbol (str): Символ торгового инструмента
            entry_time (datetime): Время входа в сделку
            entry_price (float): Цена входа
            amount (float): Количество купленного/проданного актива
            side (str): Сторона сделки ('buy' или 'sell')
            strategy_name (str): Название стратегии, сгенерировавшей сигнал
        """
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.amount = amount
        self.side = side
        self.strategy_name = strategy_name
        
        # Инициализация полей, которые будут заполнены при закрытии сделки
        self.exit_time = None
        self.exit_price = None
        self.profit = None
        self.profit_pct = None
        self.duration = None
    
    def close(self, exit_time: datetime, exit_price: float):
        """
        Закрытие сделки.
        
        Args:
            exit_time (datetime): Время выхода из сделки
            exit_price (float): Цена выхода
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        # Расчет прибыли
        if self.side == 'buy':
            self.profit = (exit_price - self.entry_price) * self.amount
            self.profit_pct = (exit_price / self.entry_price - 1) * 100
        else:  # 'sell'
            self.profit = (self.entry_price - exit_price) * self.amount
            self.profit_pct = (self.entry_price / exit_price - 1) * 100
        
        # Расчет продолжительности сделки
        self.duration = (exit_time - self.entry_time).total_seconds() / 3600  # в часах
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование сделки в словарь.
        
        Returns:
            Dict[str, Any]: Словарь с информацией о сделке
        """
        return {
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'side': self.side,
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'amount': self.amount,
            'profit': self.profit,
            'profit_pct': self.profit_pct,
            'duration': self.duration
        }


class Portfolio:
    """
    Класс для отслеживания капитала, позиций и сделок.
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Инициализация портфеля.
        
        Args:
            initial_capital (float): Начальный капитал
            commission (float): Комиссия за сделку (в долях от объема)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.positions = {}  # Словарь открытых позиций {symbol: amount}
        self.trades = []  # Список сделок
        self.equity_curve = []  # История изменения капитала
        
        logger.info(f"Инициализирован портфель с капиталом {initial_capital} и комиссией {commission}")
    
    def open_position(self, symbol: str, time: datetime, price: float, 
                      amount: float, side: str, strategy_name: str) -> Optional[Trade]:
        """
        Открытие позиции.
        
        Args:
            symbol (str): Символ торгового инструмента
            time (datetime): Время открытия позиции
            price (float): Цена открытия
            amount (float): Количество актива
            side (str): Сторона сделки ('buy' или 'sell')
            strategy_name (str): Название стратегии
            
        Returns:
            Optional[Trade]: Сделка или None, если не хватает капитала
        """
        # Проверка наличия достаточного капитала
        cost = price * amount
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost
        
        if side == 'buy' and total_cost > self.capital:
            logger.warning(f"Недостаточно капитала для покупки {amount} {symbol} по цене {price}")
            return None
        
        # Изменение капитала и позиций
        if side == 'buy':
            self.capital -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + amount
        else:  # 'sell'
            # Для шорта пока просто уменьшаем капитал
            self.capital -= commission_cost
            self.positions[symbol] = self.positions.get(symbol, 0) - amount
        
        # Создание сделки
        trade = Trade(symbol, time, price, amount, side, strategy_name)
        self.trades.append(trade)
        
        logger.info(f"Открыта позиция: {side} {amount} {symbol} по цене {price}")
        return trade
    
    def close_position(self, symbol: str, time: datetime, price: float) -> Optional[float]:
        """
        Закрытие позиции.
        
        Args:
            symbol (str): Символ торгового инструмента
            time (datetime): Время закрытия позиции
            price (float): Цена закрытия
            
        Returns:
            Optional[float]: Прибыль от сделки или None, если нет открытой позиции
        """
        if symbol not in self.positions or self.positions[symbol] == 0:
            logger.warning(f"Нет открытой позиции для {symbol}")
            return None
        
        # Получаем количество актива в позиции
        amount = self.positions[symbol]
        
        # Находим последнюю открытую сделку для этого символа
        for trade in reversed(self.trades):
            if trade.symbol == symbol and trade.exit_time is None:
                # Закрываем сделку
                trade.close(time, price)
                
                # Расчет стоимости закрытия с учетом комиссии
                close_cost = price * abs(amount)
                commission_cost = close_cost * self.commission
                
                # Изменение капитала в зависимости от типа сделки
                if amount > 0:  # Была покупка, теперь продаем
                    self.capital += close_cost - commission_cost
                else:  # Был шорт, теперь покупаем
                    self.capital -= close_cost + commission_cost
                    self.capital += trade.profit  # Добавляем прибыль
                
                # Обнуляем позицию
                self.positions[symbol] = 0
                
                logger.info(f"Закрыта позиция: {symbol} по цене {price}, прибыль: {trade.profit}")
                return trade.profit
        
        logger.warning(f"Не найдена открытая сделка для {symbol}")
        return None
    
    def update_equity(self, time: datetime, prices: Dict[str, float]):
        """
        Обновление кривой капитала.
        
        Args:
            time (datetime): Текущее время
            prices (Dict[str, float]): Словарь текущих цен {symbol: price}
        """
        # Рассчитываем стоимость открытых позиций
        positions_value = 0
        for symbol, amount in self.positions.items():
            if symbol in prices and amount != 0:
                positions_value += prices[symbol] * amount
        
        # Добавляем запись в кривую капитала
        total_equity = self.capital + positions_value
        self.equity_curve.append({'time': time, 'equity': total_equity})
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        Получение DataFrame со сделками.
        
        Returns:
            pd.DataFrame: DataFrame со сделками
        """
        trades_data = [trade.to_dict() for trade in self.trades if trade.exit_time is not None]
        return pd.DataFrame(trades_data)
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """
        Получение DataFrame с кривой капитала.
        
        Returns:
            pd.DataFrame: DataFrame с кривой капитала
        """
        return pd.DataFrame(self.equity_curve).set_index('time')


class Backtester:
    """
    Класс для бэктестинга торговых стратегий.
    """
    
    def __init__(self, strategy: BaseStrategy, initial_capital: float = 10000.0, 
                 commission: float = 0.001):
        """
        Инициализация бэктестера.
        
        Args:
            strategy (BaseStrategy): Стратегия для тестирования
            initial_capital (float): Начальный капитал
            commission (float): Комиссия за сделку (в долях от объема)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = Portfolio(initial_capital, commission)
        self.metrics_calculator = MetricsCalculator()
        
        logger.info(f"Инициализирован Backtester для стратегии {strategy.name}")
    
    def run(self, data: pd.DataFrame, symbol: str = "BTC/USDT", 
            stake_amount: float = 100.0, stake_percent: bool = False) -> Dict[str, Any]:
        """
        Запуск бэктестинга.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            symbol (str): Символ торгового инструмента
            stake_amount (float): Размер ставки на сделку
            stake_percent (bool): Если True, stake_amount рассматривается как процент от капитала
            
        Returns:
            Dict[str, Any]: Результаты бэктестинга
        """
        if data.empty:
            logger.warning("Бэктестинг невозможен: пустой DataFrame")
            return {}
        
        logger.info(f"Запуск бэктестинга для {symbol} с {len(data)} барами")
        
        # Инициализация портфеля
        self.portfolio = Portfolio(self.initial_capital, self.commission)
        
        # Текущая открытая позиция
        current_position = None
        
        # Проходим по данным
        for i in range(1, len(data)):
            # Получаем срез данных до текущего момента для анализа
            current_time = data.index[i]
            historical_data = data.iloc[:i]
            current_price = data['close'].iloc[i]
            
            # Получаем сигнал от стратегии
            analysis = self.strategy.analyze(historical_data)
            signal = analysis["signal"]
            
            # Определяем размер ставки
            if stake_percent:
                actual_stake = self.portfolio.capital * stake_amount / 100
            else:
                actual_stake = min(stake_amount, self.portfolio.capital)
            
            # Рассчитываем количество актива
            amount = actual_stake / current_price
            
            # Выполняем торговые операции на основе сигнала
            if signal == "buy" and symbol not in self.portfolio.positions:
                # Открываем длинную позицию
                trade = self.portfolio.open_position(
                    symbol, current_time, current_price, amount, 'buy', self.strategy.name
                )
                if trade:
                    current_position = trade
            
            elif signal == "sell" and symbol in self.portfolio.positions and self.portfolio.positions[symbol] > 0:
                # Закрываем длинную позицию
                self.portfolio.close_position(symbol, current_time, current_price)
                current_position = None
            
            # Обновляем кривую капитала
            self.portfolio.update_equity(current_time, {symbol: current_price})
        
        # Закрываем все оставшиеся позиции по последней цене
        for symbol, amount in list(self.portfolio.positions.items()):
            if amount != 0:
                last_time = data.index[-1]
                last_price = data['close'].iloc[-1]
                self.portfolio.close_position(symbol, last_time, last_price)
        
        # Получаем DataFrame со сделками и кривой капитала
        trades_df = self.portfolio.get_trades_df()
        equity_curve_df = self.portfolio.get_equity_curve_df()
        
        # Рассчитываем метрики
        metrics = {}
        if not equity_curve_df.empty:
            metrics = self.metrics_calculator.generate_full_metrics_report(
                equity_curve_df['equity'], trades_df
            )
        
        logger.info(f"Бэктестинг завершен: {len(trades_df)} сделок")
        
        return {
            "trades": trades_df,
            "equity_curve": equity_curve_df,
            "metrics": metrics,
            "final_capital": self.portfolio.capital,
            "total_return": (self.portfolio.capital / self.initial_capital - 1) * 100
        }
    
    def optimize(self, data: pd.DataFrame, param_grid: Dict[str, List[Any]], 
                 symbol: str = "BTC/USDT", stake_amount: float = 100.0) -> Dict[str, Any]:
        """
        Оптимизация параметров стратегии.
        
        Args:
            data (pd.DataFrame): DataFrame с историческими данными
            param_grid (Dict[str, List[Any]]): Сетка параметров для перебора
            symbol (str): Символ торгового инструмента
            stake_amount (float): Размер ставки на сделку
            
        Returns:
            Dict[str, Any]: Результаты оптимизации
        """
        logger.info(f"Запуск оптимизации параметров для стратегии {self.strategy.name}")
        
        # Функция для создания комбинаций параметров
        def create_param_combinations(param_grid):
            import itertools
            
            keys = param_grid.keys()
            values = param_grid.values()
            combinations = list(itertools.product(*values))
            
            return [dict(zip(keys, combo)) for combo in combinations]
        
        # Получаем все комбинации параметров
        param_combinations = create_param_combinations(param_grid)
        logger.info(f"Количество комбинаций параметров: {len(param_combinations)}")
        
        # Результаты оптимизации
        optimization_results = []
        
        # Перебираем комбинации параметров
        for params in param_combinations:
            # Создаем новый экземпляр стратегии с текущими параметрами
            strategy_class = self.strategy.__class__
            strategy_instance = strategy_class(**params)
            
            # Создаем новый бэктестер с этой стратегией
            backtester = Backtester(strategy_instance, self.initial_capital, self.commission)
            
            # Запускаем бэктестинг
            result = backtester.run(data, symbol, stake_amount)
            
            # Добавляем результаты
            result_summary = {
                "params": params,
                "strategy_name": strategy_instance.name,
                "total_return": result.get("total_return", 0),
                "sharpe_ratio": result.get("metrics", {}).get("sharpe_ratio", 0),
                "max_drawdown": result.get("metrics", {}).get("max_drawdown", 0),
                "win_rate": result.get("metrics", {}).get("trades", {}).get("win_rate", 0),
                "trades_count": len(result.get("trades", {})),
                "final_capital": result.get("final_capital", self.initial_capital)
            }
            
            optimization_results.append(result_summary)
            
            logger.info(f"Параметры: {params}, Доходность: {result_summary['total_return']:.2f}%, Sharpe: {result_summary['sharpe_ratio']:.2f}")
        
        # Сортируем результаты по доходности
        optimization_results.sort(key=lambda x: x["total_return"], reverse=True)
        
        # Выбираем лучшие параметры
        best_params = optimization_results[0]["params"] if optimization_results else {}
        
        logger.info(f"Оптимизация завершена. Лучшие параметры: {best_params}")
        
        return {
            "best_params": best_params,
            "results": optimization_results
        }
    
    def visualize_results(self, results: Dict[str, Any], 
                          output_dir: str = "backtesting_results") -> None:
        """
        Визуализация результатов бэктестинга.
        
        Args:
            results (Dict[str, Any]): Результаты бэктестинга
            output_dir (str): Директория для сохранения визуализаций
        """
        if not results:
            logger.warning("Нет результатов для визуализации")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Получаем данные
        equity_curve = results.get("equity_curve", pd.DataFrame())
        trades = results.get("trades", pd.DataFrame())
        metrics = results.get("metrics", {})
        
        if equity_curve.empty:
            logger.warning("Нет данных о кривой капитала для визуализации")
            return
        
        # Визуализация кривой капитала
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve['equity'], label='Equity')
        plt.title(f'Equity Curve - {self.strategy.name}')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Добавляем метки сделок, если они есть
        if not trades.empty:
            # Маркеры покупки
            buy_trades = trades[trades['side'] == 'buy']
            if not buy_trades.empty:
                plt.scatter(buy_trades['entry_time'], buy_trades['entry_price'], 
                           color='green', marker='^', s=100, label='Buy')
            
            # Маркеры продажи
            sell_trades = trades[trades['side'] == 'sell']
            if not sell_trades.empty:
                plt.scatter(sell_trades['entry_time'], sell_trades['entry_price'], 
                           color='red', marker='v', s=100, label='Sell')
        
        plt.savefig(os.path.join(output_dir, f'{self.strategy.name}_equity_curve.png'), dpi=150)
        plt.close()
        
        # Визуализация распределения прибыли сделок
        if not trades.empty and 'profit' in trades.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(trades['profit'], bins=20, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title(f'Trade Profit Distribution - {self.strategy.name}')
            plt.xlabel('Profit')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{self.strategy.name}_profit_distribution.png'), dpi=150)
            plt.close()
            
            # Визуализация продолжительности сделок
            if 'duration' in trades.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(trades['duration'], bins=20, alpha=0.7)
                plt.title(f'Trade Duration Distribution - {self.strategy.name}')
                plt.xlabel('Duration (hours)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'{self.strategy.name}_duration_distribution.png'), dpi=150)
                plt.close()
        
        # Создание сводного отчета
        metrics_text = f"""
        # Backtesting Results: {self.strategy.name}
        
        ## Summary
        - Initial Capital: ${self.initial_capital:.2f}
        - Final Capital: ${results.get('final_capital', 0):.2f}
        - Total Return: {results.get('total_return', 0):.2f}%
        - Number of Trades: {len(trades)}
        
        ## Performance Metrics
        - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        - Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
        - Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
        - Annual Return: {metrics.get('annual_return', 0):.2f}%
        - Volatility: {metrics.get('volatility', 0):.2f}%
        
        ## Trade Statistics
        """
        
        if not trades.empty and 'profit' in trades.columns:
            win_trades = trades[trades['profit'] > 0]
            loss_trades = trades[trades['profit'] < 0]
            
            metrics_text += f"""
        - Win Rate: {len(win_trades) / len(trades) * 100:.2f}%
        - Average Profit: ${trades['profit'].mean():.2f}
        - Average Win: ${win_trades['profit'].mean() if not win_trades.empty else 0:.2f}
        - Average Loss: ${loss_trades['profit'].mean() if not loss_trades.empty else 0:.2f}
        - Profit Factor: {win_trades['profit'].sum() / abs(loss_trades['profit'].sum()) if not loss_trades.empty and loss_trades['profit'].sum() != 0 else float('inf'):.2f}
        - Average Duration: {trades['duration'].mean():.2f} hours
        """
        
        # Сохраняем отчет
        with open(os.path.join(output_dir, f'{self.strategy.name}_report.md'), 'w') as f:
            f.write(metrics_text)
        
        logger.info(f"Визуализации сохранены в {output_dir}")