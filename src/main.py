#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной файл запуска торгового бота.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import time

# Добавляем корневой каталог проекта в путь Python
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config_loader import load_config, load_api_keys, create_api_keys_example
from src.api.okx_exchange import OkxExchange
from src.data.data_fetcher import DataFetcher
from src.data.data_visualizer import DataVisualizer
from src.data.metrics_calculator import MetricsCalculator
from src.strategies.sma_strategy import SmaStrategy
from src.strategies.rsi_strategy import RsiStrategy
from src.strategies.macd_strategy import MacdStrategy
from src.strategies.bollinger_strategy import BollingerStrategy
from src.strategies.combined_strategy import CombinedStrategy
from src.backtesting.backtester import Backtester
from src.risk.risk_manager import PositionSizer, StopLossCalculator, TakeProfitCalculator, DrawdownProtection, VolatilityGuard
from src.risk.order_manager import OrderManager, Order, Position
from src.risk.crisis_manager import CrisisManager

# Настройка логирования
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(description='Торговый бот для криптовалютных бирж')
    parser.add_argument('--backtest', dest='backtest', action='store_true',
                        help='Запустить бэктестинг стратегий')
    parser.add_argument('--risk-test', dest='risk_test', action='store_true',
                        help='Протестировать модули управления рисками')
    parser.add_argument('--symbol', dest='symbol', type=str, default='BTC/USDT',
                        help='Символ торговой пары (по умолчанию BTC/USDT)')
    parser.add_argument('--timeframe', dest='timeframe', type=str, default='1h',
                        help='Временной интервал (по умолчанию 1h)')
    
    return parser.parse_args()

def main():
    """
    Основная функция запуска бота.
    """
    # Парсинг аргументов командной строки
    args = parse_args()
    
    logger.info("Запуск торгового бота")
    
    # Создаем пример файла с API ключами, если он не существует
    create_api_keys_example()
    
    # Загрузка конфигурации
    config = load_config()
    if not config:
        logger.error("Невозможно запустить бота без конфигурации")
        return
    
    # Загрузка API ключей
    api_keys = load_api_keys()
    if not api_keys:
        logger.warning("API ключи не найдены, работа в режиме только для чтения")
    
    # Получаем параметры из конфигурации
    general_config = config.get('general', {})
    exchanges_config = config.get('exchanges', {})
    trading_config = config.get('trading', {})
    
    # Проверяем, какие биржи включены
    enabled_exchanges = []
    for exchange_name, exchange_config in exchanges_config.items():
        if exchange_config.get('enabled', False):
            enabled_exchanges.append(exchange_name)
    
    if not enabled_exchanges:
        logger.error("Нет включенных бирж в конфигурации")
        return
    
    logger.info(f"Включенные биржи: {', '.join(enabled_exchanges)}")
    
    # Инициализация соединения с биржей
    # На данный момент поддерживаем только OKX
    exchange = None
    if 'okx' in enabled_exchanges:
        okx_api_keys = api_keys.get('okx', {})
        okx_config = exchanges_config.get('okx', {})
        
        api_key = okx_api_keys.get('api_key', '')
        secret_key = okx_api_keys.get('secret_key', '')
        password = okx_api_keys.get('password', '')
        sandbox = okx_config.get('sandbox', True)
        
        exchange = OkxExchange(api_key, secret_key, password, sandbox)
        if not exchange.connect():
            logger.error("Невозможно подключиться к OKX")
            return
    else:
        logger.error("Нет поддерживаемых бирж в конфигурации")
        return
    
    # Инициализация сборщика данных
    data_fetcher = DataFetcher(exchange)
    
    # Инициализация визуализатора данных
    visualizer = DataVisualizer()
    
    # Инициализация калькулятора метрик
    metrics_calc = MetricsCalculator()
    
    # Создание экземпляров стратегий
    strategies = {
        "SMA": SmaStrategy(),
        "RSI": RsiStrategy(),
        "MACD": MacdStrategy(),
        "Bollinger": BollingerStrategy(),
        "Combined": CombinedStrategy()
    }
    
    # Параметры торговли
    symbol = args.symbol
    timeframe = args.timeframe or general_config.get('trade_interval', '1h')
    max_open_trades = trading_config.get('max_open_trades', 3)
    stake_amount = trading_config.get('stake_amount', 100)
    
    logger.info(f"Параметры торговли: symbol={symbol}, timeframe={timeframe}, max_open_trades={max_open_trades}, stake_amount={stake_amount}")
    
    # Основной цикл работы бота
    try:
        # Получение исторических данных
        logger.info(f"Получение данных для {symbol}")
        data = data_fetcher.fetch_ohlcv(symbol, timeframe, 100)
        
        if data.empty:
            logger.error(f"Не удалось получить данные для {symbol}")
            return
        
        # Добавление индикаторов
        processed_data = data_fetcher.add_indicators(data)
        
        # Сохраняем данные
        data_fetcher.save_ohlcv(data, symbol, timeframe)
        data_fetcher.save_processed_data(processed_data, symbol, timeframe)
        
        # Создание визуализаций
        os.makedirs('visualizations', exist_ok=True)
        
        # Создание свечного графика
        visualizer.plot_candlestick(processed_data[-50:], symbol, 
                                    indicators=['sma7', 'sma25'], 
                                    save=True, show=False)
        
        # Создание графика с индикаторами
        indicators_dict = {
            'MACD': ['macd', 'macd_signal', 'macd_hist'],
            'RSI': ['rsi']
        }
        visualizer.plot_indicators(processed_data[-50:], symbol, 
                                   indicators_dict, save=True, show=False)
        
        # Анализ текущего рынка с помощью стратегий
        for strategy_name, strategy in strategies.items():
            analysis = strategy.analyze(processed_data)
            logger.info(f"Стратегия {strategy_name}: {analysis['signal']} ({analysis['reason']})")
        
        # Режим бэктестинга стратегий
        if args.backtest:
            logger.info("Запуск бэктестинга стратегий")
            
            # Получаем данные за более длительный период для бэктестинга
            backtest_data = data_fetcher.fetch_ohlcv(symbol, timeframe, 500)
            
            if backtest_data.empty:
                logger.error(f"Не удалось получить данные для бэктестинга")
            else:
                # Добавляем индикаторы к данным
                backtest_data = data_fetcher.add_indicators(backtest_data)
                
                # Создаем директорию для результатов бэктестинга
                os.makedirs('backtesting_results', exist_ok=True)
                
                # Тестируем каждую стратегию
                for strategy_name, strategy in strategies.items():
                    logger.info(f"Запуск бэктеста для стратегии {strategy_name}")
                    
                    # Создаем бэктестер
                    backtester = Backtester(strategy, initial_capital=10000.0, commission=0.001)
                    
                    # Запускаем бэктест
                    results = backtester.run(backtest_data, symbol, stake_amount=100.0)
                    
                    # Визуализируем результаты
                    backtester.visualize_results(results)
                    
                    # Выводим основные метрики
                    metrics = results.get('metrics', {})
                    logger.info(f"Стратегия {strategy_name}: Доходность {results.get('total_return', 0):.2f}%, "
                               f"Sharpe {metrics.get('sharpe_ratio', 0):.2f}, "
                               f"Max Drawdown {metrics.get('max_drawdown', 0):.2f}%")
                
                # Для SMA стратегии пробуем оптимизировать параметры
                logger.info("Оптимизация параметров SMA стратегии")
                
                sma_backtester = Backtester(strategies["SMA"], initial_capital=10000.0, commission=0.001)
                param_grid = {
                    "short_window": [5, 7, 10, 15],
                    "long_window": [20, 25, 30, 40]
                }
                
                optimization_results = sma_backtester.optimize(backtest_data, param_grid, symbol)
                
                logger.info(f"Лучшие параметры SMA: {optimization_results['best_params']}")
                
                logger.info("Бэктестинг стратегий завершен")
        
        # Режим тестирования управления рисками
        if args.risk_test:
            logger.info("Тестирование модулей управления рисками")
            
            # Инициализация калькуляторов и менеджеров
            position_sizer = PositionSizer(initial_capital=10000.0)
            stop_loss_calc = StopLossCalculator()
            take_profit_calc = TakeProfitCalculator()
            order_manager = OrderManager(exchange)
            crisis_manager = CrisisManager(order_manager, initial_capital=10000.0)
            
            # Демонстрация расчета размера позиции
            current_price = processed_data['close'].iloc[-1]
            
            # Расчет размера позиции разными методами
            fixed_size_units, fixed_size_cost = position_sizer.fixed_size(10000.0, current_price, 0.001)
            logger.info(f"Фиксированный размер: {fixed_size_units} BTC, {fixed_size_cost} USDT")
            
            fixed_risk_units, fixed_risk_cost = position_sizer.fixed_risk(10000.0, current_price, 1.0, 2.0)
            logger.info(f"Фиксированный риск (1% капитала, 2% стоп-лосс): {fixed_risk_units} BTC, {fixed_risk_cost} USDT")
            
            kelly_units, kelly_cost = position_sizer.kelly_criterion(10000.0, current_price, 0.55, 2.0)
            logger.info(f"Критерий Келли (55% побед, 2:1 R/R): {kelly_units} BTC, {kelly_cost} USDT")
            
            percent_units, percent_cost = position_sizer.percent_of_equity(10000.0, current_price, 2.0)
            logger.info(f"2% от капитала: {percent_units} BTC, {percent_cost} USDT")
            
            # Демонстрация расчета стоп-лосса
            atr = processed_data['high'].rolling(14).max() - processed_data['low'].rolling(14).min()
            atr_value = atr.iloc[-1] / current_price * 100  # в процентах от цены
            
            fixed_sl = stop_loss_calc.fixed_percent(current_price, 'buy', 2.0)
            logger.info(f"Фиксированный стоп-лосс (2%): {fixed_sl}")
            
            atr_sl = stop_loss_calc.atr_based(current_price, 'buy', atr_value * current_price / 100, 2.0)
            logger.info(f"ATR стоп-лосс (2*ATR): {atr_sl}")
            
            # Демонстрация расчета тейк-профита
            fixed_tp = take_profit_calc.fixed_percent(current_price, 'buy', 4.0)
            logger.info(f"Фиксированный тейк-профит (4%): {fixed_tp}")
            
            rr_tp = take_profit_calc.risk_reward_ratio(current_price, fixed_sl, 'buy', 2.0)
            logger.info(f"Тейк-профит с R/R=2: {rr_tp}")
            
            multiple_tps = take_profit_calc.multiple_targets(current_price, fixed_sl, 'buy', [1.0, 2.0, 3.0])
            logger.info(f"Множественные тейк-профиты: {multiple_tps}")
            
            # Демонстрация защиты от просадки
            drawdown_protection = DrawdownProtection(initial_capital=10000.0)
            drawdown_protection.update_capital(9500.0)  # 5% просадка
            logger.info(f"Просадка: {drawdown_protection.current_drawdown_percent:.2f}%, множитель размера позиции: {drawdown_protection.position_size_multiplier:.2f}")
            
            logger.info("Демонстрация антикризисного управления")
            
            # Проверка на экстремальные условия
            extreme_conditions, reason = crisis_manager.detect_extreme_conditions(processed_data)
            if extreme_conditions:
                logger.warning(f"Обнаружены экстремальные условия: {reason}")
                # В реальном режиме здесь бы выполнялось закрытие позиций
                # crisis_manager.close_positions_in_crisis(True, reason)
            else:
                logger.info("Экстремальных условий не обнаружено, торговля возможна")
            
            logger.info("Тестирование модулей управления рисками завершено")
        
        # Рассчитываем метрики для демонстрации
        returns_data = metrics_calc.calculate_returns(processed_data)
        
        # Создаем кривую капитала (для демонстрации используем цену закрытия)
        equity_curve = processed_data['close']
        
        # Генерируем отчет с метриками
        metrics_report = metrics_calc.generate_full_metrics_report(equity_curve)
        
        logger.info(f"Метрики рынка: Волатильность {metrics_report.get('volatility', 0):.2f}%, "
                   f"Sharpe {metrics_report.get('sharpe_ratio', 0):.2f}")
        
        # Сохраняем график кривой капитала
        metrics_calc.plot_equity_curve(equity_curve, title=f"{symbol} Price History", 
                                      save_path="visualizations/equity_curve.png")
        
        logger.info("Анализ данных завершен")
        logger.info("Бот завершил демонстрационную работу. В реальном режиме здесь бы выполнялись торговые операции.")
        
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка в работе бота: {e}", exc_info=True)
    finally:
        logger.info("Завершение работы бота")

if __name__ == "__main__":
    main()