
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной файл запуска торгового бота.
"""

import os
import yaml
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Загрузка конфигурации из файла."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        return None

def main():
    """Основная функция запуска бота."""
    logger.info("Запуск торгового бота")
    
    # Загрузка конфигурации
    config = load_config()
    if not config:
        logger.error("Невозможно запустить бота без конфигурации")
        return
    
    logger.info(f"Загружена конфигурация: {config['general']}")
    logger.info("Бот пока не реализован, ожидайте обновлений")
    
if __name__ == "__main__":
    # Создание директории для логов, если её нет
    os.makedirs('logs', exist_ok=True)
    main()