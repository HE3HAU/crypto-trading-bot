#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Утилиты для загрузки конфигурации.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла.
    
    Args:
        config_path (Optional[str]): Путь к файлу конфигурации. 
            Если None, используется стандартный путь.
            
    Returns:
        Dict[str, Any]: Словарь с конфигурацией
    """
    if config_path is None:
        # Получаем путь к директории с текущим файлом
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Получаем корневую директорию проекта (на два уровня выше)
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(project_dir, 'config', 'config.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Загружена конфигурация из {config_path}")
            return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации из {config_path}: {e}")
        return {}

def load_api_keys(api_keys_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Загрузка API ключей из YAML файла.
    
    Args:
        api_keys_path (Optional[str]): Путь к файлу с API ключами.
            Если None, используется стандартный путь.
            
    Returns:
        Dict[str, Dict[str, str]]: Словарь с API ключами
    """
    if api_keys_path is None:
        # Получаем путь к директории с текущим файлом
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Получаем корневую директорию проекта (на два уровня выше)
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        api_keys_path = os.path.join(project_dir, 'config', 'api_keys.yaml')
    
    try:
        with open(api_keys_path, 'r') as file:
            api_keys = yaml.safe_load(file)
            logger.info(f"Загружены API ключи из {api_keys_path}")
            return api_keys
    except Exception as e:
        logger.error(f"Ошибка при загрузке API ключей из {api_keys_path}: {e}")
        return {}

def create_api_keys_example():
    """
    Создание примера файла с API ключами, если он не существует.
    """
    # Получаем путь к директории с текущим файлом
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Получаем корневую директорию проекта (на два уровня выше)
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    example_path = os.path.join(project_dir, 'config', 'api_keys.yaml.example')
    
    # Проверяем, существует ли уже пример
    if os.path.exists(example_path):
        return
    
    example_config = {
        'okx': {
            'api_key': '',
            'secret_key': '',
            'password': ''
        },
        'kucoin': {
            'api_key': '',
            'secret_key': '',
            'passphrase': ''
        },
        'bybit': {
            'api_key': '',
            'secret_key': ''
        }
    }
    
    try:
        with open(example_path, 'w') as file:
            yaml.dump(example_config, file, default_flow_style=False)
            logger.info(f"Создан пример файла с API ключами: {example_path}")
    except Exception as e:
        logger.error(f"Ошибка при создании примера файла с API ключами: {e}")