import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict, Optional

class Logger:
    """
    Класс для логирования операций и торговых событий
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Создаем два логгера: для общих событий и для торговых операций
        self._setup_general_logger()
        self._setup_trade_logger()
        
    def _setup_general_logger(self) -> None:
        """Настройка основного логгера"""
        log_file = self.log_dir / f"general_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Настройка форматирования
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Хендлер для файла
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Хендлер для консоли
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Настройка логгера
        self.logger = logging.getLogger('PairTrading')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _setup_trade_logger(self) -> None:
        """Настройка логгера для торговых операций"""
        self.trade_log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
        if not self.trade_log_file.exists():
            with open(self.trade_log_file, 'w') as f:
                json.dump([], f)
                
    def info(self, message: str) -> None:
        """Логирование информационного сообщения"""
        self.logger.info(message)
        
    def error(self, message: str) -> None:
        """Логирование ошибки"""
        self.logger.error(message)
        
    def warning(self, message: str) -> None:
        """Логирование предупреждения"""
        self.logger.warning(message)
        
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Логирование торговой операции
        
        Args:
            trade_data: словарь с информацией о сделке
                {
                    'timestamp': str,
                    'pair': tuple,
                    'action': str,
                    'price': float,
                    'size': float,
                    'reason': str
                }
        """
        try:
            with open(self.trade_log_file, 'r') as f:
                trades = json.load(f)
                
            trade_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trades.append(trade_data)
            
            with open(self.trade_log_file, 'w') as f:
                json.dump(trades, f, indent=4)
                
            self.info(f"Trade logged: {trade_data['action']} {trade_data['pair']} "
                     f"at {trade_data['price']}")
                     
        except Exception as e:
            self.error(f"Failed to log trade: {str(e)}")
            
    def log_performance(self, performance_data: Dict[str, Any]) -> None:
        """
        Логирование результатов работы стратегии
        
        Args:
            performance_data: словарь с метриками производительности
        """
        log_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            performance_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(log_file, 'w') as f:
                json.dump(performance_data, f, indent=4)
                
            self.info("Performance metrics logged successfully")
            
        except Exception as e:
            self.error(f"Failed to log performance metrics: {str(e)}")