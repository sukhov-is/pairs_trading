import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
from collections import defaultdict

class PositionStatus(Enum):
    """Статус позиции"""
    PENDING = "pending"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    CANCELLED = "cancelled"

@dataclass
class Position:
    """Класс для хранения информации о позиции"""
    pair: Tuple[str, str]
    entry_time: datetime
    entry_prices: Tuple[float, float]
    sizes: Tuple[float, float]
    direction: int  # 1 для long, -1 для short
    status: PositionStatus
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_prices: Optional[Tuple[float, float]] = None
    pnl: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""

class PositionManager:
    """
    Класс управления позициями в торговой системе
    """
    
    def __init__(self,
                 initial_capital: float,
                 risk_manager,
                 commission: float = 0.001,
                 max_positions: int = 5,
                 logger: Optional[logging.Logger] = None):
        """
        Инициализация менеджера позиций

        Args:
            initial_capital: Начальный капитал
            risk_manager: Экземпляр RiskManager
            commission: Комиссия за сделку
            max_positions: Максимальное количество открытых позиций
            logger: Логгер для записи событий
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_manager = risk_manager
        self.commission = commission
        self.max_positions = max_positions
        self.logger = logger or logging.getLogger(__name__)
        
        # Хранение позиций
        self.active_positions: Dict[Tuple[str, str], Position] = {}
        self.closed_positions: List[Position] = []
        self.pending_orders: List[Position] = []
        
        # Статистика
        self.stats = defaultdict(float)
        self.position_history: List[Dict] = []
        
    def open_position(self,
                     pair: Tuple[str, str],
                     prices: Tuple[float, float],
                     direction: int,
                     timestamp: datetime) -> Optional[Position]:
        """
        Открытие новой позиции

        Args:
            pair: Торгуемая пара
            prices: Цены входа
            direction: Направление позиции
            timestamp: Время входа

        Returns:
            Optional[Position]: Созданная позиция или None
        """
        try:
            # Проверка возможности открытия позиции
            if not self._can_open_position(pair):
                return None
                
            # Расчет размера позиции
            position_size = self.risk_manager.calculate_position_size(
                pair=pair,
                prices=pd.DataFrame({pair[0]: [prices[0]], pair[1]: [prices[1]]}),
                volatility=None  # TODO: добавить расчет волатильности
            )
            
            # Расчет размеров для каждого актива
            size1 = position_size * self.current_capital / prices[0]
            size2 = position_size * self.current_capital / prices[1]
            
            # Расчет уровней stop-loss и take-profit
            stop_loss, take_profit = self._calculate_exit_levels(
                prices, direction, position_size
            )
            
            # Создание позиции
            position = Position(
                pair=pair,
                entry_time=timestamp,
                entry_prices=prices,
                sizes=(size1, size2),
                direction=direction,
                status=PositionStatus.OPEN,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Учет комиссии
            position.commission = self._calculate_commission(prices, (size1, size2))
            self.current_capital -= position.commission
            
            # Сохранение позиции
            self.active_positions[pair] = position
            self._update_stats(position)
            
            self.logger.info(f"Открыта позиция: {pair}, направление: {direction}, "
                           f"размер: {position_size:.4f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Ошибка при открытии позиции: {str(e)}")
            return None
            
    def close_position(self,
                      pair: Tuple[str, str],
                      prices: Tuple[float, float],
                      timestamp: datetime,
                      reason: str = "signal") -> Optional[Position]:
        """
        Закрытие позиции

        Args:
            pair: Торгуемая пара
            prices: Цены выхода
            timestamp: Время выхода
            reason: Причина закрытия

        Returns:
            Optional[Position]: Закрытая позиция или None
        """
        if pair not in self.active_positions:
            return None
            
        position = self.active_positions[pair]
        
        try:
            # Расчет P&L
            pnl = self._calculate_pnl(position, prices)
            
            # Обновление позиции
            position.exit_time = timestamp
            position.exit_prices = prices
            position.pnl = pnl
            position.status = PositionStatus.CLOSED
            position.exit_reason = reason
            
            # Учет комиссии
            exit_commission = self._calculate_commission(prices, position.sizes)
            position.commission += exit_commission
            pnl -= exit_commission
            
            # Обновление капитала
            self.current_capital += pnl
            
            # Перемещение позиции в историю
            self.closed_positions.append(position)
            del self.active_positions[pair]
            
            self._update_stats(position)
            
            self.logger.info(f"Закрыта позиция: {pair}, P&L: {pnl:.2f}, "
                           f"причина: {reason}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии позиции: {str(e)}")
            return None
            
    def update_positions(self,
                        current_prices: pd.DataFrame,
                        timestamp: datetime) -> List[Position]:
        """
        Обновление и проверка открытых позиций

        Args:
            current_prices: Текущие цены
            timestamp: Текущее время

        Returns:
            List[Position]: Список закрытых позиций
        """
        closed_positions = []
        
        for pair, position in list(self.active_positions.items()):
            prices = (current_prices[pair[0]], current_prices[pair[1]])
            
            # Проверка stop-loss и take-profit
            if self._check_exit_conditions(position, prices):
                reason = "stop_loss" if self._is_stop_loss_hit(position, prices) else "take_profit"
                closed_position = self.close_position(pair, prices, timestamp, reason)
                if closed_position:
                    closed_positions.append(closed_position)
                    
        return closed_positions
        
    def _can_open_position(self, pair: Tuple[str, str]) -> bool:
        """Проверка возможности открытия позиции"""
        if len(self.active_positions) >= self.max_positions:
            return False
        if pair in self.active_positions:
            return False
        return self.risk_manager.check_portfolio_risk()
        
    def _calculate_exit_levels(self,
                             prices: Tuple[float, float],
                             direction: int,
                             position_size: float) -> Tuple[float, float]:
        """Расчет уровней stop-loss и take-profit"""
        spread = np.log(prices[0]) - np.log(prices[1])
        volatility = self.risk_manager.risk_params.get('volatility', 0.02)
        
        stop_loss = spread - direction * volatility * 2
        take_profit = spread + direction * volatility * 3
        
        return stop_loss, take_profit
        
    def _calculate_commission(self,
                            prices: Tuple[float, float],
                            sizes: Tuple[float, float]) -> float:
        """Расчет комиссии"""
        return sum(p * s * self.commission for p, s in zip(prices, sizes))
        
    def _calculate_pnl(self,
                      position: Position,
                      current_prices: Tuple[float, float]) -> float:
        """Расчет P&L позиции"""
        entry_value = sum(p * s for p, s in zip(position.entry_prices, position.sizes))
        current_value = sum(p * s for p, s in zip(current_prices, position.sizes))
        return (current_value - entry_value) * position.direction
        
    def _check_exit_conditions(self,
                             position: Position,
                             current_prices: Tuple[float, float]) -> bool:
        """Проверка условий выхода"""
        spread = np.log(current_prices[0]) - np.log(current_prices[1])
        return (self._is_stop_loss_hit(position, current_prices) or
                self._is_take_profit_hit(position, current_prices))
        
    def _is_stop_loss_hit(self,
                         position: Position,
                         current_prices: Tuple[float, float]) -> bool:
        """Проверка срабатывания stop-loss"""
        spread = np.log(current_prices[0]) - np.log(current_prices[1])
        if position.direction == 1:
            return spread <= position.stop_loss
        return spread >= position.stop_loss
        
    def _is_take_profit_hit(self,
                           position: Position,
                           current_prices: Tuple[float, float]) -> bool:
        """Проверка срабатывания take-profit"""
        spread = np.log(current_prices[0]) - np.log(current_prices[1])
        if position.direction == 1:
            return spread >= position.take_profit
        return spread <= position.take_profit
        
    def _update_stats(self, position: Position) -> None:
        """Обновление статистики"""
        if position.status == PositionStatus.CLOSED:
            self.stats['total_trades'] += 1
            self.stats['total_pnl'] += position.pnl
            self.stats['total_commission'] += position.commission
            
            if position.pnl > 0:
                self.stats['winning_trades'] += 1
                self.stats['gross_profit'] += position.pnl
            else:
                self.stats['losing_trades'] += 1
                self.stats['gross_loss'] += position.pnl
                
        # Обновление истории
        self.position_history.append({
            'timestamp': position.entry_time,
            'pair': position.pair,
            'direction': position.direction,
            'status': position.status.value,
            'pnl': position.pnl,
            'capital': self.current_capital
        })