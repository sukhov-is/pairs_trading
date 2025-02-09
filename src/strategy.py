import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import statsmodels.api as sm
from scipy.stats import norm

class Position(Enum):
    """Enum для типов позиций"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

@dataclass
class Trade:
    """Класс для хранения информации о сделке"""
    timestamp: pd.Timestamp
    pair: Tuple[str, str]
    position: Position
    entry_prices: Tuple[float, float]
    entry_sizes: Tuple[float, float]
    exit_prices: Optional[Tuple[float, float]] = None
    exit_sizes: Optional[Tuple[float, float]] = None
    pnl: float = 0.0
    exit_reason: str = ""

class PairTradingStrategy:
    """
    Класс реализации стратегии парного трейдинга
    """
    
    def __init__(self,
                 pair: Tuple[str, str],
                 params: Dict,
                 initial_capital: float = 100000.0):
        """
        Инициализация стратегии

        Args:
            pair: Торгуемая пара
            params: Параметры стратегии
            initial_capital: Начальный капитал
        """
        self.pair = pair
        self.params = self._validate_params(params)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Хранение состояния
        self.current_position = Position.NEUTRAL
        self.active_trade: Optional[Trade] = None
        self.trades_history: List[Trade] = []
        self.equity_curve = []
        
        # Статистики
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def _validate_params(self, params: Dict) -> Dict:
        """Проверка и установка параметров по умолчанию"""
        default_params = {
            'window': 20,              # Окно для расчета z-score
            'entry_zscore': 2.0,       # Z-score для входа
            'exit_zscore': 0.5,        # Z-score для выхода
            'stop_loss': 0.02,         # Стоп-лосс (2%)
            'take_profit': 0.03,       # Тейк-профит (3%)
            'position_size': 0.1,      # Размер позиции (10% от капитала)
            'commission': 0.001,       # Комиссия (0.1%)
            'beta_hedge': True,        # Использовать бета-хеджирование
            'vol_adjust': True,        # Корректировка на волатильность
            'max_positions': 1         # Максимальное количество одновременных позиций
        }
        
        return {**default_params, **params}
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет торговых сигналов

        Args:
            data: DataFrame с ценами активов пары

        Returns:
            DataFrame с сигналами
        """
        # Расчет спреда
        spread = self._calculate_spread(data)
        
        # Расчет z-score
        zscore = self._calculate_zscore(spread)
        
        # Создание DataFrame с сигналами
        signals = pd.DataFrame(index=data.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        
        # Сигналы входа
        signals['long_entry'] = zscore <= -self.params['entry_zscore']
        signals['short_entry'] = zscore >= self.params['entry_zscore']
        
        # Сигналы выхода
        signals['exit'] = abs(zscore) <= self.params['exit_zscore']
        
        # Добавление стоп-сигналов
        if self.active_trade:
            signals['stop_loss'] = self._check_stop_loss(data)
            signals['take_profit'] = self._check_take_profit(data)
        
        return signals
    
    def _calculate_spread(self, data: pd.DataFrame) -> pd.Series:
        """Расчет спреда между активами"""
        if self.params['beta_hedge']:
            # Расчет беты для хеджирования
            returns1 = data[self.pair[0]].pct_change()
            returns2 = data[self.pair[1]].pct_change()
            model = sm.OLS(returns1, returns2)
            beta = model.fit().params[0]
            
            # Расчет спреда с учетом беты
            spread = np.log(data[self.pair[0]]) - beta * np.log(data[self.pair[1]])
        else:
            # Простой лог-спред
            spread = np.log(data[self.pair[0]]) - np.log(data[self.pair[1]])
            
        return spread
    
    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Расчет z-score"""
        rolling_mean = spread.rolling(window=self.params['window']).mean()
        rolling_std = spread.rolling(window=self.params['window']).std()
        
        return (spread - rolling_mean) / rolling_std
    
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        """
        Исполнение торговых сигналов

        Args:
            data: DataFrame с ценами
            signals: DataFrame с сигналами
        """
        for timestamp in signals.index:
            current_prices = data.loc[timestamp]
            current_signals = signals.loc[timestamp]
            
            # Проверка выхода из позиции
            if self.current_position != Position.NEUTRAL:
                if self._should_exit(current_signals):
                    self._close_position(timestamp, current_prices)
                    continue
            
            # Проверка входа в позицию
            if self.current_position == Position.NEUTRAL:
                if current_signals['long_entry']:
                    self._open_position(timestamp, current_prices, Position.LONG)
                elif current_signals['short_entry']:
                    self._open_position(timestamp, current_prices, Position.SHORT)
    
    def _should_exit(self, signals: pd.Series) -> bool:
        """Проверка условий выхода"""
        return (signals['exit'] or 
                signals.get('stop_loss', False) or 
                signals.get('take_profit', False))
    
    def _open_position(self, 
                      timestamp: pd.Timestamp,
                      prices: pd.Series,
                      position_type: Position) -> None:
        """Открытие позиции"""
        if len(self.trades_history) >= self.params['max_positions']:
            return
            
        # Расчет размеров позиций
        position_value = self.capital * self.params['position_size']
        if self.params['vol_adjust']:
            # Корректировка на относительную волатильность
            vol1 = prices[self.pair[0]].rolling(window=self.params['window']).std().iloc[-1]
            vol2 = prices[self.pair[1]].rolling(window=self.params['window']).std().iloc[-1]
            ratio = vol1 / vol2
            size1 = position_value / (1 + ratio)
            size2 = position_value - size1
        else:
            size1 = size2 = position_value / 2
            
        # Создание сделки
        self.active_trade = Trade(
            timestamp=timestamp,
            pair=self.pair,
            position=position_type,
            entry_prices=(prices[self.pair[0]], prices[self.pair[1]]),
            entry_sizes=(size1, size2)
        )
        
        self.current_position = position_type
        
    def _close_position(self, 
                       timestamp: pd.Timestamp,
                       prices: pd.Series,
                       reason: str = "signal") -> None:
        """Закрытие позиции"""
        if not self.active_trade:
            return
            
        # Расчет P&L
        entry_value = (self.active_trade.entry_prices[0] * self.active_trade.entry_sizes[0] +
                      self.active_trade.entry_prices[1] * self.active_trade.entry_sizes[1])
        exit_value = (prices[self.pair[0]] * self.active_trade.entry_sizes[0] +
                     prices[self.pair[1]] * self.active_trade.entry_sizes[1])
                     
        pnl = (exit_value - entry_value) * self.active_trade.position.value
        pnl -= entry_value * self.params['commission'] * 2  # Комиссия за вход и выход
        
        # Обновление сделки
        self.active_trade.exit_prices = (prices[self.pair[0]], prices[self.pair[1]])
        self.active_trade.exit_sizes = self.active_trade.entry_sizes
        self.active_trade.pnl = pnl
        self.active_trade.exit_reason = reason
        
        # Обновление статистик
        self.trades_history.append(self.active_trade)
        self.capital += pnl
        self.equity_curve.append((timestamp, self.capital))
        
        # Сброс текущей позиции
        self.active_trade = None
        self.current_position = Position.NEUTRAL
        
    def update_statistics(self) -> None:
        """Обновление статистик торговли"""
        if not self.trades_history:
            return
            
        self.stats['total_trades'] = len(self.trades_history)
        self.stats['winning_trades'] = sum(1 for t in self.trades_history if t.pnl > 0)
        self.stats['losing_trades'] = sum(1 for t in self.trades_history if t.pnl <= 0)
        
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = self.stats['winning_trades'] / self.stats['total_trades']
            
        winning_pnls = [t.pnl for t in self.trades_history if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.trades_history if t.pnl <= 0]
        
        self.stats['avg_profit'] = np.mean(winning_pnls) if winning_pnls else 0
        self.stats['avg_loss'] = np.mean(losing_pnls) if losing_pnls else 0
        
        # Расчет максимальной просадки
        equity = pd.Series([e[1] for e in self.equity_curve])
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        self.stats['max_drawdown'] = drawdowns.min()
        
        # Расчет коэффициента Шарпа
        returns = equity.pct_change().dropna()
        self.stats['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()