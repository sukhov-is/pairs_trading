import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

class RiskLevel(Enum):
    """Уровни риска"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class PositionRisk:
    """Класс для хранения риск-метрик позиции"""
    value_at_risk: float
    expected_shortfall: float
    max_loss: float
    correlation_risk: float
    liquidity_risk: float
    total_risk_score: float

class RiskManager:
    """
    Класс управления рисками торговой системы
    """
    
    def __init__(self,
                 initial_capital: float,
                 risk_params: Dict[str, float],
                 logger: Optional[logging.Logger] = None):
        """
        Инициализация риск-менеджера

        Args:
            initial_capital: Начальный капитал
            risk_params: Параметры риск-менеджмента
            logger: Логгер для записи событий
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_params = self._validate_risk_params(risk_params)
        self.logger = logger or logging.getLogger(__name__)
        
        # Хранение метрик риска
        self.positions_risk: Dict[Tuple[str, str], PositionRisk] = {}
        self.portfolio_risk: Optional[PositionRisk] = None
        
        # Исторические данные
        self.risk_history: List[Dict] = []
        
    def _validate_risk_params(self, params: Dict) -> Dict:
        """Проверка и установка параметров риска по умолчанию"""
        default_params = {
            'max_position_size': 0.2,      # Максимальный размер позиции (% от капитала)
            'max_portfolio_var': 0.02,     # Максимальный VaR портфеля
            'confidence_level': 0.99,      # Уровень доверия для VaR
            'max_correlation': 0.7,        # Максимальная корреляция между парами
            'min_liquidity_ratio': 0.1,    # Минимальный коэффициент ликвидности
            'max_drawdown': 0.15,          # Максимальная просадка
            'risk_free_rate': 0.02,        # Безрисковая ставка
            'stress_test_scenarios': 100    # Количество сценариев для стресс-тестирования
        }
        
        return {**default_params, **params}
        
    def calculate_position_size(self,
                              pair: Tuple[str, str],
                              prices: pd.DataFrame,
                              volatility: pd.DataFrame) -> float:
        """
        Расчет оптимального размера позиции

        Args:
            pair: Торгуемая пара
            prices: DataFrame с ценами
            volatility: DataFrame с волатильностью

        Returns:
            float: Размер позиции в % от капитала
        """
        # Расчет риск-метрик
        position_risk = self._calculate_position_risk(pair, prices, volatility)
        self.positions_risk[pair] = position_risk
        
        # Определение размера позиции на основе риска
        risk_score = position_risk.total_risk_score
        max_size = self.risk_params['max_position_size']
        
        # Корректировка размера в зависимости от риска
        position_size = max_size * (1 - risk_score)
        
        # Проверка ограничений
        position_size = min(position_size, max_size)
        position_size = max(position_size, 0.0)
        
        return position_size
        
    def _calculate_position_risk(self,
                               pair: Tuple[str, str],
                               prices: pd.DataFrame,
                               volatility: pd.DataFrame) -> PositionRisk:
        """Расчет риск-метрик для позиции"""
        # Расчет VaR
        returns = prices[list(pair)].pct_change().dropna()
        var = self._calculate_var(returns)
        
        # Расчет Expected Shortfall
        es = self._calculate_expected_shortfall(returns)
        
        # Расчет максимальной потери
        max_loss = returns.min().sum()
        
        # Расчет корреляционного риска
        correlation = returns.corr().iloc[0, 1]
        correlation_risk = abs(correlation) / self.risk_params['max_correlation']
        
        # Расчет риска ликвидности
        liquidity_risk = self._calculate_liquidity_risk(pair, prices)
        
        # Общий риск-скор
        total_risk = (0.3 * (var / self.risk_params['max_portfolio_var']) +
                     0.2 * correlation_risk +
                     0.2 * liquidity_risk +
                     0.3 * (abs(max_loss) / self.risk_params['max_drawdown']))
        
        return PositionRisk(
            value_at_risk=var,
            expected_shortfall=es,
            max_loss=max_loss,
            correlation_risk=correlation_risk,
            liquidity_risk=liquidity_risk,
            total_risk_score=min(total_risk, 1.0)
        )
        
    def _calculate_var(self, returns: pd.DataFrame) -> float:
        """Расчет Value at Risk"""
        portfolio_returns = returns.sum(axis=1)
        confidence_level = self.risk_params['confidence_level']
        return abs(portfolio_returns.quantile(1 - confidence_level))
        
    def _calculate_expected_shortfall(self, returns: pd.DataFrame) -> float:
        """Расчет Expected Shortfall (Conditional VaR)"""
        portfolio_returns = returns.sum(axis=1)
        confidence_level = self.risk_params['confidence_level']
        var = self._calculate_var(returns)
        return abs(portfolio_returns[portfolio_returns < -var].mean())
        
    def _calculate_liquidity_risk(self,
                                pair: Tuple[str, str],
                                prices: pd.DataFrame) -> float:
        """Расчет риска ликвидности"""
        # В реальной системе здесь должен быть расчет на основе
        # объемов торгов и спредов bid/ask
        return 0.5  # Упрощенная реализация
        
    def check_portfolio_risk(self) -> bool:
        """Проверка рисков портфеля"""
        if not self.positions_risk:
            return True
            
        # Расчет общего VaR портфеля
        total_var = sum(risk.value_at_risk for risk in self.positions_risk.values())
        
        # Проверка просадки
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        
        # Обновление истории рисков
        self.risk_history.append({
            'timestamp': pd.Timestamp.now(),
            'var': total_var,
            'drawdown': drawdown,
            'capital': self.current_capital
        })
        
        # Проверка ограничений
        if (total_var > self.risk_params['max_portfolio_var'] or
            drawdown > self.risk_params['max_drawdown']):
            self.logger.warning(f"Превышены лимиты риска: VaR={total_var:.4f}, DD={drawdown:.4f}")
            return False
            
        return True
        
    def run_stress_test(self,
                       positions: Dict[Tuple[str, str], float],
                       prices: pd.DataFrame) -> Dict[str, float]:
        """
        Проведение стресс-тестирования портфеля

        Args:
            positions: Текущие позиции {pair: size}
            prices: Исторические цены

        Returns:
            Dict[str, float]: Результаты стресс-теста
        """
        returns = prices.pct_change().dropna()
        n_scenarios = self.risk_params['stress_test_scenarios']
        
        # Генерация сценариев
        scenarios = []
        for _ in range(n_scenarios):
            # Случайная выборка исторических возвратов
            scenario_returns = returns.sample(n=20, replace=True)
            scenario_pnl = 0
            
            # Расчет P&L для каждой позиции
            for pair, size in positions.items():
                pair_returns = scenario_returns[list(pair)].sum(axis=1)
                scenario_pnl += (pair_returns * size).sum()
                
            scenarios.append(scenario_pnl)
            
        return {
            'worst_loss': min(scenarios),
            'var_99': np.percentile(scenarios, 1),
            'expected_shortfall': np.mean([s for s in scenarios if s < np.percentile(scenarios, 1)]),
            'mean_loss': np.mean([s for s in scenarios if s < 0]),
            'probability_of_loss': len([s for s in scenarios if s < 0]) / n_scenarios
        }
        
    def generate_risk_report(self) -> Dict:
        """Генерация отчета по рискам"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'current_capital': self.current_capital,
            'drawdown': (self.initial_capital - self.current_capital) / self.initial_capital,
            'positions_risk': {
                str(pair): {
                    'var': risk.value_at_risk,
                    'es': risk.expected_shortfall,
                    'risk_score': risk.total_risk_score
                }
                for pair, risk in self.positions_risk.items()
            },
            'portfolio_metrics': {
                'total_var': sum(r.value_at_risk for r in self.positions_risk.values()),
                'total_es': sum(r.expected_shortfall for r in self.positions_risk.values()),
                'avg_correlation': np.mean([r.correlation_risk for r in self.positions_risk.values()]),
                'max_position_risk': max(r.total_risk_score for r in self.positions_risk.values())
            }
        }
        
        return report