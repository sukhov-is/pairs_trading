import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
import optuna
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings

@dataclass
class OptimizationResult:
    """Класс для хранения результатов оптимизации"""
    params: Dict[str, float]
    performance: float
    trades_count: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    optimization_time: float
    cross_validation_scores: List[float]

class StrategyOptimizer:
    """
    Класс для оптимизации параметров торговой стратегии
    """
    
    def __init__(self,
                 strategy_class: type,
                 data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 optimization_target: str = 'sharpe',
                 cv_splits: int = 5):
        """
        Инициализация оптимизатора

        Args:
            strategy_class: Класс торговой стратегии
            data: DataFrame с ценовыми данными
            initial_capital: Начальный капитал
            optimization_target: Целевая метрика оптимизации
            cv_splits: Количество разбиений для кросс-валидации
        """
        self.strategy_class = strategy_class
        self.data = data
        self.initial_capital = initial_capital
        self.optimization_target = optimization_target
        self.cv_splits = cv_splits
        
        # Определение границ параметров
        self.param_bounds = {
            'window': (10, 50),
            'entry_zscore': (1.0, 3.0),
            'exit_zscore': (0.0, 1.0),
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.10),
            'position_size': (0.05, 0.30)
        }
        
        self.best_result: Optional[OptimizationResult] = None
        
    def optimize(self, 
                method: str = 'optuna',
                n_trials: int = 100,
                parallel: bool = True) -> OptimizationResult:
        """
        Запуск оптимизации

        Args:
            method: Метод оптимизации ('optuna', 'scipy', 'grid')
            n_trials: Количество итераций
            parallel: Использовать параллельные вычисления

        Returns:
            OptimizationResult: Результаты оптимизации
        """
        if method == 'optuna':
            result = self._optimize_optuna(n_trials, parallel)
        elif method == 'scipy':
            result = self._optimize_scipy()
        elif method == 'grid':
            result = self._optimize_grid(parallel)
        else:
            raise ValueError(f"Неизвестный метод оптимизации: {method}")
            
        self.best_result = result
        return result
    
    def _optimize_optuna(self, n_trials: int, parallel: bool) -> OptimizationResult:
        """Оптимизация с помощью Optuna"""
        import optuna
        
        def objective(trial):
            params = {
                'window': trial.suggest_int('window', 
                    *self.param_bounds['window']),
                'entry_zscore': trial.suggest_float('entry_zscore', 
                    *self.param_bounds['entry_zscore']),
                'exit_zscore': trial.suggest_float('exit_zscore', 
                    *self.param_bounds['exit_zscore']),
                'stop_loss': trial.suggest_float('stop_loss', 
                    *self.param_bounds['stop_loss']),
                'take_profit': trial.suggest_float('take_profit', 
                    *self.param_bounds['take_profit']),
                'position_size': trial.suggest_float('position_size', 
                    *self.param_bounds['position_size'])
            }
            
            return -self._evaluate_params(params)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=-1 if parallel else 1)
        
        best_params = study.best_params
        performance = -study.best_value
        
        return self._create_optimization_result(best_params, performance)
    
    def _optimize_scipy(self) -> OptimizationResult:
        """Оптимизация с помощью scipy.optimize"""
        def objective(x):
            params = {
                'window': int(x[0]),
                'entry_zscore': x[1],
                'exit_zscore': x[2],
                'stop_loss': x[3],
                'take_profit': x[4],
                'position_size': x[5]
            }
            return -self._evaluate_params(params)
        
        x0 = np.array([
            np.mean(self.param_bounds['window']),
            np.mean(self.param_bounds['entry_zscore']),
            np.mean(self.param_bounds['exit_zscore']),
            np.mean(self.param_bounds['stop_loss']),
            np.mean(self.param_bounds['take_profit']),
            np.mean(self.param_bounds['position_size'])
        ])
        
        bounds = [
            self.param_bounds['window'],
            self.param_bounds['entry_zscore'],
            self.param_bounds['exit_zscore'],
            self.param_bounds['stop_loss'],
            self.param_bounds['take_profit'],
            self.param_bounds['position_size']
        ]
        
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')
        
        best_params = {
            'window': int(result.x[0]),
            'entry_zscore': result.x[1],
            'exit_zscore': result.x[2],
            'stop_loss': result.x[3],
            'take_profit': result.x[4],
            'position_size': result.x[5]
        }
        
        return self._create_optimization_result(best_params, -result.fun)
    
    def _optimize_grid(self, parallel: bool) -> OptimizationResult:
        """Оптимизация методом сетки"""
        param_grid = {
            'window': np.linspace(*self.param_bounds['window'], 5).astype(int),
            'entry_zscore': np.linspace(*self.param_bounds['entry_zscore'], 5),
            'exit_zscore': np.linspace(*self.param_bounds['exit_zscore'], 5),
            'stop_loss': np.linspace(*self.param_bounds['stop_loss'], 3),
            'take_profit': np.linspace(*self.param_bounds['take_profit'], 3),
            'position_size': np.linspace(*self.param_bounds['position_size'], 3)
        }
        
        param_combinations = self._generate_param_combinations(param_grid)
        
        if parallel:
            with ProcessPoolExecutor() as executor:
                scores = list(executor.map(self._evaluate_params, param_combinations))
        else:
            scores = [self._evaluate_params(params) for params in param_combinations]
        
        best_idx = np.argmax(scores)
        best_params = param_combinations[best_idx]
        
        return self._create_optimization_result(best_params, scores[best_idx])
    
    def _evaluate_params(self, params: Dict[str, float]) -> float:
        """Оценка параметров с помощью кросс-валидации"""
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Создание и тестирование стратегии
            strategy = self.strategy_class(params=params, 
                                        initial_capital=self.initial_capital)
            strategy.backtest(test_data)
            
            # Расчет метрики
            if self.optimization_target == 'sharpe':
                score = strategy.stats['sharpe_ratio']
            elif self.optimization_target == 'returns':
                score = (strategy.capital - self.initial_capital) / self.initial_capital
            elif self.optimization_target == 'sortino':
                score = self._calculate_sortino_ratio(strategy)
            
            scores.append(score)
            
        return np.mean(scores)
    
    def _calculate_sortino_ratio(self, strategy) -> float:
        """Расчет коэффициента Сортино"""
        returns = pd.Series([t.pnl for t in strategy.trades_history])
        negative_returns = returns[returns < 0]
        
        if len(returns) == 0 or len(negative_returns) == 0:
            return 0
            
        excess_return = returns.mean()
        downside_std = np.sqrt(np.mean(negative_returns ** 2))
        
        return np.sqrt(252) * excess_return / downside_std if downside_std != 0 else 0
    
    def _create_optimization_result(self, 
                                  params: Dict[str, float],
                                  performance: float) -> OptimizationResult:
        """Создание объекта с результатами оптимизации"""
        # Тестирование на всем периоде
        strategy = self.strategy_class(params=params, 
                                     initial_capital=self.initial_capital)
        strategy.backtest(self.data)
        
        return OptimizationResult(
            params=params,
            performance=performance,
            trades_count=len(strategy.trades_history),
            sharpe_ratio=strategy.stats['sharpe_ratio'],
            max_drawdown=strategy.stats['max_drawdown'],
            win_rate=strategy.stats['win_rate'],
            optimization_time=0.0,  # TODO: добавить измерение времени
            cross_validation_scores=[]  # TODO: добавить сохранение scores
        )
    
    def plot_optimization_results(self) -> None:
        """Визуализация результатов оптимизации"""
        if not self.best_result:
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (param, value) in enumerate(self.best_result.params.items()):
            bounds = self.param_bounds[param]
            x = np.linspace(bounds[0], bounds[1], 100)
            y = []
            
            params = self.best_result.params.copy()
            for xi in x:
                params[param] = xi
                y.append(self._evaluate_params(params))
            
            axes[i].plot(x, y)
            axes[i].axvline(value, color='r', linestyle='--')
            axes[i].set_title(param)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Performance')
        
        plt.tight_layout()
        plt.show()