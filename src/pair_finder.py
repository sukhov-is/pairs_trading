import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import spearmanr
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import statsmodels.api as sm
from dataclasses import dataclass
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

@dataclass
class PairStats:
    """Класс для хранения статистик пары"""
    pair: Tuple[str, str]
    correlation: float
    cointegration_score: float
    adf_stat: float
    p_value: float
    half_life: float
    hurst_exponent: float
    spread_std: float
    beta: float

class PairFinder:
    """
    Класс для поиска и анализа торговых пар
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 min_correlation: float = 0.7,
                 max_p_value: float = 0.05,
                 min_half_life: int = 1,
                 max_half_life: int = 252,  # ~1 торговый год
                 min_zscore: float = 1.5):
        """
        Инициализация поиска пар

        Args:
            data: DataFrame с ценовыми данными
            min_correlation: Минимальная корреляция
            max_p_value: Максимальное p-value для тестов
            min_half_life: Минимальный период полураспада
            max_half_life: Максимальный период полураспада
            min_zscore: Минимальный z-score для торговых сигналов
        """
        self.data = data
        self.min_correlation = min_correlation
        self.max_p_value = max_p_value
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_zscore = min_zscore
        
        self.pairs_stats: List[PairStats] = []
        self.filtered_pairs: List[Tuple[str, str]] = []
        
    def find_pairs(self, parallel: bool = True) -> List[Tuple[str, str]]:
        """
        Основной метод поиска пар
        
        Args:
            parallel: Использовать параллельные вычисления

        Returns:
            List[Tuple[str, str]]: Список отфильтрованных пар
        """
        # Получение всех возможных комбинаций пар
        all_pairs = list(combinations(self.data.columns, 2))
        
        try:
            if parallel:
                # Параллельная обработка
                with ProcessPoolExecutor() as executor:
                    self.pairs_stats = list(executor.map(self._analyze_pair, all_pairs))
            else:
                # Последовательная обработка
                self.pairs_stats = [self._analyze_pair(pair) for pair in all_pairs]
            
            # Фильтрация результатов
            self.filtered_pairs = self._filter_pairs()
            
            # Кластеризация пар
            self._cluster_pairs()
            
            return self.filtered_pairs
            
        except Exception as e:
            raise Exception(f"Ошибка при поиске пар: {str(e)}")
    
    def _analyze_pair(self, pair: Tuple[str, str]) -> PairStats:
        """Анализ одной пары"""
        stock1 = self.data[pair[0]]
        stock2 = self.data[pair[1]]
        
        # Расчет корреляции
        correlation, _ = spearmanr(stock1, stock2)
        
        # Тест на коинтеграцию
        score, p_value, _ = coint(stock1, stock2)
        
        # ADF тест на стационарность спреда
        spread = np.log(stock1) - np.log(stock2)
        adf_result = adfuller(spread)
        
        # Расчет периода полураспада
        half_life = self._calculate_half_life(spread)
        
        # Расчет экспоненты Херста
        hurst = self._calculate_hurst_exponent(spread)
        
        # Расчет беты
        beta = self._calculate_beta(stock1, stock2)
        
        return PairStats(
            pair=pair,
            correlation=correlation,
            cointegration_score=score,
            adf_stat=adf_result[0],
            p_value=p_value,
            half_life=half_life,
            hurst_exponent=hurst,
            spread_std=spread.std(),
            beta=beta
        )
    
    def _filter_pairs(self) -> List[Tuple[str, str]]:
        """Фильтрация пар по критериям"""
        filtered = []
        
        for stats in self.pairs_stats:
            if (abs(stats.correlation) >= self.min_correlation and
                stats.p_value <= self.max_p_value and
                self.min_half_life <= stats.half_life <= self.max_half_life and
                stats.hurst_exponent < 0.5):  # Признак возврата к среднему
                filtered.append(stats.pair)
                
        return filtered
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Расчет периода полураспада"""
        lag_spread = spread.shift(1)
        delta_spread = spread - lag_spread
        lag_spread = lag_spread.dropna()
        delta_spread = delta_spread.dropna()
        
        # Регрессия для расчета скорости возврата к среднему
        model = sm.OLS(delta_spread, lag_spread)
        result = model.fit()
        
        return -np.log(2) / result.params[0] if result.params[0] < 0 else np.inf
    
    def _calculate_hurst_exponent(self, spread: pd.Series, lags: List[int] = None) -> float:
        """Расчет экспоненты Херста"""
        if lags is None:
            lags = [2, 4, 8, 16, 32, 64]
            
        tau = []
        rs = []
        
        for lag in lags:
            tau.append(lag)
            rs.append(self._calculate_rs(spread, lag))
            
        return np.polyfit(np.log(tau), np.log(rs), 1)[0]
    
    def _calculate_rs(self, spread: pd.Series, lag: int) -> float:
        """Вспомогательный метод для расчета R/S"""
        # Разделение на подпериоды
        values = pd.Series([])
        for i in range(0, len(spread) - lag, lag):
            chunk = spread.iloc[i:i + lag]
            values = values.append(pd.Series(chunk.mean()))
            
        return values.std() / values.diff().std()
    
    def _calculate_beta(self, stock1: pd.Series, stock2: pd.Series) -> float:
        """Расчет беты между активами"""
        returns1 = stock1.pct_change().dropna()
        returns2 = stock2.pct_change().dropna()
        
        # Регрессия доходностей
        model = sm.OLS(returns1, returns2)
        result = model.fit()
        
        return result.params[0]
    
    def _cluster_pairs(self, n_clusters: int = 5) -> None:
        """Кластеризация пар по характеристикам"""
        if not self.pairs_stats:
            return
            
        # Подготовка данных для кластеризации
        features = np.array([[
            stats.correlation,
            stats.cointegration_score,
            stats.half_life,
            stats.spread_std
        ] for stats in self.pairs_stats])
        
        # Нормализация
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Сохранение результатов
        self.clusters = {i: [] for i in range(n_clusters)}
        for pair_stats, cluster in zip(self.pairs_stats, clusters):
            self.clusters[cluster].append(pair_stats.pair)
    
    def plot_pair_analysis(self, pair: Tuple[str, str]) -> None:
        """Визуализация анализа пары"""
        stock1 = self.data[pair[0]]
        stock2 = self.data[pair[1]]
        spread = np.log(stock1) - np.log(stock2)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # График цен
        axes[0].plot(stock1, label=pair[0])
        axes[0].plot(stock2, label=pair[1])
        axes[0].set_title('Price Series')
        axes[0].legend()
        
        # График спреда
        axes[1].plot(spread)
        axes[1].set_title('Spread')
        axes[1].axhline(y=spread.mean(), color='r', linestyle='--')
        axes[1].axhline(y=spread.mean() + spread.std(), color='g', linestyle='--')
        axes[1].axhline(y=spread.mean() - spread.std(), color='g', linestyle='--')
        
        # График автокорреляции спреда
        sm.graphics.tsa.plot_acf(spread.dropna(), lags=40, ax=axes[2])
        axes[2].set_title('Spread Autocorrelation')
        
        plt.tight_layout()
        plt.show()
    
    def get_pair_stats(self, pair: Tuple[str, str]) -> Optional[PairStats]:
        """Получение статистик для конкретной пары"""
        for stats in self.pairs_stats:
            if stats.pair == pair:
                return stats
        return None
    
    def get_best_pairs(self, n: int = 10) -> List[Tuple[str, str]]:
        """Получение лучших пар по комбинированному скору"""
        if not self.pairs_stats:
            return []
            
        # Расчет комбинированного скора
        scored_pairs = []
        for stats in self.pairs_stats:
            score = (abs(stats.correlation) +
                    (1 / stats.p_value if stats.p_value > 0 else np.inf) +
                    (1 / stats.half_life if stats.half_life > 0 else 0))
            scored_pairs.append((stats.pair, score))
            
        # Сортировка по скору
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [pair for pair, _ in scored_pairs[:n]]