import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    """
    Класс для визуализации данных и результатов торговли
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Настройка стиля графиков
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_pair_trading(self, pair: Tuple[str, str], 
                         data: pd.DataFrame, 
                         signals: pd.DataFrame,
                         save: bool = True) -> None:
        """
        Визуализация торговли парой активов
        
        Args:
            pair: кортеж с тикерами пары
            data: DataFrame с ценами
            signals: DataFrame с сигналами
            save: сохранить график в файл
        """
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Asset Prices', 'Spread', 'Z-Score'))
        
        # График цен активов
        fig.add_trace(
            go.Scatter(x=data.index, y=data[pair[0]], 
                      name=pair[0], line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data[pair[1]], 
                      name=pair[1], line=dict(color='red')),
            row=1, col=1
        )
        
        # График спреда
        spread = np.log(data[pair[0]]) - np.log(data[pair[1]])
        fig.add_trace(
            go.Scatter(x=data.index, y=spread, 
                      name='Spread', line=dict(color='green')),
            row=2, col=1
        )
        
        # График z-score и сигналов
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['z_score'], 
                      name='Z-Score', line=dict(color='purple')),
            row=3, col=1
        )
        
        # Добавление линий входа/выхода
        fig.add_hline(y=2.0, line_dash="dash", line_color="gray", row=3, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="gray", row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)
        
        # Настройка макета
        fig.update_layout(
            height=900,
            title_text=f"Pair Trading Analysis: {pair[0]} - {pair[1]}",
            showlegend=True
        )
        
        if save:
            fig.write_html(self.output_dir / f"pair_trading_{pair[0]}_{pair[1]}.html")
        else:
            fig.show()
            
    def plot_performance_metrics(self, results: pd.DataFrame) -> None:
        """
        Визуализация метрик производительности
        
        Args:
            results: DataFrame с результатами торговли
        """
        fig = plt.figure(figsize=(15, 10))
        
        # График доходности
        plt.subplot(2, 2, 1)
        plt.plot(results['cumulative_returns'])
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        
        # Гистограмма доходности
        plt.subplot(2, 2, 2)
        results['returns'].hist(bins=50)
        plt.title('Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        
        # Drawdown
        plt.subplot(2, 2, 3)
        plt.plot(results['drawdown'])
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        
        # Статистика по месяцам
        plt.subplot(2, 2, 4)
        monthly_returns = results['returns'].resample('M').sum()
        monthly_returns.plot(kind='bar')
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Return')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png')
        plt.close()
        
    def plot_correlation_matrix(self, data: pd.DataFrame) -> None:
        """
        Визуализация корреляционной матрицы
        
        Args:
            data: DataFrame с ценами активов
        """
        corr = data.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.savefig(self.output_dir / 'correlation_matrix.png')
        plt.close()