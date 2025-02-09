import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from datetime import datetime
from src.data_manager import DataManager
from src.pair_finder import PairFinder
from src.strategy import PairTradingStrategy
from src.risk_manager import RiskManager
from src.optimizer import StrategyOptimizer
from utils.logger import Logger
from utils.visualizer import Visualizer
from config.config import STRATEGY_PARAMS, RISK_PARAMS

class PairTradingSystem:
    """Основной класс системы парного трейдинга"""
    
    def __init__(self):
        self.logger = Logger()
        self.visualizer = Visualizer()
        
        # Загрузка конфигурации
        self.strategy_params = STRATEGY_PARAMS
        self.risk_params = RISK_PARAMS
        
    def run(self):
        try:
            # 1. Инициализация и загрузка данных
            self.logger.info("Начало работы системы")
            data_manager = DataManager(
                start_date="2020-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            # Получение и подготовка данных
            tickers = data_manager.get_sp500_tickers()
            data = data_manager.load_data(tickers)
            prepared_data = data_manager.prepare_data()
            
            self.logger.info(f"Загружено {len(prepared_data.columns)} тикеров")
            
            # 2. Поиск пар
            pair_finder = PairFinder(prepared_data)
            potential_pairs = pair_finder.find_pairs()
            self.logger.info(f"Найдено {len(potential_pairs)} потенциальных пар")
            
            # 3. Инициализация менеджеров
            risk_manager = RiskManager(self.risk_params)
            strategy = PairTradingStrategy(self.strategy_params)
            
            # 4. Тестирование на исторических данных
            results = []
            for pair in potential_pairs[:10]:  # Берем топ-10 пар для примера
                pair_data = prepared_data[list(pair)]
                
                # Расчет сигналов
                signals = strategy.calculate_signals(pair_data)
                
                # Проверка риск-менеджмента
                position_size = risk_manager.calculate_position_size(
                    capital=self.strategy_params['initial_capital'],
                    price=pair_data.iloc[-1, 0],  # Последняя цена первого актива
                    volatility=pair_data.std().mean()
                )
                
                # Исполнение торговли
                pair_results = strategy.execute_trades(signals, pair_data)
                
                results.append({
                    'pair': pair,
                    'returns': pair_results['final_capital'] - self.strategy_params['initial_capital'],
                    'trades': pair_results['trades']
                })
                
                # Визуализация результатов по паре
                self.visualizer.plot_pair_trading(pair, pair_data, signals)
            
            # 5. Анализ результатов
            results_df = pd.DataFrame(results)
            best_pairs = results_df.nlargest(3, 'returns')
            
            self.logger.info("Топ-3 лучших пары:")
            for _, row in best_pairs.iterrows():
                self.logger.info(f"Пара: {row['pair']}, Доходность: {row['returns']:.2f}")
            
            # 6. Оптимизация параметров для лучших пар
            optimizer = StrategyOptimizer(strategy, prepared_data)
            for pair in best_pairs['pair']:
                optimal_params = optimizer.optimize(pair)
                self.logger.info(f"Оптимальные параметры для пары {pair}: {optimal_params}")
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Ошибка в работе системы: {str(e)}")
            raise
            
def main():
    """Точка входа в программу"""
    try:
        # Конфигурация логирования
        logging_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s'
        }
        
        # Запуск системы
        system = PairTradingSystem()
        results = system.run()
        
        # Сохранение результатов
        results.to_csv('results/trading_results.csv')
        
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()