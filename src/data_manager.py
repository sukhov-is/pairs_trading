import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import requests
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class DataManager:
    """
    Класс для загрузки, обработки и управления данными
    """
    
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 data_dir: str = "data",
                 use_cache: bool = True,
                 cache_expire_days: int = 1):
        """
        Инициализация менеджера данных

        Args:
            start_date: Начальная дата в формате 'YYYY-MM-DD'
            end_date: Конечная дата в формате 'YYYY-MM-DD'
            data_dir: Директория для хранения кэшированных данных
            use_cache: Использовать ли кэширование данных
            cache_expire_days: Срок действия кэша в днях
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.cache_expire_days = cache_expire_days
        
        # Инициализация хранилищ данных
        self.price_data: Optional[pd.DataFrame] = None
        self.volume_data: Optional[pd.DataFrame] = None
        self.tickers: List[str] = []
        
    def get_sp500_tickers(self) -> List[str]:
        """Получение списка тикеров S&P 500"""
        cache_file = self.data_dir / "sp500_tickers.pkl"
        
        # Проверка кэша
        if self.use_cache and cache_file.exists():
            if self._is_cache_valid(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].tolist()
            
            # Обработка символов
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            # Сохранение в кэш
            if self.use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(tickers, f)
            
            self.tickers = tickers
            return tickers
            
        except Exception as e:
            raise Exception(f"Ошибка при получении тикеров S&P 500: {str(e)}")
            
    def load_data(self, 
                  tickers: Optional[List[str]] = None, 
                  parallel: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка данных для заданных тикеров

        Args:
            tickers: Список тикеров для загрузки
            parallel: Использовать параллельную загрузку

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (price_data, volume_data)
        """
        if tickers is None:
            tickers = self.tickers
            
        cache_file = self.data_dir / f"market_data_{self.start_date}_{self.end_date}.pkl"
        
        # Проверка кэша
        if self.use_cache and cache_file.exists():
            if self._is_cache_valid(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.price_data = data['prices']
                    self.volume_data = data['volumes']
                    return self.price_data, self.volume_data
        
        try:
            if parallel:
                # Параллельная загрузка данных
                with ThreadPoolExecutor() as executor:
                    download_func = partial(self._download_ticker_data)
                    results = list(executor.map(download_func, tickers))
                
                # Объединение результатов
                prices_list = []
                volumes_list = []
                for price_data, volume_data in results:
                    if price_data is not None:
                        prices_list.append(price_data)
                        volumes_list.append(volume_data)
                
                self.price_data = pd.concat(prices_list, axis=1)
                self.volume_data = pd.concat(volumes_list, axis=1)
                
            else:
                # Последовательная загрузка
                self.price_data = yf.download(tickers, 
                                            start=self.start_date,
                                            end=self.end_date)['Close']
                self.volume_data = yf.download(tickers,
                                             start=self.start_date,
                                             end=self.end_date)['Volume']
            
            # Сохранение в кэш
            if self.use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'prices': self.price_data,
                        'volumes': self.volume_data
                    }, f)
            
            return self.price_data, self.volume_data
            
        except Exception as e:
            raise Exception(f"Ошибка при загрузке данных: {str(e)}")
            
    def prepare_data(self, 
                    threshold: float = 0.1,
                    min_volume: float = 100000,
                    min_price: float = 5.0) -> pd.DataFrame:
        """
        Подготовка и фильтрация данных

        Args:
            threshold: Максимальная доля пропущенных значений
            min_volume: Минимальный средний объем торгов
            min_price: Минимальная цена актива

        Returns:
            pd.DataFrame: Подготовленные данные
        """
        if self.price_data is None:
            raise ValueError("Данные не загружены. Сначала выполните load_data()")
            
        try:
            # Копия данных
            data = self.price_data.copy()
            
            # Фильтрация по пропущенным значениям
            data = data.dropna(thresh=int((1 - threshold) * len(data)), axis=1)
            
            # Фильтрация по объему
            if self.volume_data is not None:
                avg_volume = self.volume_data.mean()
                valid_volume = avg_volume[avg_volume > min_volume].index
                data = data[valid_volume]
            
            # Фильтрация по цене
            avg_price = data.mean()
            valid_price = avg_price[avg_price > min_price].index
            data = data[valid_price]
            
            # Заполнение оставшихся пропусков
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Расчет дополнительных характеристик
            self._calculate_statistics(data)
            
            return data
            
        except Exception as e:
            raise Exception(f"Ошибка при подготовке данных: {str(e)}")
    
    def _download_ticker_data(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Загрузка данных для одного тикера"""
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            return pd.DataFrame({ticker: data['Close']}), pd.DataFrame({ticker: data['Volume']})
        except Exception:
            return None, None
            
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Проверка валидности кэша"""
        if not cache_file.exists():
            return False
            
        # Проверка времени создания файла
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        cache_age = datetime.now() - file_time
        
        return cache_age.days < self.cache_expire_days
        
    def _calculate_statistics(self, data: pd.DataFrame) -> None:
        """Расчет статистических характеристик"""
        self.statistics = {
            'volatility': data.pct_change().std(),
            'returns': data.pct_change().mean(),
            'sharpe': self._calculate_sharpe_ratio(data),
            'correlation_matrix': data.corr()
        }
        
    def _calculate_sharpe_ratio(self, data: pd.DataFrame, risk_free_rate: float = 0.01) -> pd.Series:
        """Расчет коэффициента Шарпа"""
        returns = data.pct_change()
        excess_returns = returns - risk_free_rate/252  # Дневной risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def get_market_data(self, 
                       tickers: List[str], 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Получение рыночных данных для заданных тикеров и периода
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        if self.price_data is None:
            self.load_data(tickers)
            
        mask = (self.price_data.index >= start_date) & (self.price_data.index <= end_date)
        return self.price_data.loc[mask, tickers]