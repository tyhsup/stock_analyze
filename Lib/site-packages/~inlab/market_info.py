import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

import sys
if sys.platform != 'emscripten':
    from multiprocessing import shared_memory

import finlab.data

class MarketInfo():
    """市場類別
    假如希望開發新的交易市場套用到回測系統，可以繼承 `finlab.market_info.MarketInfo` 來實做新類別。
    """
    @staticmethod
    def get_freq() -> str:
        """
        Returns the frequency of the data.
        Used to determine how to resample the data when the data is not daily.
        The freq will be saved in `finlab.core.report`.
        
        Returns:
        str: The frequency of the data.
        """
        return '1d'

    @staticmethod
    def get_name() -> str:
        """
        Returns the name of the market data source.
        
        This function is used to get the name of the market data source.
        """
        return 'auto'

    @staticmethod
    def get_benchmark() -> pd.Series:
        """設定對標報酬率的時間序列
        
        這個函數用於設定對標報酬率的時間序列。

        Returns:
            pd.Series: 時間序列的報酬率。

        Raises:
            ExceptionType: Description of conditions under which the exception is raised.

        Examples:
            | date       |   0050 |
            |:-----------|-------:|
            | 2007-04-23 |   100 |
            | 2007-04-24 |   100.1 |
            | 2007-04-25 |   99 |
            | 2007-04-26 |   98.3 |
            | 2007-04-27 |   99.55 |
        """
        return pd.Series([], index=pd.Index([], dtype='datetime64[ns]'), dtype='float64')

    @staticmethod
    def get_asset_id_to_name() -> Dict:
        """設定對標報酬率的時間序列
        Returns:
          (dict): 股號與股名對照表，ex:`{'2330':'台積電'}`
        """
        return {}

    @staticmethod
    def get_price(trade_at_price:str, adj:bool=True) -> pd.DataFrame:
        """取得回測用價格數據

        Args:
           trade_at_price (str): 選擇回測之還原股價以收盤價或開盤價計算，預設為'close'。可選'close'或'open'。
           adj (str): 是否使用還原股價計算。
        
        Returns:
          (pd.DataFrame): 價格數據
        
        Examples:
            格式範例

            | date       |   0015 |   0050 |   0051 |   0052 |
            |:-----------|-------:|-------:|-------:|-------:|
            | 2007-04-23 |   9.54 |  57.85 |  32.83 |  38.4  |
            | 2007-04-24 |   9.54 |  58.1  |  32.99 |  38.65 |
            | 2007-04-25 |   9.52 |  57.6  |  32.8  |  38.59 |
            | 2007-04-26 |   9.59 |  57.7  |  32.8  |  38.6  |
            | 2007-04-27 |   9.55 |  57.5  |  32.72 |  38.4  |
            """
        return pd.DataFrame()

    @staticmethod
    def get_market_value():
        """取得回測用市值數據

        Returns:
          (pd.DataFrame): 市值數據，其中 index 為日期，而 columns 是股票代號。

        """
        return pd.DataFrame()

    def get_trading_price(self, name: str, adj=True) -> pd.DataFrame:

        """取得回測用價格數據
        
        Args:
            name (str): 選擇回測之還原股價以收盤價或開盤價計算，預設為'close'。可選 'open'、'close'、'high'、'low'、'open_close_avg'、'high_low_avg'、或 'price_avg'。
        Returns:
            (pd.DataFrame): 價格數據

        """


        if name in ['open', 'close', 'high', 'low']:
            return self.get_price(name, adj=adj)
        elif name == 'close_open_avg' or name == 'open_close_avg':
            return (self.get_price('open', adj=adj) + self.get_price('close', adj=adj)) / 2
        elif name == 'high_low_avg' or name == 'low_high_avg':
            return (self.get_price('high', adj=adj) + self.get_price('low', adj=adj)) / 2
        elif name == 'price_avg':
            return (self.get_price('open', adj=adj) + self.get_price('close', adj=adj)\
                     + self.get_price('high', adj=adj) + self.get_price('low', adj=adj)) / 4

        raise ValueError(f"Unknown trade price name: {name}")
    

    def market_close_at_timestamp(self, timestamp=None):
        """
        Returns the timestamp of the market close of the given timestamp.

        Args:
            timestamp (datetime): The timestamp to find the market close to.

        Returns:
            datetime: The timestamp of the closest market close.
        """

        indexes = self.get_price('close').index

        if timestamp is None:
            timestamp = indexes[-1]

        # find min delta between indexes
        delta = np.abs(indexes[1:] - indexes[:-1]).min()
        return timestamp + delta
    
    def get_reference_price(self):
        """Returns the most recent reference price of the market.
        
        Returns:
            pandas.Series: The most recent reference price of the market.
        """
        return self.get_price('close').iloc[-1]
    
    @staticmethod
    def tzinfo():
        """Returns the timezone of the market.
        
        Returns:
            datetime.timezone: The timezone of the market.
        """
        return None

    @staticmethod
    def get_odd_lot():
        """Returns the odd lot size of the market.

        Returns:
            int: The odd lot size of the market.
        """
        return 1


class TWMarketInfo(MarketInfo):

    @staticmethod
    def get_freq():
        return '1d'

    @staticmethod
    def get_name():
        return 'tw_stock'

    @staticmethod
    def get_benchmark() -> pd.Series:
        return finlab.data.get('benchmark_return:發行量加權股價報酬指數').squeeze()

    @staticmethod
    def get_asset_id_to_name():
        categories = finlab.data.get('security_categories')
        stock_names = dict(
            zip(categories.reset_index()['stock_id'], categories['name']))

        return stock_names

    @staticmethod
    def get_price(trade_at_price, adj=True):
        if isinstance(trade_at_price, pd.Series):
            return trade_at_price.to_frame()

        if isinstance(trade_at_price, pd.DataFrame):
            return trade_at_price

        if isinstance(trade_at_price, str):
            if trade_at_price == 'volume':
                return finlab.data.get('price:成交股數')

            if adj:
                table_name = 'etl:adj_'
                price_name = trade_at_price
            else:
                table_name = 'price:'
                price_name = {'open': '開盤價', 'close': '收盤價', 'high': '最高價', 'low': '最低價'}[trade_at_price]

            price = finlab.data.get(f'{table_name}{price_name}')
            return price

        raise ValueError('trade_at_price is not allowed (accepted types: pd.DataFrame, pd.Series, str).')

    @staticmethod
    def get_market_value():
        return finlab.data.get('etl:market_value')
    

    def market_close_at_timestamp(self, timestamp=None):

        if not isinstance(timestamp, pd.Timestamp) and timestamp is not None:
            timestamp = pd.Timestamp(timestamp).tz_localize('Asia/Taipei')

        if timestamp is None:
            timestamp = pd.Timestamp.now().tz_localize('Asia/Taipei')

        # check if timestamp is localized, if not, localize it to Asia/Taipei
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('Asia/Taipei')
        
        # get market close time
        market_close = pd.Timestamp(timestamp.date()) + pd.Timedelta('15:00:00')
        return market_close.tz_localize('Asia/Taipei')
    
    @staticmethod
    def tzinfo():
        return datetime.timezone(datetime.timedelta(hours=8))
    
    @staticmethod
    def get_reference_price():
        ref_price = finlab.data.get('reference_price')
        return ref_price.set_index('stock_id')['收盤價'].to_dict()

    @staticmethod
    def get_odd_lot():
        return 1000


class USMarketInfo(MarketInfo):

    @staticmethod
    def get_freq():
        return '1d'

    @staticmethod
    def get_name():
        return 'us_stock'

    @staticmethod
    def get_benchmark():
        return finlab.data.get('world_index:adj_close')['^GSPC']

    @staticmethod
    def get_asset_id_to_name():
        categories = finlab.data.get('us_tickers')
        stock_names = dict(
            zip(categories['stock_id'], categories['name']))
        return stock_names

    @staticmethod
    def get_price(trade_at_price, adj=True):
        if isinstance(trade_at_price, pd.Series):
            return trade_at_price.to_frame()

        if isinstance(trade_at_price, pd.DataFrame):
            return trade_at_price

        if isinstance(trade_at_price, str):
            if trade_at_price == 'volume':
                return finlab.data.get('us_price:volume')

            if adj:
                table_name = 'us_price:adj_'
                price_name = trade_at_price
            else:
                table_name = 'us_price:'
                price_name = trade_at_price

            price = finlab.data.get(f'{table_name}{price_name}')
            return price

        raise ValueError('trade_at_price is not allowed. Accepted types are pd.DataFrame, pd.Series, and str.')
    

    def market_close_at_timestamp(self, timestamp=None):

        if timestamp is None:
            timestamp = self.get_price('close').index[-1]

        # check if timestamp is localized, if not, localize it to US/Eastern
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            timestamp = timestamp.tz_localize('US/Eastern')

        # get market close time
        market_close = pd.Timestamp(timestamp.date()) + pd.Timedelta('16:00:00')
        return market_close.tz_localize('US/Eastern')

    @staticmethod    
    def tzinfo():
        return datetime.timezone(datetime.timedelta(hours=-4))


class USAllMarketInfo(USMarketInfo):
    @staticmethod
    def get_price(trade_at_price, adj=True):
        if isinstance(trade_at_price, pd.Series):
            return trade_at_price.to_frame()

        if isinstance(trade_at_price, pd.DataFrame):
            return trade_at_price

        if isinstance(trade_at_price, str):
            if trade_at_price == 'volume':
                return finlab.data.get('us_price_all:volume')

            if adj:
                table_name = 'us_price_all:adj_'
                price_name = trade_at_price
            else:
                table_name = 'us_price_all:'
                price_name = trade_at_price

            price = finlab.data.get(f'{table_name}{price_name}')
            return price

        raise Exception(f'**ERROR: trade_at_price is not allowed (accepted types: pd.DataFrame, pd.Series, str).')

    @staticmethod
    def get_market_value():
        return finlab.data.get('us_fundamental:marketcap')


def get_market_info(df:Union[None, pd.DataFrame, pd.Series]=None, 
                    user_market_info:Union[None, str, MarketInfo]='AUTO') -> Optional[MarketInfo]:

    
    # return market info base on user_market_info
    if user_market_info == 'TW_STOCK':
        return TWMarketInfo()

    if user_market_info == 'US_STOCK':
        return USMarketInfo()

    if user_market_info == 'US_STOCK_ALL':
        return USAllMarketInfo()

    if user_market_info != 'AUTO':
        return user_market_info

    # deal with user_market_info == 'AUTO'
    if df is not None:
        ids = set([df.name]) if isinstance(df, pd.Series) else set(df.columns)
        for market in [TWMarketInfo(), USMarketInfo()]:
            market_ids = set(market.get_asset_id_to_name().keys())
            id_not_found = ids - market_ids

            if len(id_not_found) / len(ids) < 0.5:
                if len(id_not_found):
                    logging.warning(f"Symbols {str(id_not_found)[:30]}... not found in MarketInfo.")
                return market

    raise Exception("Market cannot be determined automatically. "
                    "Please set the market argument to either "
                    "'TW_STOCK' for Taiwan stock market, 'US_STOCK' "
                    "for US stock market, or use subclass of MarketInfo"
                    " to specify a different market.")


def create_shared_memory_from_df(df, name):
    """Create a shared memory from a pandas DataFrame."""

    names = [name, name+'_index', name+'_columns']
    for name in names:
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()

        except FileNotFoundError:
            pass

    # create
    array = df.to_numpy()
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes, name=name)
    np_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    np_array[:] = array[:]

    # index
    array_index = df.index.to_numpy()
    shm_index = shared_memory.SharedMemory(create=True, size=array_index.nbytes, name=name+'_index')
    np_array_index = np.ndarray(array_index.shape, dtype=array_index.dtype, buffer=shm_index.buf)
    np_array_index[:] = array_index[:]

    # columns
    array_columns = df.columns.to_numpy()
    shm_columns = shared_memory.SharedMemory(create=True, size=array_columns.nbytes, name=name+'_columns')
    np_array_columns = np.ndarray(array_columns.shape, dtype=array_columns.dtype, buffer=shm_columns.buf)
    np_array_columns[:] = array_columns[:]
    # return shm, array.shape, array.dtype

    return shm, array.shape, array.dtype, \
        shm_index, array_index.shape, array_index.dtype, \
        shm_columns, array_columns.shape, array_columns.dtype

def create_df_from_shared_memory(shm, shape, dtype, shm_index, index_shape, index_dtype, shm_columns, columns_shape, columns_dtype):
    """Create a pandas DataFrame from a shared memory."""

    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf) # np.frombuffer(shm.buf).reshape(shape)#
    np_array_index = np.ndarray(index_shape, dtype=index_dtype, buffer=shm_index.buf)
    return pd.DataFrame(np_array, index=np_array_index, )


class MarketInfoSharedMemory(MarketInfo):

    names = ['close', 'open', 'high', 'low', 'volume']

    def __init__(self, market:MarketInfo, adj=True, start_time=None, end_time=None):

        self.df_args = []
        self.cache = {}
        self.localize = None

        shape_check = None

        # fetch all dataframe and set localize to None (because localize is not supported in shared memory)
        dfs = {}
        for n in self.names:
            df = market.get_price(n, adj=adj).loc[start_time:end_time].dropna(axis=1, how='all')

            # get localize
            if self.localize is None:
                self.localize = df.index.tzinfo
            else:
                if self.localize != df.index.tzinfo:
                    raise Exception("The timezone of the data is not the same. "
                                    "Please check if the timezone of the data is correct.")
                
            self.cache[n] = df.copy()
                
            df.index = df.index.tz_localize(None)
            dfs[n] = df

        # intersection of columns of dataframe
        columns = dfs['close'].columns.copy()
        for name in dfs:
            columns = columns.intersection(dfs[name].columns)

        # reshape dataframe
        for name, df in dfs.items():
            dfs[name] = df[columns]

        for n, df in dfs.items():
            shape = df.shape
            args = create_shared_memory_from_df(df, n)
            self.df_args.append(args)

            # check shape
            if shape_check is None:
                shape_check = shape
            else:
                if shape_check != shape:
                    raise Exception("Shape of open, high, low, close, volume are not the same. "
                                    "This is because the data is not updated yet. "
                                    "Please try again few minutes later.")

    def to_args(self):
        return self.df_args + [self.localize]
    
    @classmethod
    def from_args(cls, args):
        self = cls.__new__(cls)
        self.df_args = args[:-1]
        self.localize = args[-1]
        self.cache = {}
        return self

    def get_price(self, name, adj=True):

        if name in self.cache:
            return self.cache[name]
        
        args = self.df_args[self.names.index(name)]
        df = create_df_from_shared_memory(*args)

        df.index = df.index.tz_localize(self.localize)
        self.cache[name] = df
        return df

    def close(self):
        for args in self.df_args:
            args[0].close()
            args[0].unlink()
            args[3].close()
            args[3].unlink()
            args[6].close()
            args[6].unlink()
