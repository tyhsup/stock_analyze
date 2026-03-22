import re
import gc
import sys
import copy
import random
import logging
import traceback
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import List, Protocol, Dict, Optional, Generator

import talib
from talib import abstract

from finlab import ml
from finlab import data
import finlab.market_info
from finlab.ml.utils import resampler
from finlab.dataframe import FinlabDataFrame


class IndicatorName():
  
    @staticmethod
    def encode(package_name, func, output, params):

        encoded = package_name + '.' + func + '__' + output + '__'
        for k, v in params.items():
          encoded += f'{k}__{v}__'

        return encoded
    
    @staticmethod
    def decode(encoded):
      
        tokens = encoded.split('__')
        
        func = tokens[0].split('.')[-1]
        output = tokens[1]
        params = dict(zip(tokens[2:][::2], tokens[2:][1::2]))

        return func, output, params


class TalibIndicatorFactory():

    @staticmethod
    def set_dynamic_types(f, params):

        ret = {}
        for k, v in params.items():
            try:
                f.set_parameters(**{k:v})

            except Exception as ex:
                s = str(ex)
                regex = re.compile(r'expected\s([a-z]*),\s')
                match = regex.search(s)
                correct_type = match.group(1)
                v = {'int':int, 'float': float}[correct_type](v)
                f.set_parameters(**{k:v})

            ret[k] = v

        return ret
  
    def calculate_indicator(self, func, output, params, adj=False, market=None):

        if market is None:
            print("market is None. set to default")
            market = ml.get_market()
      
        func = func.split('.')[0]

        # get ith output
        f = getattr(abstract, func)
        org_params = copy.copy(f.parameters)
        org_params = self.set_dynamic_types(f, org_params)

        params = self.set_dynamic_types(f, params)
        f.set_parameters(org_params)
        target_i = -1
        for i, o in enumerate(f.output_names):
            if o == output:
                target_i = i
                break

        if target_i == -1:
            raise Exception("Cannot find output names")
        
        # apply talib
        indicator = data.indicator(func, adj=adj, market=market, **params)
        f.set_parameters(org_params)

        if isinstance(indicator, tuple):
            indicator = indicator[target_i]

        # normalize result
        if func in TalibIndicatorFactory.normalized_funcs():
            indicator /= market.get_price('close', adj=adj)

        return indicator
    
    @staticmethod
    def all_functions():
        talib_categories = [
          'Cycle Indicators', 
          'Momentum Indicators', 
          'Overlap Studies', 
          'Price Transform', 
          'Statistic Functions', 
          'Volatility Indicators']

        talib_functions = sum([talib.get_function_groups()[c] for c in talib_categories], [])
        talib_functions = ['talib.'+f for f in talib_functions if f != 'SAREXT' and f != 'MAVP']
        return talib_functions

    @staticmethod
    @lru_cache
    def normalized_funcs():
      talib_normalized = talib.get_function_groups()['Overlap Studies']\
        + talib.get_function_groups()['Price Transform']\
        + ['APO', 'MACD', 'MACDEXT', 'MACDFIX', 'MOM', 'MINUS_DM', 'PLUS_DM', 'HT_PHASOR']
      return [t for t in talib_normalized]
    
    def generate_feature_names(self, func, lb, ub, n):

        func = func.split('.')[-1]

        if func == 'MAMA':
            return []

        f = getattr(abstract, func)
        outputs = f.output_names
        org_params = f.parameters
        params_lb = {k:v*lb for k, v in org_params.items()}
        params_ub = {k:v*ub for k, v in org_params.items()}
        
        min_value = {
          'signalperiod': 2,
          'timeperiod': 2,
          'fastperiod': 2,
          'slowperiod': 2,
          'timeperiod1': 2, 'timeperiod2': 2,
          'timeperiod3': 2,
          'fastk_period': 2, 
          'fastd_period': 2,
          'slowk_period': 2,
          'slowd_period': 2,
          'vfactor': 0,
        }

        ret = []
        for _ in range(n):

          new_params = {}
          for k, v in org_params.items():
            rvalue = np.random.random_sample(1)[0] * (params_ub[k] - params_lb[k]) + params_lb[k]
            rvalue = type(v)(rvalue)
            new_params[k] = rvalue
            
          
          if 'nbdevup' in new_params:
            new_params['nbdevup'] = 2.0
          if 'nbdevdn' in new_params:
            new_params['nbdevdn'] = 2.0
          if 'vfactor' in new_params:
            new_params['vfactor'] = float(random.uniform(0, 1))
          if 'nbdev' in new_params:
            new_params['nbdev'] = 2.5
            
          for p in new_params:
            if p in min_value and new_params[p] < min_value[p]:
              new_params[p] = min_value[p]
            
          for o in outputs:
            ret.append(IndicatorName.encode('talib', func, o, new_params))

        return list(set(ret))

class Factory(Protocol):
    def __init__(self, market:Optional[finlab.market_info.MarketInfo]) -> None:
        pass

    def all_functions(self) -> List[str]:
        return []

    def calculate_indicator(self, func, output, params) -> pd.DataFrame:
        return pd.DataFrame()

 
def ta_names(lb:int=1, ub:int=10, n:int=1, factory=None) -> List[str]:
    """
    Generate a list of technical indicator feature names.

    Args:
        lb (int): The lower bound of the multiplier of the default parameter for the technical indicators.
        ub (int): The upper bound of the multiplier of the default parameter for the technical indicators.
        n (int): The number of random samples for each technical indicator.
        factory (IndicatorFactory): A factory object to generate technical indicators.
            Defaults to TalibIndicatorFactory.

    Returns:
        List[str]: A list of technical indicator feature names.

    Examples:
        ```py
        import finlab.ml.feature as f


        # method 1: generate each indicator with random parameters
        features = f.ta()

        # method 2: generate specific indicator
        feature_names = ['talib.MACD__macdhist__fastperiod__52__slowperiod__212__signalperiod__75__']
        features = f.ta(feature_names, resample='W')

        # method 3: generate some indicator
        feature_names = f.ta_names()
        features = f.ta(feature_names)
        ```
    """

    if factory is None:
        factory = TalibIndicatorFactory()

    return sum([factory.generate_feature_names(f, lb, ub, n) for f in factory.all_functions()], [])


def create_feature(params):

    name, factories, resample, end_time, adj, kwargs = params[:6]
    market = finlab.market_info.MarketInfoSharedMemory.from_args(*params[6:])
    func, output, params = IndicatorName.decode(name)

    factory = factories[name.split('.')[0]]
    try:
        f = resampler(factory.calculate_indicator(func, output, params, adj=adj, market=market), resample, **kwargs).T.unstack()
        return name, np.array(f.values)

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        logging.warn(f"Cannot calculate indicator {(func, output, params)}. Skipped")
        logging.warn(f"Exception occurred: {traceback.format_exc()}")
    return None


def ta(feature_names:Optional[List[str]], 
       factories=None,
       resample=None, 
       start_time=None, 
       end_time=None, 
       adj=False,
       cpu=-1,
       **kwargs) -> pd.DataFrame:
    """Calculate technical indicator values for a list of feature names.

    Args:
        feature_names (Optional[List[str]]): A list of technical indicator feature names. Defaults to None.
        factories (Optioanl[Dict[str, TalibIndicatorFactory]]): A dictionary of factories to generate technical indicators. Defaults to {"talib": TalibIndicatorFactory()}.
        resample (Optional[str]): The frequency to resample the data to. Defaults to None.
        start_time (Optional[str]): The start time of the data. Defaults to None.
        end_time (Optional[str]): The end time of the data. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the resampler function.

    Returns:
        pd.DataFrame: technical indicator feature names and their corresponding values.
    """

    if factories is None:
        factories = {'talib':TalibIndicatorFactory()}

    if feature_names is None:
        feature_names = ta_names()

    if cpu == -1:
        import multiprocessing
        cpu = multiprocessing.cpu_count()

    if cpu == 1:
        market = ml.get_market()
    else:
        market = finlab.market_info.MarketInfoSharedMemory(ml.get_market(), adj=adj, start_time=start_time, end_time=end_time)

    test_f = resampler(TalibIndicatorFactory().calculate_indicator("RSI", 'real', {}, adj=adj, market=market), 
                       resample, **kwargs).T.unstack()

    final_columns = []

    def create_features() -> Generator[np.ndarray, None, None]:

        nonlocal final_columns

        if cpu == 1:
            for name in feature_names:

                # parallel processing wrapper function
                # name, values = create_feature((name, factories, resample, end_time, adj, kwargs) + tuple([market.to_args()]))

                # single processing
                func, output, params = IndicatorName.decode(name)

                factory = factories[name.split('.')[0]]
                values = resampler(factory.calculate_indicator(func, output, params, adj=adj, market=market), resample, **kwargs).T.unstack()
                if values is not None:
                    final_columns.append(name)
                    yield values
        else:
            import multiprocessing
            with multiprocessing.Pool(processes=cpu) as pool:

                for result in pool.imap_unordered(
                                            create_feature, 
                                            [(name, factories, resample, end_time, adj, kwargs) + tuple([market.to_args()])
                                            for name in feature_names]):
                    if result is not None:
                        name, values = result
                        final_columns.append(name)
                        yield values


    values = np.fromiter(
            create_features(), 
            dtype=np.dtype((np.float64, len(test_f))))
    
    if cpu != 1:
        market.close()

    final_names = set(final_columns)
    ordered_names = [n for n in feature_names if n in final_names]

    ret = pd.DataFrame(values.T, index=test_f.index, 
                       columns=final_columns, copy=False)
    
    ret.index.names = ['datetime', 'instrument']
    return ret[ordered_names]



def combine(features:Dict[str, pd.DataFrame], resample=None, sample_filter=None, **kwargs):

    """The combine function takes a dictionary of features as input and combines them into a single pandas DataFrame. combine 函數接受一個特徵字典作為輸入，並將它們合併成一個 pandas DataFrame。

    Args:
        features (Dict[str, pd.DataFrame]): a dictionary of features where index is datetime and column is instrument. 一個特徵字典，其中索引為日期時間，欄位為證券代碼。
        resample (str): Optional argument to resample the data in the features. Default is None. 選擇性的參數，用於重新取樣特徵中的資料。預設為 None。
        sample_filter (pd.DataFrame): a boolean dictionary where index is date and columns are instrument representing the filter of features.
        **kwargs: Additional keyword arguments to pass to the resampler function. 傳遞給重新取樣函數 resampler 的其他關鍵字引數。

    Returns:
        A pandas DataFrame containing all the input features combined. 一個包含所有輸入特徵合併後的 pandas DataFrame。

    Examples:
        這段程式碼教我們如何使用finlab.ml.feature和finlab.data模組，來合併兩個特徵：RSI和股價淨值比。我們使用f.combine函數來進行合併，其中特徵的名稱是字典的鍵，對應的資料是值。
        我們從data.indicator('RSI')取得'rsi'特徵，這個函數計算相對強弱指數。我們從data.get('price_earning_ratio:股價淨值比')取得'pb'特徵，這個函數獲取股價淨值比。最後，我們得到一個包含這兩個特徵的DataFrame。

        ``` py
        from finlab import data
        import finlab.ml.feature as f
        import finlab.ml.qlib as q

        features = f.combine({
            
            # 用 data.get 簡單產生出技術指標
            'pb': data.get('price_earning_ratio:股價淨值比'),

            # 用 data.indicator 產生技術指標的特徵
            'rsi': data.indicator('RSI'),

            # 用 f.ta 枚舉超多種 talib 指標
            'talib': f.ta(f.ta_names()),

            # 利用 qlib alph158 產生技術指標的特徵(請先執行 q.init(), q.dump() 才能使用)
            'qlib158': q.alpha('Alpha158')

            })

        features.head()
        ```

        |    datetime   | instrument |     rsi    |     pb     |
        |---------------|------------|------------|------------|
        |   2020-01-01  |    1101    |     0      |     2      |
        |   2020-01-02  |    1102    |     100    |     3      |
        |   2020-01-03  |    1108    |     100    |     4      |

    """

    if len(features) == 0:
        return pd.DataFrame()

    def resampling(df) -> pd.DataFrame:
        return resampler(df, resample, **kwargs)
    
    unstacked = {}

    union_index = None
    union_columns = None
    unstacked = {}
    concats = []

    for name, df in features.items():

        if isinstance(df.index, pd.MultiIndex):
            concats.append(df)
        else:
            if isinstance(df, FinlabDataFrame):
                df = df.index_str_to_date()

            udf = resampling(df)
            unstacked[name] = udf
            if union_index is not None:
                union_index = union_index.union(udf.index)
            else:
                union_index = udf.index
            if union_columns is not None:
                union_columns = union_columns.intersection(udf.columns)
            else:
                union_columns = udf.columns
            
    final_index = None
    for name, udf in unstacked.items():
        udf = udf\
            .reindex(index=union_index, columns=union_columns)\
            .ffill()\
            .T\
            .unstack()
        unstacked[name] = udf.values

        if final_index is None:
            final_index = udf.index

    for i, c in enumerate(concats):
        c.index = c.index.set_names(['datetime', 'instrument'])
        if union_index is not None:
            concats[i] = c[c.index.get_level_values('datetime').isin(union_index)]

    if unstacked:
        unstack_df = pd.DataFrame(unstacked, index=final_index)
        # unstack_df = unstack_df.swaplevel(0, 1)
        unstack_df.index = unstack_df.index.set_names(['datetime', 'instrument'])
        concats.append(unstack_df)

    ret = pd.concat(concats, axis=1)
    ret.sort_index(inplace=True)

    if sample_filter is not None:
        if isinstance(sample_filter, FinlabDataFrame):
            sample_filter = sample_filter.index_str_to_date()
        usf = resampling(sample_filter)

        if union_index is not None and union_columns is not None:
            usf = usf.reindex(index=union_index, columns=union_columns)

        usf = usf.ffill()\
           .T\
           .unstack()\
           .reindex(ret.index).fillna(False)
        ret = ret[usf.values]

    return ret
