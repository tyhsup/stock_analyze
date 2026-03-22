import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from finlab.dataframe import FinlabDataFrame

def event_study(factor_data, benchmark_adj_close, stock_adj_close, sample_period=(-45, -20), estimation_period=(-5, 20), plot=True):
    '''Run event study and returns the abnormal returns of each stock on each day.

    Args:
        factor_data (pd.DataFrame): factor data where index is datetime and columns is asset id
        benchmark_adj_close (pd.DataFrame): benchmark for CAPM
        stock_adj_close (pd.DataFrame): stock price for CAPM
        sample_period ((int, int)): period for fitting CAPM
        estimation_period ((int, int)): period for calculating alpha (abnormal return)
        plot (bool): plot the result
        
    Return:
        Abnormal returns of each stock on each day.

    Examples:
        ``` py title="現金增減資分析"
        from finlab.tools.event_study import create_factor_data
        from finlab.tools.event_study import event_study

        factor = data.get('price_earning_ratio:股價淨值比')
        adj_close = data.get('etl:adj_close')
        benchmark = data.get('benchmark_return:發行量加權股價報酬指數')

        # create event dataframe
        dividend_info = data.get('dividend_announcement')
        v = dividend_info[['stock_id', '除權交易日']].set_index(['stock_id', '除權交易日'])
        v['value'] = 1
        event = v[~v.index.duplicated()].reset_index().drop_duplicates(
            subset=['stock_id', '除權交易日']
        ).pivot(index='除權交易日', columns='stock_id', values='value').notna()

        # calculate factor_data
        factor_data = create_factor_data({'pb':factor}, adj_close, event=event)

        r = event_study(factor_data, benchmark, adj_close)

        plt.bar(r.columns, r.mean().values)
        plt.plot(r.columns, r.mean().cumsum().values)
        ```
    '''
    benchmark_pct = benchmark_adj_close.reindex(stock_adj_close.index, method='ffill').pct_change()
    stock_pct = stock_adj_close.pct_change()

    def get_period(df, date, sample):
        i = df.index.get_loc(date)
        return df.iloc[i+sample[0]: i+sample[1]].values

    ret = []

    for date, sid in tqdm.tqdm(factor_data.index):

        X1, Y1 = get_period(benchmark_pct, date, sample_period)[:,0], \
            get_period(stock_pct[sid], date, sample_period)
        X2, Y2 = get_period(benchmark_pct, date, estimation_period)[:,0], \
            get_period(stock_pct[sid], date, estimation_period)

        # Run CAPM
        cov_matrix = np.cov(Y1, X1)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        AR = np.array(Y2) - beta * X2
        ret.append(AR)

    ret = pd.DataFrame(ret, columns=range(*estimation_period))

    if plot:
        plot_event_study(ret)
    
    return ret


def create_factor_data(factor:pd.DataFrame or Dict[str, pd.DataFrame], adj_close:pd.DataFrame,
                       days:List[int]=[5,10,20, 60], event:pd.DataFrame or None=None):

    '''create factor data, which contains future return

    Args:
        factor (pd.DataFrame): factor data where index is datetime and columns is asset id
        adj_close (pd.DataFrame): adj close where index is datetime and columns is asset id
        days (List[int]): future return considered

    Return:
        Analytic plots and tables

    Warning:
        This function is not identical to `finlab.ml.alphalens.create_factor_data`

    Examples:
        ``` py title="現金增減資分析"
        from finlab.tools.event_study import create_factor_data
        from finlab.tools.event_study import event_study

        factor = data.get('price_earning_ratio:股價淨值比')
        adj_close = data.get('etl:adj_close')
        benchmark = data.get('benchmark_return:發行量加權股價報酬指數')

        # create event dataframe
        dividend_info = data.get('dividend_announcement')
        v = dividend_info[['stock_id', '除權交易日']].set_index(['stock_id', '除權交易日'])
        v['value'] = 1
        event = v[~v.index.duplicated()].reset_index().drop_duplicates(
            subset=['stock_id', '除權交易日']
        ).pivot(index='除權交易日', columns='stock_id', values='value').notna()

        # calculate factor_data
        factor_data = create_factor_data({'pb':factor}, adj_close, event=event)

        r = event_study(factor_data, benchmark, adj_close)

        plt.bar(r.columns, r.mean().values)
        plt.plot(r.columns, r.mean().cumsum().values)
        ```

    '''

    factor = {'factor':factor} if isinstance(factor, pd.DataFrame) else factor

    ref = next(iter(factor.values())) if event is None else event
    ref = ref[~ref.index.isna()]

    sids = adj_close.columns.intersection(ref.columns)
    dates = adj_close.index.intersection(
        FinlabDataFrame.to_business_day(ref.index))
    
    ret = {}
    for name, f in factor.items():
        reashaped_f = f.reindex(dates, method='ffill').reindex(columns=sids)
        ret[f'{name}_factor'] = reashaped_f.unstack().values
        ret[f'{name}_factor_quantile'] = (reashaped_f.rank(axis=1, pct=True) // 0.2).unstack().values

    total_index = None
    for d in days:
        temp = (adj_close.shift(-d-1) / adj_close.shift(-1) - 1)\
            .reindex(index=dates, method='ffill').reindex(columns=sids)\
            .unstack()
        
        ret[f"{d}D"] = temp.values
        total_index = temp.index

    if event is not None:
        event = event[event.index.notna()]
        reshaped_event = event.reindex(index=dates, method='ffill').reindex(columns=sids)
        ret['event'] = reshaped_event.unstack().values


    ret = pd.DataFrame(ret, index=total_index.swaplevel(0, 1))\
        .replace([-np.inf, np.inf], np.nan)\
        .dropna()

    if 'event' in ret:
        ret = ret[ret['event'] == True]
        ret.drop(columns=['event'], inplace=True)

    ret.index.names = ['date', 'asset']
    return ret

def plot_event_study(returns:pd.DataFrame):
    """
    Plot the event study for the given returns.

    Args:
        returns (pd.DataFrame): A DataFrame containing the returns data.

    Return:
        ax (matplotlib.axes.Axes): The axes object containing the plot.

    """
    returns.mul(100).mean().plot.bar(use_index=False, label='Return')

    s = returns.mul(100).cumsum(axis=1).mean()
    std = returns.mul(100).cumsum(axis=1).std() * 0.1

    ax = s.plot(use_index=False, label='Cumulative Return')

    y1 = s + std
    y2 = s - std

    # fill between
    ax.fill_between(range(len(returns.columns)), y1, y2, color='gray', alpha=0.1, label='0.1 std')

    # set labels
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_xlabel('Days')
    ax.set_title('Event Study')

    # set xticks
    ax.set_xticks(range(len(returns.columns)))
    ax.set_xticklabels(returns.columns, rotation=45)
    ax.legend(loc='upper left')
    
    # show ax
    plt.show()
    
    return ax