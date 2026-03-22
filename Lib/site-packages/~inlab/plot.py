import pandas as pd
from finlab import data
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from finlab.utils import logger
import itertools

"""
Candles
"""


def str_to_indicator(s, df):
    from talib import abstract
    import talib

    params = {}
    if '(' in s:
        params = 'dict(' + s.split('(')[-1][:-1] + ')'
        params = eval(params)
    s = s.split('(')[0]

    func = getattr(abstract, s)
    real_func = getattr(talib, s)

    abstract_input = list(func.input_names.values())[0]
    if isinstance(abstract_input, str):
        abstract_input = [abstract_input]

    pos_paras = [df[k] for k in abstract_input]

    ret = real_func(*pos_paras, **params)

    if isinstance(ret, np.ndarray):
        ret = pd.Series(ret, index=df.index)

    if isinstance(ret, pd.Series):
        return ret.to_frame(s)
    return ret


def color_generator():
    for i in itertools.cycle(px.colors.qualitative.Plotly):
        yield i


def average(series, n):
    return series.rolling(n, min_periods=int(n / 2)).mean()


def evaluate_to_df(node, stock_id, df):
    if callable(node):
        node = node(df)

    if isinstance(node, str):
        node = str_to_indicator(node, df)

    if isinstance(node, pd.Series):
        return node.to_frame('0')

    if isinstance(node, np.ndarray):
        return pd.Series(node, df.index).to_frame('0')

    if isinstance(node, pd.DataFrame):
        if stock_id in node.columns:
            return pd.DataFrame({'0': node[stock_id]})
        else:
            return node

    if isinstance(node, list) or isinstance(node, tuple):
        new_node = {}
        ivalue = 0
        for n in node:
            if isinstance(n, str):
                new_node[n] = n
            else:
                new_node[ivalue] = n
                ivalue += 1
        node = new_node

    if isinstance(node, dict):
        dfs = []
        for name, n in node.items():
            nn = evaluate_to_df(n, stock_id, df)
            if len(nn.columns) == 1:
                nn.columns = [name]
            dfs.append(nn)

        return pd.concat(dfs, axis=1)

    assert 0


def format_indicators(indicators, stock_id, stock_df):
    if not isinstance(indicators, list):
        indicators = [indicators]

    ret = [evaluate_to_df(i, stock_id, stock_df) for i in indicators]

    return ret


def plot_candles(stock_id, close, open_, high, low, volume, recent_days=250, resample='D', overlay_func=None,
                 technical_func=None):
    c = color_generator()
    next(c)
    next(c)

    df = (pd.DataFrame({
        'close': close.values,
        'open': open_.values,
        'high': high.values,
        'low': low.values,
        'volume': volume.values}, index=close.index).iloc[-abs(recent_days):]
    )

    if resample:
        df = df.resample(resample).agg({
            'close': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'})

    if overlay_func is None:
        upperband, middleband, lowerband = data.indicator('BBANDS')
        overlay_func = {
            'upperband': upperband,
            'middleband': middleband,
            'lowerband': lowerband,
        }

    if technical_func is None:
        k, d = data.indicator('STOCH')
        technical_func = [{'K': k, 'D': d}]

    overlay_indicator = format_indicators(overlay_func, stock_id, df)

    # merge overlay indicator if it has multiple plots
    if len(overlay_indicator) > 1:
        overlay_indicator = [pd.concat(overlay_indicator, axis=1)]
        overlay_indicator[0].columns = range(len(overlay_indicator[0].columns))

    technical_indicator = format_indicators(technical_func, stock_id, df)

    # truncate recent days
    for i, d in enumerate(overlay_indicator):
        o_ind = d.iloc[-abs(recent_days):]
        if resample != 'D':
            o_ind = o_ind.reindex(df.index, method='ffill')
        overlay_indicator[i] = o_ind
    for i, d in enumerate(technical_indicator):
        t_ind = d.iloc[-abs(recent_days):]
        if resample != 'D':
            t_ind = t_ind.reindex(df.index, method='ffill')
        technical_indicator[i] = t_ind

    technical_func_num = len(technical_indicator)
    index_value = close.index

    nrows = 1 + len(technical_indicator)

    fig_titles = ['']
    if isinstance(technical_func, list):
        for t in technical_func:
            fig_titles.append(','.join(list(t.keys())))
    elif isinstance(technical_func, dict):
        fig_titles.append(','.join(list(technical_func.keys())))

    fig = make_subplots(
        rows=nrows,
        specs=[[{"secondary_y": True}]] * nrows,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=fig_titles,
        row_heights=[0.4] + [0.1] * (nrows - 1))

    fig.add_trace(
        go.Bar(x=df.index, y=df.volume, opacity=0.3, name="volume",
               marker={'color': 'gray', 'line_width': 0}),
        row=1, col=1
    )

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df.open,
                                 high=df.high,
                                 low=df.low,
                                 close=df.close,
                                 increasing_line_color='#ff5084',
                                 decreasing_line_color='#2bbd91',
                                 legendgroup='1',
                                 name='candle',
                                 ), row=1, col=1, secondary_y=True)

    # overlay plot
    if overlay_indicator:
        fig_overlay = px.line(overlay_indicator[0])
        for o in fig_overlay.data:
            fig.add_trace(go.Scatter(x=o['x'], y=o['y'], name=o['name'], line=dict(color=next(c)), legendgroup="1"),
                          row=1, col=1, secondary_y=True)

    for num, tech_ind in enumerate(technical_indicator):
        fig_tech = px.line(tech_ind)
        for t in fig_tech.data:
            color = next(c)

            fig.add_trace(
                go.Scatter(x=t['x'], y=t['y'], name=t['name'], line=dict(color=color),
                           legendgroup=str(2 + num),

                           ),
                row=2 + num, col=1)

    # hide holiday
    if resample == 'D':
        dt_all = pd.date_range(start=index_value[0], end=index_value[-1])
        # retrieve the dates that are in the original dataset
        dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(index_value)]
        # define dates with missing values
        dt_breaks = [d for d in dt_all.strftime(
            "%Y-%m-%d").tolist() if d not in dt_obs]
        # hide dates with no values
        fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

    fig.update_layout(
        height=600 + 100 * technical_func_num,
    )

    fig.update_layout(
        yaxis1=dict(
            title="volume",
            titlefont=dict(
                color="#777"
            ),
            tickfont=dict(
                color="#777"
            ),
            range=[df.volume.min(), df.volume.max() * 2]
        ),
        yaxis2=dict(
            title="price",
            titlefont=dict(
                color="#777"
            ),
            tickfont=dict(
                color="#777"
            ),
            showgrid=False
        ),
        hovermode='x unified',
    )

    fig.update_layout(**{
        'xaxis1_rangeslider_visible': False,
        f'xaxis': dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
        ),
        f'xaxis{nrows}': dict(
            rangeslider=dict(
                visible=True,
                thickness=0.1,
                bgcolor='gainsboro',
            ),
            type="date",
        ),
    })

    # fig.update_traces(xaxis='x2')
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True, spikemode="across")
    fig.update_layout(showlegend=False)
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(showline=True, linecolor='#ddd')
    fig.update_yaxes(showline=True, linecolor='#ddd')
    fig.update_yaxes(titlefont=dict(
        color="#777"
    ),
        tickfont=dict(
            color="#777"
    ))

    fig.update_layout(title={'text': f'Candlestick Plot {stock_id}', 'font': {
                      'size': 18, 'color': 'gray'}})

    return fig


def plot_tw_stock_candles(stock_id, recent_days=400, adjust_price=False, resample='D', overlay_func=None,
                          technical_func=None):
    """繪製台股技術線圖圖組
    Args:
        stock_id (str): 台股股號，ex:`'2330'`。
        recent_days (int):取近n個交易日資料。
        adjust_price (bool):是否使用還原股價計算。
        resample (str): 技術指標價格週期，ex: `D` 代表日線, `W` 代表週線, `M` 代表月線。
        overlay_func (dict):
            K線圖輔助線，預設使用布林通道。
             ```py
             from finlab.data import indicator

             overlay_func={
                          'ema_5':indicator('EMA',timeperiod=5),
                          'ema_10':indicator('EMA',timeperiod=10),
                          'ema_20':indicator('EMA',timeperiod=20),
                          'ema_60':indicator('EMA',timeperiod=60),
                         }
             ```
        technical_func (list):
            技術指標子圖，預設使用KD技術指標單組子圖。

            設定多組技術指標：
            ```py
            from finlab.data import indicator

            k,d = indicator('STOCH')
            rsi = indicator('RSI')
            technical_func = [{'K':k,'D':d},{'RSI':rsi}]
            ```

    Returns:
        (plotly.graph_objects.Figure): 技術線圖

    Examples:
        ```py
        from finlab.plot import plot_tw_stock_candles
        from finlab.data import indicator

        overlay_func={
                      'ema_5':indicator('EMA',timeperiod=5),
                      'ema_10':indicator('EMA',timeperiod=10),
                      'ema_20':indicator('EMA',timeperiod=20),
                      'ema_60':indicator('EMA',timeperiod=60),
                     }
        k,d = indicator('STOCH')
        rsi = indicator('RSI')
        technical_func = [{'K':k,'D':d},{'RSI':rsi}]
        plot_tw_stock_candles(stock_id='2330',recent_days=600,adjust_price=False,overlay_func=overlay_func,technical_func=technical_func)
        ```
    """
    if adjust_price:
        close = data.get('etl:adj_close')[stock_id]
        open_ = data.get('etl:adj_open')[stock_id]
        high = data.get('etl:adj_high')[stock_id]
        low = data.get('etl:adj_low')[stock_id]
    else:
        close = data.get('price:收盤價')[stock_id]
        open_ = data.get('price:開盤價')[stock_id]
        high = data.get('price:最高價')[stock_id]
        low = data.get('price:最低價')[stock_id]

    volume = data.get('price:成交股數')[stock_id]

    return plot_candles(stock_id, close, open_, high, low, volume, recent_days=recent_days, resample=resample,
                        overlay_func=overlay_func, technical_func=technical_func)


"""
Treemap
"""


def df_date_filter(df, start=None, end=None):
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def create_treemap_data(start, end, item='return_ratio', clip=None):
    """產生台股板塊圖資料

    產生繪製樹狀圖所用的資料，可再外加FinLab資料庫以外的指標製作客製化DataFrame，
    並傳入`plot_tw_stock_treemap(treemap_data=treemap_data)。`

    Args:
      start (str): 資料開始日，ex:`"2021-01-02"`。
      end (str):資料結束日，ex:`"2021-01-05"`。
      item (str): 決定板塊顏色深淺的指標。
                  除了可選擇依照 start 與 end 計算的`"return_ratio"`(報酬率)，
                  亦可選擇[FinLab資料庫](https://ai.finlab.tw/database)內的指標顯示近一期資料。
          example:

          * `'price_earning_ratio:本益比'` - 顯示近日產業的本益比高低。
          * `'monthly_revenue:去年同月增減(%)'` - 顯示近月的單月營收年增率。

      clip (tuple): 將item邊界外的值分配給邊界值，防止資料上限值過大或過小，造成顏色深淺變化不明顯。
                    ex:(0,100)，將數值低高界線，設為0~100，超過的數值。
        !!! note

            參考[pandas文件](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.clip.html)更了解`pd.clip`細節。

    Returns:
        (pd.DataFrame): 台股個股指標
    Examples:

        欲下載所有上市上櫃之價量歷史資料與產業分類，只需執行此函式:

        ``` py
        from finlab.plot import create_treemap_data
        create_treemap_data(start= '2021-07-01',end = '2021-07-02')
        ```

        | stock_id   |  close |turnover|category|market|market_value|return_ratio|country|
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | 1101       |   20 |  57.85 |  水泥工業 |  sii   |    111  |    0.1  |  TW-Stock|
        | 1102       |  20 |  58.1  |  水泥工業 |  sii    |    111  |    -0.1 |  TW-Stock|


    """
    close = data.get('price:收盤價')
    basic_info = data.get('company_basic_info')
    turnover = data.get('price:成交金額')
    close_data = df_date_filter(close, start, end)
    turnover_data = df_date_filter(
        turnover, start, end).iloc[1:].sum() / 100000000
    return_ratio = (
        close_data.loc[end] / close_data.loc[start]).dropna().replace(np.inf, 0)
    return_ratio = round((return_ratio - 1) * 100, 2)

    concat_list = [close_data.iloc[-1], turnover_data, return_ratio]
    col_names = ['stock_id', 'close', 'turnover', 'return_ratio']
    if item not in ["return_ratio", "turnover_ratio"]:
        try:
            custom_item = df_date_filter(
                data.get(item), start, end).iloc[-1].fillna(0)
        except Exception as e:
            logger.error(
                'data error, check the data is existed between start and end.')
            logger.error(e)
            return None
        if clip:
            custom_item = custom_item.clip(*clip)
        concat_list.append(custom_item)
        col_names.append(item)

    df = pd.concat(concat_list, axis=1).dropna()
    df = df.reset_index()
    df.columns = col_names

    basic_info_df = basic_info.copy().reset_index()
    basic_info_df['stock_id_name'] = basic_info_df['stock_id'].astype(
        str) + basic_info_df['公司簡稱']

    df = df.merge(basic_info_df[['stock_id', 'stock_id_name', '產業類別', '市場別', '實收資本額(元)']], how='left',
                  on='stock_id')
    df = df.rename(columns={'產業類別': 'category',
                   '市場別': 'market', '實收資本額(元)': 'base'})
    df = df.dropna(thresh=5)
    df['market_value'] = round(df['base'] / 10 * df['close'] / 100000000, 2)
    df['country'] = 'TW-Stock'
    return df


def plot_tw_stock_treemap(start=None, end=None, area_ind='market_value', item='return_ratio', clip=None,
                          color_continuous_scale='Temps', treemap_data=None):
    """繪製台股板塊圖資料

    巢狀樹狀圖可以顯示多維度資料，將依照產業分類的台股資料絢麗顯示。

    Args:
      start (str): 資料開始日，ex:`'2021-01-02'`。
      end (str): 資料結束日，ex:`'2021-01-05'`。
      area_ind (str): 決定板塊面積數值的指標。
                      可選擇`["market_value","turnover"]`，數值代表含義分別為市值、成交金額。
      item (str): 決定板塊顏色深淺的指標。
                  除了可選擇依照 start 與 end 計算的`"return_ratio"`(報酬率)，
                  亦可選擇[FinLab資料庫](https://ai.finlab.tw/database)內的指標顯示近一期資料。
          example:

          * `'price_earning_ratio:本益比'` - 顯示近日產業的本益比高低。
          * `'monthly_revenue:去年同月增減(%)'` - 顯示近月的單月營收年增率。

      clip (tuple): 將 item 邊界外的值分配給邊界值，防止資料上限值過大或過小，造成顏色深淺變化不明顯。
                    ex:(0,100)，將數值低高界線，設為 0~100，超過的數值。
        !!!note

            參考[pandas文件](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.clip.html)更了解`pd.clip`細節。
      color_continuous_scale (str):[顏色變化序列的樣式名稱](https://plotly.com/python/builtin-colorscales/)
      treemap_data (pd.DataFrame): 客製化資料，格式參照 `create_treemap_data()` 返回值。
    Returns:
        (plotly.graph_objects.Figure): 樹狀板塊圖
    Examples:
        ex1:
        板塊面積顯示成交金額，顏色顯示'2021-07-01'～'2021-07-02'的報酬率變化，可以觀察市場資金集中的產業與漲跌強弱。
        ```py
        from finlab.plot import plot_tw_stock_treemap
        plot_tw_stock_treemap(start= '2021-07-01',end = '2021-07-02',area_ind="turnover",item="return_ratio")
        ```
        ![成交佔比/報酬率板塊圖](img/plot/treemap_return.png)
        ex2:
        板塊面積顯示市值(股本*收盤價)，顏色顯示近期本益比，可以觀察全市場哪些是權值股？哪些產業本益比評價高？限制數值範圍在(0,50)，
        將過高本益比的數值壓在50，不讓顏色變化突兀，能分出高低階層即可。
        ```py
        from finlab.plot import plot_tw_stock_treemap
        plot_tw_stock_treemap(area_ind="market_value",item="price_earning_ratio:本益比",clip=(0,50), color_continuous_scale='RdBu_r')
        ```
        ![市值/本益比板塊圖](img/plot/treemap_pe.png)
    """
    if treemap_data is None:
        df = create_treemap_data(start, end, item, clip)
    else:
        df = treemap_data.copy()

    if df is None:
        return None
    df['custom_item_label'] = round(df[item], 2).astype(str)
    df.dropna(how='any', inplace=True)

    if area_ind not in df.columns:
        return None

    if item in ['return_ratio']:
        color_continuous_midpoint = 0
    else:
        color_continuous_midpoint = np.average(df[item], weights=df[area_ind])

    fig = px.treemap(df,
                     path=['country', 'market', 'category', 'stock_id_name'],
                     values=area_ind,
                     color=item,
                     color_continuous_scale=color_continuous_scale,
                     color_continuous_midpoint=color_continuous_midpoint,
                     custom_data=['custom_item_label', 'close', 'turnover'],
                     title=f'TW-Stock Market TreeMap({start}~{end})'
                           f'---area_ind:{area_ind}---item:{item}',
                     width=1600,
                     height=800)

    fig.update_traces(textposition='middle center',
                      textfont_size=24,
                      texttemplate="%{label}<br>(%{customdata[1]})<br>%{customdata[0]}",
                      )
    return fig


"""
Radar
"""


def plot_radar(df, mode='bar_polar', line_polar_fill=None, title=None, polar_range=10):
    args = dict(data_frame=df, r="value", theta="variable", color="stock_id", line_close=True,
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                template="plotly_dark")
    if mode != 'line_polar':
        args.pop('line_close')

    fig = getattr(px, mode)(**args)
    if title is None:
        title = 'Features Radar'
    fig.update_layout(
        title={
            'text': title,
            'x': 0.49,
            'y': 0.99,
            'xanchor': 'center',
            'yanchor': 'top'},
        paper_bgcolor='rgb(41, 30, 109)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, polar_range]
            )),
        width=1200,
        height=600)
    if mode == 'line_polar':
        # None,toself,tonext
        fig.update_traces(fill=line_polar_fill)
    return fig


def get_rank(item: str, period=None, cut_bins=10):
    df = data.get(item)
    if period == None:
        df = df.iloc[-1]
    else:
        df = df.loc[period]

    df_rank = df.rank(pct=True)
    df_rank = pd.cut(x=df_rank, bins=cut_bins, labels=[
                     i for i in range(1, cut_bins + 1)])
    return df_rank


def get_rank_df(feats: list, period=None, cut_bins=10):
    df = pd.concat([get_rank(f, period, cut_bins) for f in feats], axis=1)
    columns_name = [f[f.index(':') + 1:] for f in feats]
    df.columns = columns_name
    df.index.name = 'stock_id'
    return df


def plot_tw_stock_radar(portfolio, feats=None, mode='line_polar', line_polar_fill=None,
                        period=None, cut_bins=10, title=None, custom_data=None):
    """繪製台股雷達圖

    比較持股組合的指標分級特性。若數值為nan，則不顯示分級。

    Args:
      portfolio (list):持股組合，ex:`['1101','1102']`。
      feats (list): 選定FinLab資料庫內的指標組成資料集。預設為18項財務指標。
                    ex:['fundamental_features:營業毛利率','fundamental_features:營業利益率']
      mode (str): 雷達圖模式 ，ex:'bar_polar','scatter_polar','line_polar'`。
        !!!note

            參考[不同模式的差異](https://plotly.com/python-api-reference/generated/plotly.express.html)
      line_polar_fill (str):將區域設置為用純色填充 。ex:`None,'toself','tonext'`
                           `'toself'`將跡線的端點（或跡線的每一段，如果它有間隙）連接成一個封閉的形狀。
                           如果一條完全包圍另一條（例如連續的等高線），則`'tonext'`填充兩條跡線之間的空間，如果之前沒有跡線，
                           則其行為類似於`'toself'`。如果一條跡線不包含另一條跡線，則不應使用`'tonext'`。
        欲使用 line_polar，請將pandas版本降至 1.4.4。
        !!!note

            參考[plotly.graph_objects.Scatterpolar.fill](https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatterpolar.html)

      period (str): 選擇第幾期的特徵資料，預設為近一季。
                    ex: 設定數值為'2020-Q2，取得2020年第二季資料比較。
      cut_bins (int):特徵分級級距。
      title (str):圖片標題名稱。
      custom_data (pd.DataFrame): 客製化指標分級，欄名為特徵
                    格式範例:

        | stock_id   |  營業毛利率 |營業利益率|稅後淨利率|
        |:-----------|-------:|-------:|-------:|
        | 1101       |   2    |    5   |      3|
        | 1102       |   1    |    8   |      4|
    Returns:
        (plotly.graph_objects.Figure): 雷達圖
    Examples:
        ex1:比較持股組合累計分數，看持股組合偏重哪像特徵。
        ```py
        from finlab.plot import plot_tw_stock_radar
        plot_tw_stock_radar(portfolio=["1101", "2330", "8942", "6263"], mode="bar_polar", line_polar_fill='None')
        ```
        ![持股組合雷達圖](img/plot/radar_many.png)
        ex2:看單一個股特徵分級落點。
        ```py
        from finlab.plot import plot_tw_stock_radar
        feats = ['fundamental_features:營業毛利率', 'fundamental_features:營業利益率', 'fundamental_features:稅後淨利率',
                 'fundamental_features:現金流量比率', 'fundamental_features:負債比率']
        plot_tw_stock_radar(portfolio=["9939"], feats=feats, mode="line_polar", line_polar_fill='toself', cut_bins=8)
        ```
        ![單檔標的子選指標雷達圖](img/plot/radar_single.png)
    """
    if custom_data is None:
        if feats is None:
            feats = ['fundamental_features:營業毛利率', 'fundamental_features:營業利益率',
                     'fundamental_features:稅後淨利率',
                     'fundamental_features:ROA綜合損益', 'fundamental_features:ROE綜合損益',
                     'fundamental_features:業外收支營收率',
                     'fundamental_features:現金流量比率', 'fundamental_features:負債比率',
                     'fundamental_features:流動比率', 'fundamental_features:速動比率',
                     'fundamental_features:存貨週轉率',
                     'fundamental_features:營收成長率', 'fundamental_features:營業毛利成長率',
                     'fundamental_features:營業利益成長率', 'fundamental_features:稅前淨利成長率',
                     'fundamental_features:稅後淨利成長率',
                     'fundamental_features:資產總額成長率', 'fundamental_features:淨值成長率'
                     ]
        df = get_rank_df(feats, period=period, cut_bins=cut_bins)
    else:
        df = custom_data.copy()

    col_name = df.columns
    portfolio = df.index.intersection(portfolio)
    if len(portfolio) < 1:
        logger.error('data is not existed.')
        return
    df = df.loc[portfolio]
    df = df.reset_index()
    df = pd.melt(df, id_vars=['stock_id'], value_vars=col_name)
    polar_range = cut_bins * len(portfolio)
    fig = plot_radar(df=df, mode=mode, line_polar_fill=line_polar_fill,
                     title=title, polar_range=polar_range)
    return fig


"""
PE PB River
"""


def get_pe_river_data(start=None, end=None, stock_id='2330', mode='pe', split_range=6):
    if mode not in ['pe', 'pb']:
        logger.error('mode error')
        return None
    close = df_date_filter(data.get('price:收盤價'), start, end)
    pe = df_date_filter(data.get('price_earning_ratio:本益比'), start, end)
    pb = df_date_filter(data.get('price_earning_ratio:股價淨值比'), start, end)
    df = eval(mode)
    if stock_id not in df.columns:
        logger.error('stock_id input is not in data.')
        return None
    df = df[stock_id]
    max_value = df.max()
    min_value = df.min()
    quan_value = (max_value - min_value) / split_range
    river_borders = [round(min_value + quan_value * i, 2)
                     for i in range(0, split_range + 1)]
    result = (close[stock_id] / df).dropna().to_frame()
    index_name = f'{mode}/close'
    result.columns = [index_name]
    result['close'] = close[stock_id]
    result['pe'] = pe[stock_id]
    result['pb'] = pb[stock_id]
    for r in river_borders:
        col_name = f"{r} {mode}"
        result[col_name] = result[index_name] * r
    result = round(result, 2)
    return result


def plot_tw_stock_river(stock_id='2330', start=None, end=None, mode='pe', split_range=8):
    """繪製台股河流圖

    使用 PE or PB 的最高與最低值繪製河流圖，判斷指標所處位階。

    Args:
      stock_id (str): 台股股號，ex:`'2330'`。
      start (str): 資料開始日，ex:`'2020-01-02'`。
      end (str): 資料結束日，ex:`'2022-01-05'`。
      mode (str): `'pe'` or `'pb'` (本益比或股價淨值比)。
      split_range (int): 河流階層數。
    Returns:
        (plotly.graph_objects.Figure): 河流圖
    Examples:
      ```py
      from finlab.plot import plot_tw_stock_river
      plot_tw_stock_river(stock_id='2330', start='2015-1-1', end='2022-7-1', mode='pe', split_range=10)
      ```
      ![單檔標的子選指標雷達圖](img/plot/pe_river.png)
    """
    df = get_pe_river_data(start, end, stock_id, mode, split_range)
    if df is None:
        logger.error('data error')
        return None
    col_name_set = [i for i in df.columns if any(map(str.isdigit, i))]

    fig = go.Figure()
    for n, c in enumerate(col_name_set):
        if n == 0:
            fill_mode = None
        else:
            fill_mode = 'tonexty'
        fig.add_trace(
            go.Scatter(x=df.index, y=df[c], fill=fill_mode, line=dict(width=0, color=px.colors.qualitative.Prism[n]),
                       name=c))
    customdata = [(c, p) for c, p in zip(df['close'], df[mode])]
    hovertemplate = "<br>date:%{x|%Y/%m/%d}<br>close:%{customdata[0]}" + \
        f"<br>{mode}" + ":%{customdata[1]}"
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], line=dict(width=2.5, color='#2e4391'), customdata=customdata,
                             hovertemplate=hovertemplate, name='close'))

    security_categories = data.get('security_categories').set_index('stock_id')
    stock_name = security_categories.loc[stock_id]['name']
    fig.update_layout(title=f"{stock_id} {stock_name} {mode.upper()} River Chart",
                      template="ggplot2",
                      yaxis=dict(
                          title='price',
                      ),
                      # hovermode='x unified',
                      )
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    return fig


class StrategySunburst:
    def __init__(self):
        """繪製策略部位旭日圖

        監控多策略。
        """
        d = data.get_strategies()
        for k, v in d.items():
            if 'position' in d[k]['positions']:
                d[k]['positions'] = d[k]['positions']['position']
        self.s_data = d

    def process_position(self, s_name, s_weight=1):
        if s_name == '現金':
            result = pd.DataFrame(
                {'return': 0, 'weight': 1, 'category': '現金', 'market': '現金'}, index=['現金'])
            result.index.name = 'stock_id'
        else:
            df = pd.DataFrame(self.s_data[s_name]['positions'])
            df = df[[c for c in df.columns if ' ' in c]]
            df = df.T
            df['weight'] = df['weight'].apply(
                lambda s: abs(pd.to_numeric(s, errors='coerce')))
            df = df[df['weight'] > 0]
            if len(df) == 0:
                df['weight'] = 0
            df.index.name = 'stock_id'
            old_security_categories = data.get(
                'security_categories').reset_index()
            security_categories = old_security_categories.copy()
            security_categories['category'] = security_categories['category'].fillna(
                'other_securities')
            security_categories['stock_id'] = security_categories['stock_id'] + \
                ' ' + security_categories['name']
            security_categories = security_categories.set_index(['stock_id'])
            result = df.join(security_categories)

            asset_type = self.s_data[s_name]['asset_type']
            if asset_type == '':
                asset_type = 'tw_stock'
            elif asset_type == 'crypto':
                category = 'crypto'
                result['category'] = category

            result['market'] = asset_type
            cash = pd.DataFrame({'return': 0, 'weight': 1 - (df['weight'].sum()), 'category': '現金', 'market': '現金'},
                                index=['現金'])
            cash.index.name = 'stock_id'
            result = pd.concat([result, cash])

        result['s_name'] = s_name
        result['s_weight'] = s_weight
        return result

    def get_strategy_df(self, select_strategy=None):
        """獲取策略部位與分配權重後計算的資料

        Args:
          select_strategy (dict): 選擇策略名稱並設定權重，預設是抓取權策略並平分資金比例到各策略。
                                 ex:`{'低波動本益成長比':0.5,'研發魔人':0.2, '現金':0.2}`
        Returns:
            (pd.DataFrame): strategies data
        """
        if select_strategy is None:
            s_name = self.s_data.keys()
            s_num = len(s_name)
            if s_num == 0:
                return None
            s_weight = [1 / s_num] * len(s_name)
        else:
            s_name = select_strategy.keys()
            s_weight = select_strategy.values()

        all_position = pd.concat([self.process_position(
            name, weight) for name, weight in zip(s_name, s_weight)])
        all_position['weight'] *= all_position['s_weight']
        all_position['return'] = round(all_position['return'].astype(float), 2)
        all_position['color'] = round(
            all_position['return'].clip(all_position['return'].min() / 2, all_position['return'].max() / 2), 2)
        all_position = all_position[all_position['weight'] > 0]
        all_position = all_position.reset_index()
        all_position = all_position[all_position['s_name'] != 'playground']
        all_position['category'] = all_position['category'].fillna(
            'other_securities')
        return all_position

    def plot(self, select_strategy=None, path=None, color_continuous_scale='RdBu_r'):
        """繪圖

        Args:
          select_strategy (dict): 選擇策略名稱並設定權重，預設是抓取權策略並平分資金比例到各策略。
                                 ex:`{'低波動本益成長比':0.5,'研發魔人':0.2, '現金':0.2}`
          path (list): 旭日圖由裡到外的顯示路徑，預設為`['s_name', 'market', 'category', 'stock_id']`。
                       `['market', 'category','stock_id','s_name']`也是常用選項。
          color_continuous_scale (str):[顏色變化序列的樣式名稱](https://plotly.com/python/builtin-colorscales/)

        Returns:
            (plotly.graph_objects.Figure): 策略部位旭日圖
        Examples:
            ```py
            from finlab.plot import StrategySunburst

            # 實例化物件
            strategies = StrategySunburst()
            strategies.plot().show()
            strategies.plot(select_strategy={'高殖利率烏龜':0.4,'營收強勢動能瘋狗':0.25,'低波動本益成長比':0.2,'現金':0.15},path =  ['market', 'category','stock_id','s_name']).show()
            ```
        ex1:策略選到哪些標的?
        ![市值/本益比板塊圖](img/plot/sunburst1.png)

        ex2:部位被哪些策略選到，標的若被不同策略選到，可能有獨特之處喔！
        ![市值/本益比板塊圖](img/plot/sunburst2.png)
        """
        position = self.get_strategy_df(select_strategy)
        if position is None:
            return
        position = position[(position['return'] != np.inf)
                            | (position['weight'] != np.inf)]
        if path is None:
            path = ['s_name', 'market', 'category', 'stock_id']
        fig = px.sunburst(position, path=path, values='weight',
                          color='color', hover_data=['return'],
                          color_continuous_scale=color_continuous_scale,
                          color_continuous_midpoint=0,
                          width=1000, height=800)
        return fig


class StrategyReturnStats:

    def __init__(self, start_date: str, end_date: str, strategy_names=[], benchmark_return=None):
        """繪製策略報酬率統計比較圖

        監控策略群體相對對標指數的表現。

        Args:
            start_date (str): 報酬率計算開始日
            end_date (str): 報酬率計算結束日
            strategy_names (list): 用戶本人的策略集設定，填入欲納入統計的策略名稱，只限定自己的策略。ex:`['膽小貓','三頻率RSI策略', '二次創高股票',...]`，預設為全部已上傳的策略。
            benchmark_return (pandas.Series): 策略比對基準序列，預設為台股加權報酬指數。


        Examples:

            統計2022-12-31~2023-07-31的報酬率數據
            ``` py
            # 回測起始時間
            start_date = '2022-12-31'
            end_date  = '2023-07-31'

            # 選定策略範圍
            strategy_names = ['膽小貓','三頻率RSI策略', '二次創高股票', '低波動本益成長比', '合約負債建築工', '多產業價投', '小蝦米跟大鯨魚', '小資族資優生策略', '本益成長比', '營收股價雙渦輪', '現金流價值成長', '研發魔人', '股價淨值比策略', '藏獒', '高殖利率烏龜','監獄兔', '財報指標20大']

            report = StrategyReturnStats(start_date ,end_date, strategy_names)
            # 繪製策略報酬率近期報酬率長條圖
            report.plot_strategy_last_return().show()
            # 繪製策略累積報酬率時間序列
            report.plot_strategy_creturn().show()
            ```

        """
        # 回測起始時間
        self.start_date = start_date
        self.end_date = end_date

        # 選定策略範圍
        self.s_data = data.get_strategies()
        self.strategy_names = strategy_names
        self.benchmark_return = data.get('benchmark_return:發行量加權股價報酬指數')[
            '發行量加權股價報酬指數'] if benchmark_return is None else benchmark_return
        self.returns_set = self._get_returns_set()

    def _get_returns_set(self):
        """計算報酬率數據
        Returns:
            (dict): 報酬率數據，ex: `{'膽小貓': -5.98,'股價淨值比策略': -1.68,...}`
        """
        returns_set = {}
        strategy_names = self.s_data.keys() if len(
            self.strategy_names) == 0 else self.strategy_names
        for s_name in strategy_names:
            try:
                s_df = self.s_data[s_name]
                returns = pd.Series(
                    s_df['returns']['value'], index=s_df['returns']['time'])
                if self.start_date:
                    returns = returns[returns.index >= self.start_date]
                if self.end_date:
                    returns = returns[returns.index <= self.end_date]
                return_value = round(
                    ((returns.iloc[-1] / returns.iloc[0]) - 1) * 100, 2)
                returns_set[s_name] = return_value
            except:
                pass

        returns_set = dict(sorted(returns_set.items(), key=lambda x: x[1]))
        return returns_set

    def get_benchmark_return(self):
        """設定對標指數
        Returns:
            (pandas.Series): 對標指數時間序列
        """
        benchmark_return = self.benchmark_return
        benchmark_return = benchmark_return[
            (benchmark_return.index >= self.start_date) & (benchmark_return.index <= self.end_date)]
        return benchmark_return

    def plot_strategy_last_return(self):
        """繪製策略報酬率近期報酬率長條圖
        Returns:
            (plotly.graph_objects.Figure): 圖表物件
        ![繪製策略報酬率近期報酬率長條圖](img/plot/finlab_strategy_performance.png)
        """
        returns_set = self.returns_set
        benchmark_return = self.get_benchmark_return()
        benchmark_last_return = round(
            ((benchmark_return.iloc[-1] / benchmark_return.iloc[0]) - 1) * 100, 2)

        fig = go.Figure(go.Bar(
            x=list(returns_set.values()),
            y=list(returns_set.keys()),
            marker=dict(
                color='rgba(50, 171, 96, 0.6)',
                line=dict(color='rgba(50, 171, 96, 1)', width=3)
            ),
            orientation='h'))

        fig.add_vline(x=benchmark_last_return, line_width=3, line_dash="dash", line_color="#4a6dce",
                      annotation_text=f"benchmark:{benchmark_last_return}",
                      annotation_position="top left")

        returns_mean = round(sum((returns_set.values())) / len(returns_set), 2)

        fig.add_vline(x=returns_mean, line_width=3, line_dash="dash", line_color="#aa00ff",
                      annotation_font_color="#aa00ff", annotation_text=f"strategy_mean:{returns_mean}",
                      annotation_position="bottom right")

        fig.update_layout(
            title=f'FinLab Strategy Performance ({self.start_date}~{self.end_date})',
            legend=dict(x=0.029, y=1.038, font_size=12),
            margin=dict(l=100, r=20, t=70, b=70),
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            xaxis_title="return(%)",
        )

        return fig

    def plot_strategy_creturn(self):
        """繪製策略累積報酬率時間序列
        Returns:
            (plotly.graph_objects.Figure): 圖表物件
        ![繪製策略累積報酬率時間序列](img/plot/finlab_strategy_creturn.png)
        """

        returns_set = self.returns_set
        benchmark_return = self.get_benchmark_return()

        sorted_s_names = list(returns_set.keys())[::-1]

        returns_set2 = {}
        for s_name in sorted_s_names:
            s_df = self.s_data[s_name]
            returns = pd.Series(s_df['returns']['value'],
                                index=s_df['returns']['time'])
            if self.start_date:
                returns = returns[returns.index >= self.start_date]
            if self.end_date:
                returns = returns[returns.index <= self.end_date]
            returns_set2[s_name] = round(
                ((returns / returns.iloc[0]) - 1) * 100, 2)

        returns_df = pd.DataFrame(
            returns_set2).unstack().to_frame().reset_index()
        returns_df.columns = ['strategy', 'date', 'creturns']
        benchmark_creturn = round(
            ((benchmark_return / benchmark_return.iloc[0]) - 1) * 100, 2)

        fig = px.line(returns_df, x="date", y="creturns", color="strategy")
        fig.add_trace(
            go.Scatter(x=benchmark_creturn.index, y=benchmark_creturn.values, fill='tonexty', name='benchmark',
                       opacity=0.2,
                       marker=dict(
                           color='LightSkyBlue',
                           opacity=0.2,
                           size=20,
                           line=dict(
                               color='MediumPurple',
                               width=2
                           )
                       ), ))

        fig.update_layout(
            title=f'FinLab Strategy Creturn ({self.start_date}~{self.end_date})',
            margin=dict(l=100, r=20, t=70, b=70),
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            yaxis_title="creturn (%)",
        )

        fig.update_layout(yaxis_range=(
            returns_df["creturns"].min() - 0.1, returns_df["creturns"].max() + 0.1))

        return fig
