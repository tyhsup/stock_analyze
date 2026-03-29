import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import plotly.tools as tls
import plotly.graph_objects as go
from matplotlib.font_manager import fontManager
import mplfinance as mpf
import mpld3
from io import BytesIO
import pandas as pd
import talib
import json
import numpy as np
import logging

from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class chart_create:

    def cost_plt(self, Pred_data, real_data, stock_name):
        plt.figure(figsize=(12,6))
        plt.plot(real_data, color = 'red', label = 'Real Stock Close Price')
        plt.plot(Pred_data, color = 'blue', label = 'Predicted Stock Close Price')
        plt.title(str(stock_name) + ' ' + 'Stock Price Prediction')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price TWD ($)', fontsize=18)
        plt.legend()
        plt.show()
        
    def kline_MA(self, data, pred_days = 'NA'):
        #buf = BytesIO()
        data.set_index('Date', inplace = True)
        sma_5 = mpf.make_addplot(talib.SMA(data["Close"], 5), color = 'cyan', label = 'sma_5')
        sma_20 = mpf.make_addplot(talib.SMA(data["Close"], 20), color = 'orange', label = 'sma_20')
        sma_60 = mpf.make_addplot(talib.SMA(data["Close"], 60), color = 'purple',label = 'sma_60')
        Candle = mpf.make_addplot(data, color = 'blue', type = 'candle')
        mpf.plot(data, type = 'line', style = 'yahoo', addplot = [sma_5, sma_20, sma_60, Candle],
                 volume = True, volume_panel = 1)#, savefig = buf, block = True)
        #buf.seek(0)
        #return buf
    
    def kline_ver2(self, data, date, pred_days = 'NA'):
        fig = mpf.figure()
        ax1 = fig.subplot()
        ax2 = ax1.twinx()
        ax3 = fig.subplot()
        data = pd.concat([date,data],axis = 1)
        data.set_index('Date',inplace = True)
        ap = [mpf.make_addplot(talib.SMA(data["Close"], 5), color = 'cyan', label = 'sma_5', ax = ax1),
              mpf.make_addplot(talib.SMA(data["Close"], 20), color = 'orange', label = 'sma_20',ax = ax1),
              mpf.make_addplot(talib.SMA(data["Close"], 60), color = 'purple',label = 'sma_60',ax = ax1),
              mpf.make_addplot(data, color = 'blue', type = 'candle',ax = ax1)
            ]
        mpf.plot(data, type = 'line', style = 'yahoo', ax = ax2, addplot = ap, volume = ax3, returnfig = True,
                             block = False,show_nontrading=True)
        mpld3.show(fig = fig)
        
    def kline_plotly(self, data, symbol=""):
        if data is None or data.empty:
            return None
            
        from plotly.subplots import make_subplots
        
        # Ensure index is datetime and sorted
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # Create subplots: 2 rows, 1 column, shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # 1. Candlestick Chart (Row 1)
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='#ef5350', # Red for up in TW market
            decreasing_line_color='#26a69a'  # Green for down in TW market
        ), row=1, col=1)
        
        # Add SMA indicators
        for ma, color in zip([5, 20, 60], ['#29b6f6', '#ffa726', '#ab47bc']):
            if len(data) >= ma:
                sma = talib.SMA(data['Close'], ma)
                if not np.isnan(sma).all():
                    fig.add_trace(go.Scatter(
                        x=data.index, 
                        y=sma, 
                        name=f'MA {ma}', 
                        line=dict(width=1, color=color),
                        opacity=0.8
                    ), row=1, col=1)

        # 2. Volume Bar Chart (Row 2)
        colors = ['#ef5350' if row['Close'] >= row['Open'] else '#26a69a' for _, row in data.iterrows()]
        fig.add_trace(go.Bar(
            x=data.index, 
            y=data['Volume'], 
            name='Volume',
            marker_color=colors,
            opacity=0.8
        ), row=2, col=1)

        # Formatting
        fig.update_layout(
            title=dict(text=f'{symbol} Historical Price Trend (OHLCV)', x=0.5, font=dict(size=20)),
            template='plotly_white',
            xaxis_rangeslider_visible=False, # Hide for cleaner look, row 2 serves as visual range
            height=650,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Fix X-axis date formatting and remove gaps for non-trading days
        fig.update_xaxes(
            type='category', # Use category to remove gaps
            tickformat='%Y-%m-%d',
            dtick=max(1, len(data)//10), # Show ~10 ticks
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig.to_json()

    def kline_apex(self, data: pd.DataFrame, symbol: str = "", ai_pred: dict = None) -> Optional[Dict[str, Any]]:
        """生成 ApexCharts 格式的 K 線圖資料。

        Args:
            data (pd.DataFrame): 包含 OHLCV 的股價資料。
            symbol (str, optional): 股票代碼。 預設為 ""。
            ai_pred (dict, optional): 包含歷史預測與未來預測的字典。

        Returns:
            Optional[Dict[str, Any]]: ApexCharts 配置字典，若資料為空則回傳 None。
        """
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # Vectorized timestamp and series data construction
        timestamps = (data.index.astype(np.int64) // 10**6).tolist()
        opens = data['Open'].tolist()
        highs = data['High'].tolist()
        lows = data['Low'].tolist()
        closes = data['Close'].tolist()
        volumes = data['Volume'].tolist()
        
        series_data = [
            {'x': ts, 'y': [float(o), float(h), float(l), float(c)]}
            for ts, o, h, l, c in zip(timestamps, opens, highs, lows, closes)
        ]
        
        # Volume with color logic: green=up(close>=open), red=down
        volume_data = [
            {'x': ts, 'y': float(v), 'fillColor': '#26a69a' if c >= o else '#ef5350'}
            for ts, o, c, v in zip(timestamps, opens, closes, volumes)
        ]
            
        # Prepare MAs
        ma_series = []
        for ma, color in zip([5, 20, 60], ['#29b6f6', '#ffa726', '#ab47bc']):
            if len(data) >= ma:
                sma = talib.SMA(data['Close'], ma)
                points = []
                for ts, val in zip(timestamps, sma):
                    if not np.isnan(val):
                        points.append({'x': ts, 'y': float(val)})
                if points:
                    ma_series.append({'name': f'MA{ma}', 'color': color, 'data': points})
                    
        # Process AI Prediction
        ai_hist_series = []
        ai_future_series = []
        
        if ai_pred:
            hist = ai_pred.get('historical', [])
            latest = ai_pred.get('latest', None)
            
            # 1. Historical Trajectory (Solid Line)
            hist_points = []
            for item in hist:
                dt = pd.to_datetime(item['date'])
                # Shift 1 business day backward or forward depending on definition.
                # Since model predicts T+1 based on T ending data:
                next_dt = dt + pd.offsets.BDay(1)
                ts = int(next_dt.timestamp() * 1000)
                hist_points.append({'x': ts, 'y': float(item['pred'])})
            
            if hist_points:
                ai_hist_series.append({
                    'name': 'AI Pred (History)',
                    'type': 'line',
                    'color': '#ff0000', # Red solid line
                    'data': hist_points
                })
                
            # 2. Future 5-day Prediction (Dashed Line logic)
            future_points = []
            if latest and 'predictions' in latest:
                preds = latest['predictions']
                last_dt = pd.to_datetime(data.index[-1])
                for i, p in enumerate(preds):
                    fut_dt = last_dt + pd.offsets.BDay(i + 1)
                    ts = int(fut_dt.timestamp() * 1000)
                    future_points.append({'x': ts, 'y': float(p)})
                    
            if future_points:
                # Add the last actual close price to connect the line visually
                last_ts = timestamps[-1]
                last_close = closes[-1]
                conn_points = [{'x': last_ts, 'y': last_close}] + future_points
                
                ai_future_series.append({
                    'name': 'AI Pred (Future)',
                    'type': 'line',
                    'color': '#ff0000', # Will use strokeDashArray in frontend
                    'data': conn_points
                })
                    
        config = {
            'candlestick': {'name': 'OHLC', 'data': series_data},
            'volume': {'name': 'Volume', 'data': volume_data},
            'ma': ma_series,
            'ai_history': ai_hist_series,
            'ai_future': ai_future_series,
            'symbol': symbol
        }
        return config  # Return dict - Django's json_script filter will serialize it

    def kline_png(self, data, symbol=""):
        """
        Generates a K-line chart using mplfinance and returns it as a base64 encoded string.
        """
        import io
        import urllib.parse
        import base64
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Ensure data index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        buf = io.BytesIO()
        
        # Define the style and plot
        mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
        s  = mpf.make_mpf_style(marketcolors=mc)
        
        # Prepare SMA indicators only if sufficient data exists
        ap = []
        for ma, color, name in [(5, 'blue', 'MA5'), (20, 'orange', 'MA20'), (60, 'purple', 'MA60')]:
            if len(data) >= ma:
                sma_data = talib.SMA(data["Close"], ma)
                if not np.isnan(sma_data).all():
                    ap.append(mpf.make_addplot(sma_data, color=color, width=0.8, label=name))
        
        # Set plot arguments
        plot_args = dict(
            type='candle', style=s, volume=True,
            title=f"\n{symbol} Price & Volume",
            figsize=(12, 8),
            savefig=dict(fname=buf, format='png', dpi=100)
        )
        if ap:
            plot_args['addplot'] = ap

        try:
            mpf.plot(data, **plot_args)
        except Exception as e:
            # Fallback for very sparse data that might still crash mpf
            # (e.g., all O/H/L/C same or all NaNs)
            logger.warning(f"mpf.plot failed for {symbol}: {e}")
            return None
        
        buf.seek(0)
        img_str = urllib.parse.quote(base64.b64encode(buf.read()))
        plt.close('all')
        return img_str

    def investor_plotly(self, data, symbol=""):
        # data is from get_investor which has Date as index
        # Columns selection
        mask = ['外陸資買賣超股數(不含外資自營商)', '投信買賣超股數', '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)']
        cols_to_plot = [c for c in mask if c in data.columns]
        
        fig = go.Figure()
        for col in cols_to_plot:
            fig.add_trace(go.Bar(x=data.index.astype(str), y=data[col], name=col))

        fig.update_layout(
            title=f'{symbol} Institutional Investor Trends',
            xaxis_title='Date',
            yaxis_title='Shares',
            barmode='relative',
            template='plotly_white',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(
            type='category', 
            tickformat='%Y-%m-%d',
            dtick=max(1, len(data)//8)
        )
        
        fig.update_traces(hovertemplate='%{x}<br>Shares: %{y:,.0f}')
        return fig.to_json()

    def investor_apex(self, data: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """生成 ApexCharts 格式的三大法人買賣超圖表資料。

        Args:
            data (pd.DataFrame): 法人買賣超資料。
            symbol (str, optional): 股票代碼。 預設為 ""。

        Returns:
            Optional[Dict[str, Any]]: ApexCharts 配置字典，若資料為空則回傳 None。
        """
            
        # The 4 requested columns
        req_cols = [
            '外陸資買賣超股數(不含外資自營商)',
            '投信買賣超股數',
            '自營商買賣超股數(自行買賣)',
            '自營商買賣超股數(避險)'
        ]
        
        name_map = {
            '外陸資買賣超股數(不含外資自營商)': '外陸資(不含自營商)',
            '投信買賣超股數': '投信',
            '自營商買賣超股數(自行買賣)': '自營商(自行買賣)',
            '自營商買賣超股數(避險)': '自營商(避險)'
        }

        # Check for matching garbled versions
        series = []
        for col in data.columns:
            name_to_use = None
            if '外' in col and '資' in col and '不含' in col and '買賣超' in col:
                name_to_use = name_map['外陸資買賣超股數(不含外資自營商)']
            elif '投' in col and '信' in col and '買賣超' in col:
                name_to_use = name_map['投信買賣超股數']
            elif '自' in col and '買' in col and '自行' in col and '買賣超' in col:
                name_to_use = name_map['自營商買賣超股數(自行買賣)']
            elif '自' in col and '營' in col and '避險' in col and '買賣超' in col:
                name_to_use = name_map['自營商買賣超股數(避險)']
            
            if name_to_use:
                # To prevent duplicates if multiple messy cols match, check existing series names
                if not any(s['name'] == name_to_use for s in series):
                    series.append({
                        'name': name_to_use,
                        'data': [float(x) for x in data[col].tolist()]
                    })
        
        if not series:
            # Fallback to precise exact match if available
            for c in req_cols:
                if c in data.columns:
                    series.append({
                        'name': name_map[c],
                        'data': [float(x) for x in data[c].tolist()]
                    })
        
        categories = data.index.astype(str).tolist()
        
        return {
            'series': series,
            'categories': categories,
            'symbol': symbol
        }
        
    def last_investor_H_T(self, data, amount):
        investor_last_time = data.sort_values(by = '日期', ascending = True).iloc[-1].at['日期']
        mask_last_date = (data['日期']==investor_last_time)
        investor_lastday_data = data[mask_last_date].copy()
        investor_lastday_data.sort_values(by = '三大法人買賣超股數', ascending = False, inplace = True)
        investor_head= investor_lastday_data.head(int(amount))
        investor_head.set_index('日期', inplace = True)
        investor_tail= investor_lastday_data.tail(int(amount))
        investor_tail.set_index('日期', inplace = True)
        return investor_head,investor_tail
    
    def get_investor(self, data,stock,days):
        mask = (data['number']==str(stock))
        data2 = data[mask].copy()
        data_sort = data2.sort_values(by = '日期', ascending = True)
        investor_tail = data_sort.tail(days)
        investor_tail.set_index('日期', inplace = True)
        return investor_tail

    def investor_plt(self, data, axes = 'NA'):
        matplotlib.rc('font', family='Noto Sans SC')
        cols = data.columns
        number = data['number'].to_list()
        data = data.drop(['number','證券名稱'],axis=1)
        cols = data.columns
        mask = ['外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數','自營商買賣超股數(自行買賣)','自營商買賣超股數(避險)']
        data = data[mask]
        #繪製堆疊柱狀圖
        if axes == 'NA':
            fig = data.plot(kind = 'bar',stacked = True, figsize = (12,6),fontsize = 15, title = str(number[-1]),rot = 75)
        else :
            fig = data.plot(kind = 'bar',stacked = True, figsize = (12,10),fontsize = 8, ax = axes,
                  title = str(number[-1]), rot = 75, legend = False)
        return fig

    def investor_TOP_plt(self, data, amount, days):
        load_data_H,load_data_T = self.last_investor_H_T(data, amount)
        H_number = load_data_H['number']
        T_number = load_data_T['number']
        matplotlib.rc('font', family='Noto Sans SC')
        fig, axes = plt.subplots(nrows = 2, ncols = amount, sharex = True, sharey = True)
        for i in range(len(H_number)):
            H_data = self.get_investor(data, H_number.iloc[i],days)
            self.investor_plt(H_data, axes = axes[0,i])
            if i == 0 :
                handles, labels = axes[0,i].get_legend_handles_labels()
        for i in range(len(T_number)):
            T_data = self.get_investor(data, T_number.iloc[i],days)
            self.investor_plt(T_data, axes = axes[1,i])
        fig.suptitle('investor ' + 'TOP' + str(amount) + ' Increase/decrease')
        fig.legend(handles, labels, loc = 'outside upper right', fontsize = 7)
        fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
        return fig
    def investor_comparison_apex(self, data, amount=5, days=5):
        """
        Build ApexCharts stacked bar data for top N leader/laggard stocks
        showing the last `days` days of 買賣超 activity.
        Returns a dict ready for json_script.
        """
        if data is None or data.empty:
            return None

        # Columns with 買賣超 in name
        buysell_mask = ['外陸資買賣超股數(不含外資自營商)', '投信買賣超股數', '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)']
        buysell_cols = [c for c in buysell_mask if c in data.columns]

        if '三大法人買賣超股數' not in data.columns or not buysell_cols:
            return None

        try:
            # Get top N head and tail stocks by latest day 三大法人買賣超股數
            investor_H, investor_T = self.last_investor_H_T(data, amount)
            top_tickers_H = investor_H['number'].tolist() if 'number' in investor_H.columns else []
            top_tickers_T = investor_T['number'].tolist() if 'number' in investor_T.columns else []
            ticker_names_H = investor_H['證券名稱'].tolist() if '證券名稱' in investor_H.columns else top_tickers_H
            ticker_names_T = investor_T['證券名稱'].tolist() if '證券名稱' in investor_T.columns else top_tickers_T

            def build_series_for_tickers(tickers, names):
                """For each ticker, get last `days` days; produce series per buysell column."""
                # Collect per-ticker per-day data; x-axis = ticker+date labels
                categories = []
                series_data = {col: [] for col in buysell_cols}
                for ticker, name in zip(tickers, names):
                    ticker_data = self.get_investor(data, ticker, days)
                    for date_label, row in ticker_data.iterrows():
                        label = f"{name}\n{str(date_label)[:10]}"
                        categories.append(label)
                        for col in buysell_cols:
                            series_data[col].append(float(row[col]) if col in row and not pd.isna(row[col]) else 0)
                series = [{'name': col, 'data': series_data[col]} for col in buysell_cols]
                return categories, series

            cat_H, series_H = build_series_for_tickers(top_tickers_H, ticker_names_H)
            cat_T, series_T = build_series_for_tickers(top_tickers_T, ticker_names_T)

            return {
                'leaders': {'categories': cat_H, 'series': series_H},
                'laggards': {'categories': cat_T, 'series': series_T},
            }
        except Exception as e:
            logger.error(f"investor_comparison_apex error: {e}")
            return None

    def investor_us_apex(self, df, symbol=""):
        """
        Format US institutional holder data for ApexCharts.
        Enhanced to include concentration, sentiment, and total ownership metrics.
        """
        if df is None or df.empty:
            return None

        # Sort by shares descending
        if 'shares' in df.columns:
            df = df.sort_values('shares', ascending=False)
            
        # Top 15 for the horizontal bar chart
        top_df = df.head(15).copy()

        def shorten(name, maxlen=32):
            return name[:maxlen] + '…' if len(str(name)) > maxlen else str(name)

        def safe_float(v):
            if pd.isna(v) or not np.isfinite(v): return 0.0
            return float(v)

        def safe_int(v):
            if pd.isna(v) or not np.isfinite(v): return 0
            return int(v)

        holders = [shorten(h) for h in top_df['holder_name'].tolist()]
        shares  = [safe_int(v) for v in top_df['shares'].tolist()]
        pct_out = [round(safe_float(v) * 100, 2) for v in top_df['pct_out'].tolist()]
        changes = [safe_int(v) for v in top_df.get('change_shares', pd.Series([0]*len(top_df))).tolist()]
        change_pcts = [round(safe_float(v) * 100, 2) for v in top_df.get('change_pct', pd.Series([0]*len(top_df))).tolist()]

        # Bar colors based on change
        bar_colors = []
        for c in change_pcts:
            if c > 0: bar_colors.append('#26a69a')
            elif c < 0: bar_colors.append('#ef5350')
            else: bar_colors.append('#78909c')

        # NEW: Concentration Analysis
        total_shares = df['shares'].sum()
        top_10_shares = df.head(10)['shares'].sum()
        others_shares = total_shares - top_10_shares
        
        concentration_data = [
            {'label': 'Top 10 Holders', 'value': int(top_10_shares)},
            {'label': 'Other Institutions', 'value': int(others_shares)}
        ]

        # NEW: Sentiment Analysis (Buy vs Sell vs New vs Sold Out)
        # Assuming change_shares > 0 is buy, < 0 is sell
        buy_vol = safe_float(df[df['change_shares'] > 0]['change_shares'].sum())
        sell_vol = safe_float(abs(df[df['change_shares'] < 0]['change_shares'].sum()))
        
        sentiment = {
            'buy_volume': float(buy_vol) if np.isfinite(buy_vol) else 0.0,
            'sell_volume': float(sell_vol) if np.isfinite(sell_vol) else 0.0,
            'ratio': round(buy_vol / sell_vol, 2) if sell_vol > 0 else (100.0 if buy_vol > 0 else 1.0)
        }

        # NEW: Global metrics
        total_pct = round(df['pct_out'].sum() * 100, 2)

        return {
            'symbol': symbol,
            'categories': holders,
            'shares': shares,
            'pct_out': pct_out,
            'changes': changes,
            'change_pcts': change_pcts,
            'bar_colors': bar_colors,
            'concentration': concentration_data,
            'sentiment': sentiment,
            'total_institutional_pct': total_pct,
            'total_shares_held': int(total_shares)
        }

    @staticmethod
    def get_ta_indicators(data: pd.DataFrame) -> Dict[str, Any]:
        """動態計算技術指標。

        Args:
            data (pd.DataFrame): 包含 OHLCV 的股價資料。

        Returns:
            Dict[str, Any]: 分門別類的技術指標字典。
        """

        import talib
        # Standard input normalization
        O = data['Open'].values.astype(float)
        H = data['High'].values.astype(float)
        L = data['Low'].values.astype(float)
        C = data['Close'].values.astype(float)
        V = data['Volume'].values.astype(float) if 'Volume' in data.columns else np.zeros(len(data))

        groups = talib.get_function_groups()
        res = {}
        
        # Groups to exclude (per user request and debugging)
        EXCLUDE_GROUPS = ['Math Operators', 'Math Transform', 'Statistic Functions']
        
        def clean_output(arr):
            """清理 numpy 輸出中的 NaN 值。"""
            if isinstance(arr, np.ndarray):
                return [x if not (isinstance(x, float) and np.isnan(x)) else None for x in arr.tolist()]
            elif isinstance(arr, (list, tuple)):
                return [clean_output(x) for x in arr]
            return arr

        for group_name, functions in groups.items():
            if group_name in EXCLUDE_GROUPS:
                continue
                
            safe_group = group_name.lower().replace(' ', '_')
            res[safe_group] = {}
            
            for func_name in functions:
                try:
                    # Skip problematic or multi-period functions
                    if func_name in ['MAVP', 'BBANDS', 'MACD', 'MACDEXT', 'MACDFIX', 'STOCH', 'STOCHF', 'STOCHRSI']:
                        continue
                    func = getattr(talib, func_name)
                    output = None
                    
                    # Logic-based Input Mapping
                    if group_name == 'Pattern Recognition':
                        output = func(O, H, L, C)
                    elif group_name == 'Volume Indicators':
                        if func_name in ['AD', 'ADOSC']: output = func(H, L, C, V)
                        elif func_name == 'OBV': output = func(C, V)
                        else: output = func(H, L, C, V)
                    elif group_name == 'Volatility Indicators':
                        output = func(H, L, C)
                    elif group_name == 'Price Transform':
                        if any(x in func_name for x in ['AVG', 'TYP', 'WCL']): output = func(H, L, C)
                        elif 'MED' in func_name: output = func(H, L)
                        else: output = func(C)
                    elif group_name == 'Cycle Indicators':
                        output = func(C)
                    elif group_name == 'Momentum Indicators':
                        # Precise input mapping for 30 TA-Lib Momentum Indicators
                        if func_name == 'BOP':
                            output = func(O, H, L, C)
                        elif func_name == 'MFI':
                            output = func(H, L, C, V)
                        elif func_name in ['ADX', 'ADXR', 'CCI', 'DX', 'MINUS_DI', 'PLUS_DI', 'ULTOSC', 'WILLR']:
                            output = func(H, L, C)
                        elif func_name in ['AROON', 'MINUS_DM', 'PLUS_DM']:
                            output = func(H, L)
                        else:
                            # APO, AROONOSC, CMO, MOM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, TRIX
                            output = func(C)
                    elif group_name == 'Overlap Studies':
                        if func_name in ['SAR', 'SAREXT']: output = func(H, L)
                        else: output = func(C)
                    else:
                        output = func(C)
                    
                    if output is not None:
                        if isinstance(output, tuple):
                            res[safe_group][func_name] = clean_output(output)
                        else:
                            res[safe_group][func_name] = clean_output(output)
                            
                except Exception:
                    continue

        # Add explicitly managed complex indicators
        # 1. BBANDS (20, 2)
        if len(C) >= 20:
            upper, middle, lower = talib.BBANDS(C, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            if 'overlap_studies' not in res: res['overlap_studies'] = {}
            res['overlap_studies']['BB_UPPER'] = upper.tolist()
            res['overlap_studies']['BB_MIDDLE'] = middle.tolist()
            res['overlap_studies']['BB_LOWER'] = lower.tolist()

        # 2. MACD (12, 26, 9)
        if len(C) >= 35:
            macd, signal, hist = talib.MACD(C, fastperiod=12, slowperiod=26, signalperiod=9)
            if 'momentum_indicators' not in res: res['momentum_indicators'] = {}
            res['momentum_indicators'].update({'MACD': macd.tolist(), 'MACD_SIGNAL': signal.tolist(), 'MACD_HIST': hist.tolist()})

        # 3. STOCH
        if len(H) >= 10:
            slowk, slowd = talib.STOCH(H, L, C, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            if 'momentum_indicators' not in res: res['momentum_indicators'] = {}
            res['momentum_indicators'].update({'STOCH_K': slowk.tolist(), 'STOCH_D': slowd.tolist()})

        # Sanitize all outputs
        import math
        def sanitize(v):
            if isinstance(v, list): return [sanitize(x) for x in v]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return v
        return {g: {f: sanitize(v) for f, v in fns.items()} for g, fns in res.items()}

    # -----------------------------------------------------------------------
    # TW Investor Summary Charts (mirrors investor_us_apex schema for home.html)
    # -----------------------------------------------------------------------
    def investor_tw_summary_apex(self, data, symbol=""):
        """
        Build a US-investor-apex-style dict for a TW stock, using 三大法人 data.
        `data` is the result of `get_investor(data2, ticker, days)` — date-indexed DataFrame.
        Returns the same schema as investor_us_apex so the same JS/template can render it.
        """
        if data is None or data.empty:
            return None

        # The 3 main institutional categories and their share columns
        FOREIGN = '外陸資買賣超股數(不含外資自營商)'
        TRUST   = '投信買賣超股數'
        DEALER  = '自營商買賣超股數(自行買賣)'
        HEDGE   = '自營商買賣超股數(避險)'

        avail_cols = [c for c in [FOREIGN, TRUST, DEALER, HEDGE] if c in data.columns]
        if not avail_cols:
            return None

        # Compute cumulative net buy/sell for each institution across the period
        name_map = {
            FOREIGN: '外陸資(不含自營商)',
            TRUST:   '投信',
            DEALER:  '自營商(自行買賣)',
            HEDGE:   '自營商(避險)',
        }
        categories = []
        shares = []
        pct_out = []
        changes = []
        change_pcts = []
        bar_colors = []

        for col in avail_cols:
            net = data[col].sum()
            categories.append(name_map.get(col, col))
            shares.append(int(net))
            pct_out.append(0)           # TW data doesn't have shares-outstanding pct
            changes.append(int(data[col].iloc[-1]) if len(data) else 0)
            daily_chg = (data[col].pct_change().iloc[-1] * 100) if len(data) > 1 else 0
            change_pcts.append(round(float(daily_chg) if pd.notna(daily_chg) else 0, 2))
            bar_colors.append('#26a69a' if net >= 0 else '#ef5350')

        # Concentration: top vs others (latest day)
        if '三大法人買賣超股數' in data.columns:
            latest_total = float(data['三大法人買賣超股數'].sum())
        else:
            latest_total = sum(abs(s) for s in shares)

        concentration_data = [
            {'label': col_name, 'value': abs(int(data[col].sum())) }
            for col, col_name in zip(avail_cols, [name_map.get(c, c) for c in avail_cols])
        ]

        # Sentiment (buy > 0 = bullish)
        buy_vol = sum(s for s in shares if s > 0)
        sell_vol = abs(sum(s for s in shares if s < 0))
        sentiment = {
            'buy_volume': float(buy_vol),
            'sell_volume': float(sell_vol),
            'ratio': round(buy_vol / sell_vol, 2) if sell_vol > 0 else (100 if buy_vol > 0 else 1),
        }

        return {
            'symbol': symbol,
            'categories': categories,
            'shares': shares,
            'pct_out': pct_out,
            'changes': changes,
            'change_pcts': change_pcts,
            'bar_colors': bar_colors,
            'concentration': concentration_data,
            'sentiment': sentiment,
            'total_institutional_pct': 0,
            'total_shares_held': int(sum(abs(s) for s in shares)),
            'is_tw': True,
        }

    def investor_buysell_top_apex(self, data, amount=10):
        """
        Build top-N buyer and top-N seller data for the institutional chips page.
        `data` is the full investor dataset (all stocks, all dates from MySQL).
        Returns {'leaders': {categories, series, totals}, 'laggards': {…}}
        — same schema as investor_comparison_apex but summed for 1 day rather than
        showing last N days per ticker.
        """
        if data is None or data.empty:
            return None

        TOTAL_COL = '三大法人買賣超股數'
        NAME_COL  = '證券名稱'
        NUM_COL   = 'number'

        if TOTAL_COL not in data.columns:
            return None

        try:
            # Get latest trading date
            latest_date = data['日期'].max()
            latest_data = data[data['日期'] == latest_date].copy()
            latest_data = latest_data.sort_values(TOTAL_COL, ascending=False)

            leaders = latest_data.head(amount)
            laggards = latest_data.tail(amount).sort_values(TOTAL_COL)

            buysell_cols = ['外陸資買賣超股數(不含外資自營商)', '投信買賣超股數', '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)']
            buysell_cols = [c for c in buysell_cols if c in latest_data.columns]

            def build_chart_data(df):
                cats = []
                for _, row in df.iterrows():
                    name = row.get(NAME_COL, row.get(NUM_COL, '?'))
                    cats.append(str(name)[:20])
                series = []
                for col in buysell_cols:
                    series.append({'name': col, 'data': [int(row[col]) for _, row in df.iterrows()]})
                totals = [int(row[TOTAL_COL]) for _, row in df.iterrows()]
                return {'categories': cats, 'series': series, 'totals': totals}

            return {
                'leaders': build_chart_data(leaders),
                'laggards': build_chart_data(laggards),
                'latest_date': str(latest_date),
            }
        except Exception as e:
            logger.error(f"investor_buysell_top_apex error: {e}")
            return None

