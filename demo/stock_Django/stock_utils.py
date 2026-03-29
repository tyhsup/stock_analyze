import pandas as pd
import numpy as np
import talib
import datetime, io, urllib, base64, time, os
from sklearn.preprocessing import MinMaxScaler
from stock_Django import mySQL_OP

class StockUtils:
    @staticmethod
    def load_data(table, stock_name='NA'):
        SQL_OP = mySQL_OP.OP_Fun()
        if stock_name == 'NA':
            dl_cost_table = SQL_OP.get_cost_data(table_name=table)
        else:
            dl_cost_table = SQL_OP.get_cost_data(table_name=table, stock_number=stock_name)
        return dl_cost_table

    @staticmethod
    def load_data_c(table, stock_name):
        SQL_OP = mySQL_OP.OP_Fun()
        # Try both without and with .TW suffix for robustness
        dl_cost_table = SQL_OP.get_cost_data(table_name=table, stock_number=stock_name)
        
        if dl_cost_table.empty and stock_name.isdigit():
            tw_suffix = f"{stock_name}.TW"
            dl_cost_table = SQL_OP.get_cost_data(table_name=table, stock_number=tw_suffix)
            
        if dl_cost_table.empty and stock_name.isdigit():
            two_suffix = f"{stock_name}.TWO"
            dl_cost_table = SQL_OP.get_cost_data(table_name=table, stock_number=two_suffix)
            
        if dl_cost_table.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Normalize Date column: strip time component so deduplication works properly
        # (some DB entries may have both 00:00 and HH:MM:SS for the same calendar day)
        dl_cost_table['Date'] = pd.to_datetime(dl_cost_table['Date']).dt.normalize()
        # Keep last row per date (most recent intraday entry)
        dl_cost_table = dl_cost_table.sort_values('Date').drop_duplicates(subset='Date', keep='last')
        dl_cost_table = dl_cost_table.reset_index(drop=True)
            
        cost_data = dl_cost_table[['Open', 'Close', 'High', 'Low', 'Volume']]
        Date_data = dl_cost_table[['Date']]
        return cost_data, Date_data


    @staticmethod
    def load_stock_number_all(table):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.get_cost_data(table_name=table)
        stock_number = dl_cost_table['有價證卷代號'].drop_duplicates().to_list()
        return stock_number

    @staticmethod
    def load_stock_number(table):
        SQL_OP = mySQL_OP.OP_Fun()
        stock_list = SQL_OP.get_cost_data(table_name=table)
        list_copy = stock_list.copy()
        Industry = list_copy['產業別'].drop_duplicates()
        Industry = Industry.reset_index(drop=True)
        # Choosing a default industry or specific index logic from original code
        mask = (stock_list['產業別'] == str(Industry.iat[20]))
        mask_data = (stock_list.loc[mask])['有價證卷代號']
        return mask_data

    @staticmethod
    def normalize(data, Min, Max):
        columns_name = data.columns.to_list()
        scaler = MinMaxScaler(feature_range=(Min, Max)).fit(data[columns_name])
        data[columns_name] = scaler.transform(data[columns_name])
        return data

    @staticmethod
    def de_normalize(Ori_data, norm_value, de_nor_column):
        original_value = Ori_data.iloc[:, de_nor_column].to_numpy().reshape(-1, 1)
        norm_value = norm_value.reshape(-1, 1)
        scaler = MinMaxScaler().fit(original_value)
        de_norm_value = scaler.inverse_transform(norm_value)
        return de_norm_value

    @staticmethod
    def train_test_split(data, split_rate):
        train_size = int(split_rate * len(data))
        data_train, data_test = data[:train_size], data[train_size:]
        return data_train, data_test

    @staticmethod
    def data_preprocesing(data, time_frame, split_rate, columns):
        number_features = len(data.columns)
        data = data.to_numpy()
        result = []
        if len(data) > 5:
            for i in range(len(data) - (time_frame + 1)):
                result.append(data[i: i + (time_frame + 1)])
            result = np.array(result)
            data_split = int(result.shape[0] * split_rate)
            X_train = result[:data_split, :-1]
            Y_train = result[:data_split, -1][:, columns]
            X_test = result[data_split:, :-1]
            Y_test = result[data_split:, -1][:, columns]
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], number_features))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], number_features))
            return X_train, Y_train, X_test, Y_test
        return None

    @staticmethod
    def data_index(data):
        close_SMA_10 = talib.SMA(data['Close'], 10)
        macd, macd_signal, macd_hist = talib.MACD(data['Close'])
        close_RSI_14 = talib.RSI(data['Close'], 14)
        upperband, middleband, lowerband = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data_concat = pd.concat([close_SMA_10, close_RSI_14, macd, macd_signal, macd_hist, upperband,
                                 middleband, lowerband], axis=1)
        data_concat.columns = ['SMA_10', 'RSI', 'macd', 'macd_signal', 'macd_hist', 'upperband', 'middleband', 'lowerband']
        final_data = pd.concat([data, data_concat], axis=1)
        final_data = final_data.dropna().drop_duplicates().reset_index(drop=True)
        return final_data

    @staticmethod
    def trend_indicators(data):
        if data['Date'].dtype != 'period[D]':
            data['Date'] = pd.DatetimeIndex(data['Date']).to_period('D')
        data_copy = data.copy()
        data_copy['漲跌幅度'] = data_copy['Close'].pct_change() * 100
        MA_list = [5, 10, 20, 60, 120, 240]
        for i in MA_list:
            data_copy[f'WMA_{i}'] = talib.WMA(data_copy['Close'], i)
        macd, macd_signal, macd_hist = talib.MACD(data_copy['Close'])
        MACD_dict = pd.DataFrame({'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist})
        final_data = pd.concat([data_copy, MACD_dict], axis=1)
        final_data.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'number'], axis=1, inplace=True)
        return final_data

    @staticmethod
    def momentum_indicators(data):
        if data['Date'].dtype != 'period[D]':
            data['Date'] = pd.DatetimeIndex(data['Date']).to_period('D')
        data_copy = data.copy()
        RSI_list = [7, 14, 21, 50]
        for i in RSI_list:
            data_copy[f'RSI_{i}'] = talib.RSI(data_copy['Close'], timeperiod=i)
        MOM_list = [5, 10, 20, 50]
        for j in MOM_list:
            data_copy[f'MOM_{j}'] = talib.MOM(data_copy['Close'], timeperiod=j)
        CCI_list = [14, 20, 30, 50]
        for k in CCI_list:
            data_copy[f'CCI_{k}'] = talib.CCI(data_copy['High'], data_copy['Low'], data_copy['Close'], timeperiod=k)
        data_copy.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'number'], axis=1, inplace=True)
        return data_copy

    @staticmethod
    def volatility_indicators(data):
        if data['Date'].dtype != 'period[D]':
            data['Date'] = pd.DatetimeIndex(data['Date']).to_period('D')
        data_copy = data.copy()
        upperband, middleband, lowerband = talib.BBANDS(data_copy['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        BBAND_dict = pd.DataFrame({'upperband_20': upperband, 'middleband_20': middleband, 'lowerband_20': lowerband})
        merge_data = pd.concat([data_copy, BBAND_dict], axis=1)
        ATR_list = [7, 14, 21, 50]
        for i in ATR_list:
            merge_data[f'ATR_{i}'] = talib.ATR(merge_data['High'], merge_data['Low'], merge_data['Close'], timeperiod=i)
        merge_data.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'number'], axis=1, inplace=True)
        return merge_data

    @staticmethod
    def volume_indicators(data):
        if data['Date'].dtype != 'period[D]':
            data['Date'] = pd.DatetimeIndex(data['Date']).to_period('D')
        data_copy = data.copy()
        data_copy['OBV'] = talib.OBV(data_copy['Close'], data_copy['Volume'])
        
        # Get stock number from data
        stock_num = data['number'].iloc[0] if 'number' in data.columns else '2330'
        if ".TW" in str(stock_num).upper():
            stock_num = str(stock_num).upper().replace(".TW", "")
            
        SQL_OP = mySQL_OP.OP_Fun()
        
        # Ensure stock_num is cleaned to prevent injection, even if OP_Fun handles it
        if not str(stock_num).replace('.TW', '').replace('.TWO', '').isalnum():
             return data_copy
             
        load_investor = SQL_OP.get_cost_data(table_name='stock_investor', stock_number=str(stock_num))
        
        if load_investor.empty:
            return data_copy
            
        # Bypass garbled column names using index-based mapping
        # 0: Date, 5: Foreign Net, 8: Ext Net, 11: IT Net, 15: Dealer Buy, 18: Dealer Hedge, 19: Total
        investor_indices = {
            0: '日期',
            5: '外陸資買賣超股數(不含外資自營商)',
            8: '外資自營商買賣超股數',
            11: '投信買賣超股數',
            15: '自營商買賣超股數(自行買賣)',
            18: '自營商買賣超股數(避險)',
            19: '三大法人買賣超股數'
        }
        
        # Rename by column position
        new_cols = []
        for i in range(len(load_investor.columns)):
            new_cols.append(investor_indices.get(i, f"col_{i}"))
        load_investor.columns = new_cols
        
        select_investor_column = load_investor.loc[:, list(investor_indices.values())]
        Investor_data = StockUtils.transfer_numeric(select_investor_column)
        merge_data = pd.merge(data_copy, Investor_data, left_on='Date', right_on='日期', how='left')
        merge_data.drop(['日期'], axis=1, inplace=True)
        return merge_data

    @staticmethod
    def Sentiment_indicators(stock_number: str, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        # Use configurable path for news analysis results
        webbug_dir = os.getenv('WEBBUG_DIR', 'E:/Infinity/webbug/')
        news_path = os.path.join(webbug_dir, f'{stock_number}_news_新聞正負分析結果(day).xlsx')
        
        if not os.path.exists(news_path):
            return pd.DataFrame(columns=['Date', '正面新聞占比', '負面新聞占比'])
            
        news = pd.read_excel(news_path)
        news['Date'] = pd.DatetimeIndex(news['Date']).to_period('D')
        merge_data = pd.merge(news, data_copy, left_on='Date', right_on='Date', how='left')
        Sentiment_data = merge_data.loc[:, ['Date', '正面新聞占比', '負面新聞占比']]
        return Sentiment_data

    @staticmethod
    def transfer_numeric(data):
        # Handle potential duplicate columns by de-duplicating names first
        data = data.loc[:, ~data.columns.duplicated()].copy()
        cols = data.columns
        for col in cols:
            if col == '日期':
                try:
                    data['日期'] = pd.to_datetime(data['日期']).dt.to_period('D')
                except Exception:
                    pass
            elif col in ['number', '證券名稱']:
                continue
            else:
                series = data[col]
                # Check if it's object (string) type
                if pd.api.types.is_object_dtype(series):
                    data[col] = series.astype(str).str.replace(',', '', regex=False)
                
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        return data

    @staticmethod
    def date_create(start_date, days):
        bdate_range = pd.bdate_range(start=str(start_date), periods=days, freq='B', name='Date')
        date = pd.DataFrame(bdate_range)
        return date

    @staticmethod
    def classify_change(data):
        percent = data['漲跌幅度']
        if pd.isna(percent):
            return 'NaN'
        elif percent > 2:
            return 'U'
        elif percent < -2:
            return 'D'
        else:
            return 'K'

    @staticmethod
    def summary_rules(data):
        code = data['漲跌代號']
        pos = data['正面新聞占比']
        nes = data['負面新聞占比']
        if code == 'U':
            return "強利多, 上漲趨勢延續" if pos > nes else "利空不跌, 主力護盤或有隱性利多"
        elif code == 'D':
            return "利多不漲, 市場不信任或系統誤判" if pos > nes else "強利空, 下跌趨勢延續"
        elif code == 'K':
            return "醞釀突破" if pos > nes else "醞釀下跌"
        return "Unknown"
