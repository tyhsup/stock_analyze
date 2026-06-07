import pandas as pd
#from stock_Django import mySQL_OP
import mySQL_OP
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import fontManager
import mplfinance as mpf
import time
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers,models
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.models import load_model
import talib
import datetime
import gc,io
from io import BytesIO
import urllib,base64
from PIL import Image
import yfinance as yf
import os
import pickle



class stock_cost_AI:
    
    def load_data(table):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.sel_cost_data(table_name = table)
        cost_data = dl_cost_table[['Open','Close','High','Low','Volume']]
        Date_data = dl_cost_table[['Date']]
        return cost_data, Date_data
    
    def load_data_c(table,stock_name):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.get_cost_data(table_name = table,stock_number = stock_name)
        cost_data = dl_cost_table[['Open','Close','High','Low','Volume']]
        Date_data = dl_cost_table[['Date']]
        return cost_data, Date_data
    
    def load_all_data(table):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.get_cost_data(table_name = table)
        return dl_cost_table
    
    def load_stock_number_all(table):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.get_cost_data(table_name = table)
        stock_number = dl_cost_table['number'].drop_duplicates().to_list()
        return stock_number
    
    def load_stock_number(table):
        SQL_OP = mySQL_OP.OP_Fun()
        stock_list = SQL_OP.get_cost_data(table_name = table)
        list_copy = stock_list.copy()
        Industry = list_copy['產業別'].drop_duplicates()
        Industry = Industry.reset_index(drop =True)
        mask = (stock_list['產業別'] == str(Industry.iat[20]))
        mask_data = (stock_list.loc[mask])['有價證卷代號']
        return mask_data      
          
    def normalize(data, Min, Max):
        columns_name = data.columns.to_list()
        scaler = MinMaxScaler(feature_range= (Min, Max)).fit(data[columns_name])
        data[columns_name] = scaler.transform(data[columns_name])
        return data
    
    def data_preprocesing(data, time_frame, split_rate, columns):
        number_features = len(data.columns)
        
        # 先切分訓練與測試集，避免資料洩露
        data_split = int(len(data) * split_rate)
        train_data = data.iloc[:data_split].copy()
        test_data = data.iloc[data_split:].copy()
        
        # 僅對訓練集進行 fit
        columns_name = data.columns.to_list()
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_data[columns_name])
        
        train_data[columns_name] = scaler.transform(train_data[columns_name])
        test_data[columns_name] = scaler.transform(test_data[columns_name])
        
        train_arr = train_data.to_numpy()
        test_arr = test_data.to_numpy()
        
        X_train, Y_train = [], []
        if len(train_arr) > time_frame:
            for i in range(len(train_arr) - time_frame):
                X_train.append(train_arr[i : i + time_frame])
                Y_train.append(train_arr[i + time_frame, columns])
                
        X_test, Y_test = [], []
        if len(test_arr) > time_frame:
            for i in range(len(test_arr) - time_frame):
                X_test.append(test_arr[i : i + time_frame])
                Y_test.append(test_arr[i + time_frame, columns])
                
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test), scaler
    
    def data_index(data):
        close_data = data['Close']
        high_data = data['High']
        low_data = data['Low']
        volume_data = data['Volume']
        open_data = data['Open']
        close_SMA_10 = talib.SMA(data['Close'],10)
        macd, macd_signal, macd_hist  = talib.MACD(data['Close'])
        close_RSI_14 = talib.RSI(data['Close'],14)
        upperband, middleband, lowerband = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data_concat = pd.concat([close_SMA_10,close_RSI_14,macd,macd_signal,macd_hist,upperband,
                                 middleband,lowerband], axis = 1)
        data_concat.columns = ['SMA_10','RSI','macd','macd_signal','macd_hist','upperband','middleband','lowerband'
                               ]
        final_data = pd.concat([data, data_concat], axis = 1)
        final_data = final_data.dropna()
        final_data = final_data.drop_duplicates()
        final_data = final_data.reset_index(drop = True)
        return final_data
    
    def model_install(input_L, input_d):
        model = tf.keras.Sequential()
        model.add(LSTM(units = 512, return_sequences = True, input_shape = (input_L,input_d)))
        model.add(Dropout(0.4))
        model.add(LSTM(units=256, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1, activation='elu'))
        model.compile(optimizer='adam',loss='MSLE')
        return model
    
    def model_train(table_name, time_frame, split_rate, cycle, b_size=64):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'demo', 'stock_Django', 'stock_model')
        os.makedirs(model_dir, exist_ok=True)
        
        stock_list = stock_cost_AI.load_stock_number('stock_table_tw')
        for i in range(len(stock_list)):
            try:
                stock_num = str(stock_list.iat[i])
                cost_data, Date_data = stock_cost_AI.load_data_c(table_name, stock_num)
                data = stock_cost_AI.data_index(cost_data)
                data_copy = data.copy()
                
                # 呼叫非資料洩露版
                X_train, Y_train, X_test, Y_test, scaler = stock_cost_AI.data_preprocesing(
                    data_copy, time_frame, split_rate, columns=data_copy.columns.get_loc('Close')
                )
                
                model = stock_cost_AI.model_install(input_L=time_frame, input_d=len(data_copy.columns))
                model.fit(X_train, Y_train, epochs=cycle, batch_size=b_size, validation_data=(X_test, Y_test))
                
                # 儲存權重與 scaler
                weights_path = os.path.join(model_dir, f'stock_model_weights_{stock_num}.h5')
                model.save_weights(weights_path)
                
                scaler_path = os.path.join(model_dir, f'stock_scaler_{stock_num}.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            except Exception as e:
                print(f"Error training stock {stock_list.iat[i]}: {e}")
                pass

    def de_normalize(Ori_data, norm_value, de_nor_column):
        original_value = Ori_data.iloc[:,de_nor_column].to_numpy().reshape(-1,1)
        norm_value = norm_value.reshape(-1,1)
        scaler = MinMaxScaler().fit(original_value)
        de_norm_value = scaler.inverse_transform(norm_value)
        return de_norm_value
    
    def cost_plt(Pred_data, real_data, stock_name):
        plt.figure(figsize=(12,6))
        plt.plot(real_data, color = 'red', label = 'Real Stock Close Price')
        plt.plot(Pred_data, color = 'blue', label = 'Predicted Stock Close Price')
        plt.title(str(stock_name) + ' ' + 'Stock Price Prediction')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price TWD ($)', fontsize=18)
        plt.legend()
        plt.show()
        
    def kline_MA(data, pred_days):
        buf = BytesIO()
        sma_5 = mpf.make_addplot(talib.SMA(data["Close"], 5), color = 'cyan', label = 'sma_5')
        sma_20 = mpf.make_addplot(talib.SMA(data["Close"], 20), color = 'orange', label = 'sma_20')
        sma_60 = mpf.make_addplot(talib.SMA(data["Close"], 60), color = 'purple',label = 'sma_60')
        Candle = mpf.make_addplot(data.iloc[:(len(data)-pred_days)], color = 'blue', type = 'candle')
        mpf.plot(data, type = 'line', style = 'yahoo', addplot = [sma_5, sma_20, sma_60, Candle],
                 volume = True, volume_panel = 1, savefig = buf)
        buf.seek(0)
        return buf
    
    def pred_data(data, stock_number):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'demo', 'stock_Django', 'stock_model')
        
        data_index = stock_cost_AI.data_index(data)
        data_copy = data_index.copy()
        
        if len(data_copy) < 5:
            raise ValueError("Data length is less than 5, cannot predict.")
            
        # 取最後 5 天資料進行預測
        last_5_days = data_copy.iloc[-5:].copy()
        
        # 載入 scaler
        scaler_path = os.path.join(model_dir, f'stock_scaler_{stock_number}.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            scaled_input = scaler.transform(last_5_days[data_copy.columns])
        else:
            # Fallback: 若無儲存的 scaler，則使用當前整個歷史資料區間 fit
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_copy[data_copy.columns])
            scaled_input = scaler.transform(last_5_days[data_copy.columns])
            
        # Reshape 成 (1, 5, number_features)
        X_input = np.reshape(scaled_input, (1, 5, len(data_copy.columns)))
        
        # 載入模型權重
        model = stock_cost_AI.model_install(5, len(data_copy.columns))
        weights_path = os.path.join(model_dir, f'stock_model_weights_{stock_number}.h5')
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            
        # 預測
        prediction = model.predict(X_input, verbose=0)
        
        # 逆正規化
        close_idx = data_copy.columns.get_loc('Close')
        prediction_de = stock_cost_AI.de_normalize(data_index, prediction, close_idx)
        return prediction_de[0][0]
        
    def pred_cost(stock_number, pred_days):
        cost_data, cost_Date = stock_cost_AI.load_data_c('stock_cost', stock_number)
        # 確保 Date 是 datetime 格式
        cost_Date_parsed = pd.to_datetime(cost_Date['Date'])
        start_date = cost_Date_parsed.iat[-1] + pd.Timedelta(+1, 'D')
        
        rolling_df = cost_data.copy()
        rolling_df.index = cost_Date_parsed
        
        new_data = []
        current_date = start_date
        predicted_dates = []
        
        for _ in range(pred_days):
            # 滾動預測下一交易日收盤價
            pred_val = stock_cost_AI.pred_data(rolling_df, stock_number)
            new_data.append(pred_val)
            
            # 計算下一個工作日 (跳過週末)
            while current_date.dayofweek >= 5:
                current_date = current_date + pd.Timedelta(+1, 'D')
            predicted_dates.append(current_date)
            
            # 將預測結果追加至 rolling_df
            last_row = rolling_df.iloc[-1].copy()
            last_row['Close'] = pred_val
            last_row['Open'] = pred_val
            last_row['High'] = pred_val
            last_row['Low'] = pred_val
            
            new_row_df = pd.DataFrame([last_row], index=[current_date])
            rolling_df = pd.concat([rolling_df, new_row_df])
            
            current_date = current_date + pd.Timedelta(+1, 'D')
            
        data_concat = pd.DataFrame(new_data, columns=['Close'], index=predicted_dates)
        data_concat.index.name = 'Date'
        # 轉成 DataFrame 時，把 index 轉回 Date 欄位以與原本介面相容
        data_concat = data_concat.reset_index()
        data_concat.set_index('Date', inplace=True)
        return data_concat

            
    def last_investor_H_T(data, amount):
        investor_last_time = data.sort_values(by = '日期', ascending = True).iloc[-1].at['日期']
        mask_last_date = (data['日期']==investor_last_time)
        investor_lastday_data = data[mask_last_date]
        investor_lastday_data.sort_values(by = '三大法人買賣超股數', ascending = False, inplace = True)
        investor_head= investor_lastday_data.head(int(amount))
        investor_head.set_index('日期', inplace = True)
        investor_tail= investor_lastday_data.tail(int(amount))
        investor_tail.set_index('日期', inplace = True)
        return investor_head,investor_tail
    
    def get_investor(data,stock,days):
        mask = (data['number']==str(stock))
        data2 = data[mask]
        data_sort = data2.sort_values(by = '日期', ascending = True)
        investor_tail = data_sort.tail(days)
        investor_tail.set_index('日期', inplace = True)
        return investor_tail
    
    def investor_plt(data, axes = 'NA'):
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
    
    def investor_TOP_plt(data, amount, days):
        load_data_H,load_data_T = stock_cost_AI.last_investor_H_T(data, amount)
        H_number = load_data_H['number']
        T_number = load_data_T['number']
        matplotlib.rc('font', family='Noto Sans SC')
        fig, axes = plt.subplots(nrows = 2, ncols = amount, sharex = True, sharey = True)
        for i in range(len(H_number)):
            H_data = stock_cost_AI.get_investor(data, H_number.iloc[i],days)
            stock_cost_AI.investor_plt(H_data, axes = axes[0,i])
            if i == 0 :
                handles, labels = axes[0,i].get_legend_handles_labels()
        for i in range(len(T_number)):
            T_data = stock_cost_AI.get_investor(data, T_number.iloc[i],days)
            stock_cost_AI.investor_plt(T_data, axes = axes[1,i])
        fig.suptitle('investor ' + 'TOP' + str(amount) + ' Increase/decrease')
        fig.legend(handles, labels, loc = 'outside upper right', fontsize = 7)
        fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
        return fig
    
    def transfer_numeric(data):
        cols = data.columns
        for col in cols:
            if col =='日期':
                data['日期'] = pd.DatetimeIndex(data['日期']).to_period('D')
            elif col =='number':
                pass
            elif col =='證券名稱':
                pass
            else :
                #批量去除千位符
                data[col] = data[col].str.replace(',','')
                data[col] = data[col].replace(to_replace = '', value = 0)
                data[col] = data[col].replace(to_replace = 'None', value = 0)
                data[col] = data[col].fillna(0)
                #字串轉整數
                data[col] = pd.to_numeric(data[col], errors = 'coerce' ,downcast = 'signed')
        return data
    
    def date_create(start_date, days):
        bdate_range = pd.bdate_range(start = str(start_date), periods = days, freq = 'B', name = 'Date')
        date = pd.DataFrame(bdate_range)
        return date
        
'''test code'''
'''stock_cost_AI.model_train(table_name = 'stock_cost',time_frame = 5,
                          split_rate = 0.8, cycle = 100, b_size = 512)

new_data = stock_cost_AI.pred_cost('2330', 5)
data, date = stock_cost_AI.load_data_c('stock_cost', '2330')
start_date = date['Date'].iat[-1] + pd.Timedelta(+1,'D')
r_date_range = pd.bdate_range(start = start_date, periods = 5, freq = 'B', name = 'Date')
first_date = r_date_range[0].strftime('%Y-%m-%d')
last_date = (r_date_range[-1] + pd.Timedelta(+1,'D')).strftime('%Y-%m-%d')
real_data = yf.download('2330.TW', start = first_date, end = last_date)

realitive_error = np.average(np.abs(real_data['Close']-new_data['Close'])/real_data['Close'], axis = 0)
print(realitive_error)
plot_data = stock_cost_AI.cost_plt(new_data['Close'], real_data['Close'], '2330')'''
