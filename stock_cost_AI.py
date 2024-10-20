import pandas as pd
from stock_Django import mySQL_OP
#import mySQL_OP
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
import gc


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
        dl_cost_table = SQL_OP.sel_table_data(table_name = table)
        return dl_cost_table
    
    def load_stock_number(table):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.sel_table_data(table_name = table)
        stock_number = dl_cost_table['number'].drop_duplicates().to_list()
        return stock_number
           
    def normalize(data, Min, Max):
        columns_name = data.columns.to_list()
        scaler = MinMaxScaler(feature_range= (Min, Max)).fit(data[columns_name])
        data[columns_name] = scaler.transform(data[columns_name])
        return data
    
    def data_preprocesing(data, time_frame,split_rate):
        number_features = len(data.columns)
        data = data.to_numpy()
        result = []
        if len(data) > 5:
            for i in range(len(data)-(time_frame+1)):
                result.append(data[i: i + (time_frame+1)])
            result = np.array(result)
            #取train data & test data 比例
            data_split = int(result.shape[0]*split_rate)
            X_train = result[:data_split, :-1]
            Y_train = result[:data_split, -1][:,3]#訓練資料中取最後一筆收盤價作為答案
            X_test = result[data_split:, :-1]
            Y_test = result[data_split:, -1][:,3]
            #reshape data
            X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],number_features))
            X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],number_features))
            return X_train,Y_train,X_test,Y_test
        else:
            pass
    
    def data_index(data):
        close_data = data['Close']
        close_SMA_5 = talib.SMA(data['Close'],5)
        close_SMA_10 = talib.SMA(data['Close'],10)
        close_SMA_20 = talib.SMA(data['Close'],20)
        close_RSI_14 = talib.RSI(data['Close'],14)
        data_concat = pd.concat([close_data,close_SMA_5,close_SMA_10,close_SMA_20,close_RSI_14], axis = 1)
        data_concat.columns = ['Close','SMA_5','SMA_10','SMA_20','RSI_14']
        data_concat = data_concat.dropna()
        data_concat = data_concat.drop_duplicates()
        data_concat = data_concat.reset_index(drop = True)
        return data_concat
    
    def model_install(input_L, input_d):
        model = tf.keras.Sequential()
        model.add(LSTM(units = 512, return_sequences = True, input_shape = (input_L,input_d)))
        model.add(Dropout(0.4))
        model.add(LSTM(units = 256, return_sequences = True))
        model.add(Dropout(0.3))
        model.add(LSTM(units = 128, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 64))
        model.add(Dropout(0.1))
        model.add(Dense(units = 1, activation='elu'))
        model.compile(optimizer='adam',loss='MSLE')
        return model
    
    def model_train(table_name,time_frame,split_rate, input_L, input_d, cycle, b_size):
        stock_list = stock_cost_AI.load_stock_number(table_name)
        model = stock_cost_AI.model_install(input_L, input_d)
        for i in range(len(stock_list)):
            try:
                cost_data,Date_data = stock_cost_AI.load_data_c(table_name, str(stock_list[i]))
                data = stock_cost_AI.data_index(cost_data)
                data_normalize = stock_cost_AI.normalize(data, Min = 0, Max = 1)
                X_train,Y_train,X_test,Y_test = stock_cost_AI.data_preprocesing(data_normalize, time_frame,split_rate)
                model.fit(X_train, Y_train, epochs = cycle, batch_size = b_size, validation_data = (X_test,Y_test))
                print(i)
                gc.collect()
            except Exception as e:
                print(e)
                pass
        return model

    def de_normalize(Ori_data, norm_value):
        original_value = Ori_data['Close'].to_numpy().reshape(-1,1)
        norm_value = norm_value.reshape(-1,1)
        scaler = MinMaxScaler().fit(original_value)
        de_norm_value = scaler.inverse_transform(norm_value)
        return de_norm_value
    
    def cost_plt(Pred_data, real_data):
        plt.plot(real_data, color = 'red', label = 'Real Stock Close Price')
        plt.plot(Pred_data, color = 'blue', label = 'Predicted Stock Close Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
        
    def kline_MA(data, pred_days):
        sma_5 = mpf.make_addplot(talib.SMA(data["Close"], 5), color = 'cyan', label = 'sma_5')
        sma_20 = mpf.make_addplot(talib.SMA(data["Close"], 20), color = 'orange', label = 'sma_20')
        sma_60 = mpf.make_addplot(talib.SMA(data["Close"], 60), color = 'purple',label = 'sma_60')
        Candle = mpf.make_addplot(data.iloc[:(len(data)-pred_days)], color = 'blue', type = 'candle')
        fig,axes = mpf.plot(data, type = 'line', style = 'yahoo', addplot = [sma_5, sma_20, sma_60, Candle],
                 volume = True, volume_panel =1, block = False, returnfig = True)
        return fig,axes
    
    def pred_cost(stock_number, pred_days):
        cost_data,cost_Date = stock_cost_AI.load_data_c('stock_cost', stock_number)
#        cost_data,cost_Date = stock_cost_AI.load_data('stock_cost')
        new_date = []
        date_s = pd.Timestamp(cost_Date['Date'].to_list()[-1])
        model = load_model('model.h5')
        #model = tf.keras.Sequential()
        #model.load_weights('E:/Infinity/mydjango/demo/stock_Django/model.h5')
        cost_data_pred = cost_data.copy()
        for i in range(pred_days):
            cost_data_cal = cost_data_pred.copy()
            data_normalize = stock_cost_AI.normalize(cost_data_cal, Min = 0, Max = 1)
            X_train,Y_train,X_test,Y_test = stock_cost_AI.data_preprocesing(data_normalize, time_frame = 5,split_rate = 0.8)
            prediction = model.predict(X_test)
            de_normal_pred =stock_cost_AI.de_normalize(cost_data_pred, prediction).tolist()
            open_SMA_10 = talib.SMA(cost_data_pred['Open'],10).dropna().to_list()
            high_SMA_10 = talib.SMA(cost_data_pred['High'],10).dropna().to_list()
            low_SMA_10 = talib.SMA(cost_data_pred['Low'],10).dropna().to_list()
            volume_SMA_10 = talib.SMA(cost_data_pred['Volume'],10).dropna().to_list()
            new_data = pd.DataFrame({'Open':open_SMA_10[-1:],
                         'Close':de_normal_pred[-1],
                         'High':high_SMA_10[-1:],
                         'Low':low_SMA_10[-1:],
                         'Volume':volume_SMA_10[-1:]})
            cost_data_pred = pd.concat([cost_data_pred, new_data],ignore_index = True)
            new_date.append(date_s+pd.Timedelta(+1,'D'))
            date_s = date_s+pd.Timedelta(+1,'D')
        new_date = pd.DataFrame(new_date, columns =['Date'])
        new_date = pd.concat([cost_Date,new_date],ignore_index = True)
        cost_data_pred = pd.concat([new_date,cost_data_pred], axis = 1)
        cost_data_pred.index = pd.DatetimeIndex(cost_data_pred['Date'])
        return cost_data_pred
            
    def last_investor_H_T(amount):
        SQL_OP = mySQL_OP.OP_Fun()
        investor_all = SQL_OP.sel_table_data(table_name = 'stock_investor')
        investor_last_time = investor_all.sort_values(by = '日期', ascending = True).iloc[-1].at['日期']
        mask_last_date = (investor_all['日期']==investor_last_time)
        investor_lastday_data = investor_all[mask_last_date].sort_values(by = '三大法人買賣超股數', ascending = False)
        get_data_list = ['日期','number','外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數(自行買賣)',
                     '自營商買賣超股數(避險)']
        investor_head= investor_lastday_data.head(int(amount))[get_data_list]
        investor_tail= investor_lastday_data.tail(int(amount))[get_data_list]
        return investor_head,investor_tail
    
    def get_investor(stock,days):
        SQL_OP = mySQL_OP.OP_Fun()
        investor_data = SQL_OP.get_cost_data(table_name = 'stock_investor', stock_number = str(stock))
        investor_sort = investor_data.sort_values(by = '日期', ascending = True)
        get_data_list = ['日期','number','外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數(自行買賣)',
                     '自營商買賣超股數(避險)']
        investor_tail = investor_sort.tail(days)[get_data_list]
        return investor_tail
    
    def investor_plt(data):
        matplotlib.rc('font', family='Noto Sans SC')
        cols = data.columns
        for col in cols:
            if col =='日期':
                data['日期'] = pd.DatetimeIndex(data['日期']).to_period('D')
            elif col =='number':
                pass
            else :
                #批量去除千位符
                data[col] = data[col].str.replace(',','').astype(int)
        data = data.drop(['number'],axis=1)
        #繪製堆疊柱狀圖
        fig = data.plot(x = '日期',kind = 'bar',stacked = True, figsize = (12,6)).get_figure()
        return fig
