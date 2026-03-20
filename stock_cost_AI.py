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
    
    def data_preprocesing(data, time_frame,split_rate,columns):
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
            #訓練資料中取最後一筆收盤價作為答案
            Y_train = result[:data_split, -1][:,columns]
            X_test = result[data_split:, :-1]
            Y_test = result[data_split:, -1][:,columns]
            #reshape data
            X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],number_features))
            X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],number_features))
            return X_train,Y_train,X_test,Y_test
        else:
            pass
    
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
    
    def model_train(table_name,time_frame,split_rate, cycle, b_size):
        stock_list = stock_cost_AI.load_stock_number('stock_table_tw')
        for i in range(len(stock_list)):
            try:
                cost_data,Date_data = stock_cost_AI.load_data_c(table_name, str(stock_list.iat[i]))
                data = stock_cost_AI.data_index(cost_data)
                data_copy = data.copy()
                data_normalize = stock_cost_AI.normalize(data_copy, Min = 0, Max = 1)
                X_train,Y_train,X_test,Y_test = stock_cost_AI.data_preprocesing(data_normalize, time_frame,split_rate,
                                                                                columns = data_normalize.columns.get_loc('Close'))
                model = stock_cost_AI.model_install(input_L = time_frame,input_d = len(data_normalize.columns))
                model.fit(X_train, Y_train, epochs = cycle, batch_size = b_size, validation_data = (X_test,Y_test))
                prediction = model.predict(X_test)
                prediction_de = stock_cost_AI.de_normalize(data, prediction,data.columns.get_loc('Close'))
                model.save_weights('E:/Infinity/mydjango/demo/stock_Django/stock_model/stock_model_weights_' +
                                   str(stock_list.iat[i]) + '.h5')
            except Exception as e:
                print(e)
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
        data_index = stock_cost_AI.data_index(data)
        data_copy = data_index.copy()
        model = stock_cost_AI.model_install(5,len(data_copy.columns))
        model.load_weights('E:/Infinity/mydjango/demo/stock_Django/stock_model/stock_model_weights_' +
                           str(stock_number) + '.h5')
        data_normalize = stock_cost_AI.normalize(data_copy, Min = 0, Max = 1)
        X_train,Y_train,X_test,Y_test = stock_cost_AI.data_preprocesing(data_normalize, 5,0.8,
                                                                        data_normalize.columns.get_loc('Close'))
        model.fit(X_train, Y_train, epochs = 100, batch_size = 512, validation_data = (X_test,Y_test))
        prediction = model.predict(X_test)
        prediction_de = stock_cost_AI.de_normalize(data_index, prediction,data_index.columns.get_loc('Close'))
        pre_data = prediction_de[-1]
        model.save_weights('E:/Infinity/mydjango/demo/stock_Django/stock_model/stock_model_weights_' +
                           str(stock_number) + '.h5')
        return pre_data
    
    def pred_cost(stock_number, pred_days):
        cost_data,cost_Date = stock_cost_AI.load_data_c('stock_cost', stock_number)
        start_date = cost_Date['Date'].iat[-1] + pd.Timedelta(+1,'D')
        new_data = []
        for i in range(pred_days) :
            pred_data= stock_cost_AI.pred_data(cost_data, stock_number)
            new_data.extend(pred_data)
        new_date = stock_cost_AI.date_create(start_date, pred_days)
        new_data = pd.DataFrame(new_data,columns = ['Close'])
        data_concat = pd.concat([new_date,new_data], axis = 1)
        data_concat.set_index('Date', inplace = True)
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
