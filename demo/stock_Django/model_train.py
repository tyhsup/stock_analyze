import sys
import mySQL_OP
import pandas as pd
import numpy as np
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


class model_train:
    
    def __init__(self):
        self.Min = 0
        self.Max = 1
        self.time_frame = 5
        self.split_rate = 0.8
        self.input_L = 5
        self.input_d = 5
        self.cycle = 100
        self.b_size = 512
        
    
    def load_data_c(table,stock_name):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.get_cost_data(table_name = table,stock_number = stock_name)
        return dl_cost_table
           
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
    
    def model_first_learning(self,table_name, stock_number):
        model = model_train.model_install(self.input_L, self.input_d)
        try:
            cost_data = model_train.load_data_c(table_name, stock_number)
            data = model_train.data_index(cost_data)
            data_normalize = model_train.normalize(data, self.Min, self.Max)
            X_train,Y_train,X_test,Y_test = model_train.data_preprocesing(data_normalize, self.time_frame, self.split_rate)
            model.fit(X_train, Y_train, epochs = self.cycle, batch_size = self.b_size, validation_data = (X_test,Y_test))
            gc.collect()
            model.save('model2.h5')
        except Exception as e:
            print(e)
            pass

    def model_learning(self,table_name, stock_number):
        model = load_model('model2.h5')
        try:
            cost_data = model_train.load_data_c(table_name, stock_number)
            data = model_train.data_index(cost_data)
            data_normalize = model_train.normalize(data, self.Min, self.Max)
            X_train,Y_train,X_test,Y_test = model_train.data_preprocesing(data_normalize, self.time_frame, self.split_rate)
            model.fit(X_train, Y_train, epochs = self.cycle, batch_size = self.b_size, validation_data = (X_test,Y_test))
            gc.collect()
            model.save('model2.h5')
        except Exception as e:
            print(e)
            pass   
    
if __name__=='__main__':
    table_name = sys.argv[1]
    stock_number = sys.argv[2]
    model = model_train()
    model_learn = model.model_learning(table_name, stock_number)
    print('train model done')

#model = model_train()
#model_learn = model.model_lsirst_earning('stock_cost', '1101')