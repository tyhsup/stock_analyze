import pandas as pd
import mySQL_OP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers,models
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.wrappers.scikit_learn import KerasRegressor


class stock_cost_AI:
    
    def load_data(table):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.sel_cost_data(table_name = table)
        cost_data = dl_cost_table[['Open','Close','High','Low','Volume']]
        Date_data = dl_cost_table['Date']
        return cost_data, Date_data
    
    def normalize(data, Min, Max):
        scaler = MinMaxScaler(feature_range= (Min, Max)).fit(data[['Open','Close','High','Low','Volume']])
        data[['Open','Close','High','Low','Volume']] = scaler.transform(data[['Open','Close','High','Low','Volume']])
        return data
    
    def data_preprocesing(data, time_frame,split_rate):
        number_features = len(data.columns)
        data = data.to_numpy()
        result = []
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
        
    
#training the model
cost_AI = stock_cost_AI
cost_data,Date_data = cost_AI.load_data('stock_cost')
data_normalize = cost_AI.normalize(cost_data, Min = 0, Max = 1)
X_train,Y_train,X_test,Y_test = cost_AI.data_preprocesing(data_normalize, time_frame = 5,split_rate = 0.8)
#model install & train
model = cost_AI.model_install(5,5)
model.fit(X_train, Y_train, epochs = 100, batch_size = 512, validation_data = (X_test,Y_test))
prediction = model.predict(X_test)
de_normal_pred = cost_AI.de_normalize(cost_data, prediction)
de_normal_real = cost_AI.de_normalize(cost_data, Y_test)
cost_AI.cost_plt(de_normal_pred,de_normal_real)


