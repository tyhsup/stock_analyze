import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from keras.layers import Dense, LSTM, Dropout
import os
import logging
from .stock_utils import StockUtils

logger = logging.getLogger(__name__)

class stock_cost_AI:
    def __init__(self):
        self.WINDOW_SIZE = 5
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
        self.num_classes = 6
        self.SQL_table = 'stock_cost_2'
    
    @staticmethod
    def model_install(input_L, input_d):
        model = tf.keras.Sequential()
        model.add(LSTM(units=512, return_sequences=True, input_shape=(input_L, input_d)))
        model.add(Dropout(0.4))
        model.add(LSTM(units=256, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='elu'))
        model.compile(optimizer='adam', loss='MSLE')
        return model
    
    def MultiModal_train(self, stock_number):
        OHLCV_data = StockUtils.load_data(self.SQL_table, stock_number)
        trend_data = StockUtils.trend_indicators(OHLCV_data)
        momentum_data = StockUtils.momentum_indicators(OHLCV_data)
        volatility_data = StockUtils.volatility_indicators(OHLCV_data)
        volume_data = StockUtils.volume_indicators(OHLCV_data)
        
        OHLCV_data.drop(['number'], axis=1, inplace=True)
        OHLCV_columns = OHLCV_data.columns
        Sentiment_data = StockUtils.Sentiment_indicators(stock_number, OHLCV_data[['Date', 'Close']])
        
        data_list = [OHLCV_data, trend_data, momentum_data, volatility_data, volume_data, Sentiment_data]
        date = OHLCV_data.loc[:, 'Date']
        merge_data = pd.DataFrame(date)
        for i in data_list:
            merge_data = pd.merge(merge_data, i, left_on='Date', right_on='Date', how='right')
        
        merge_data.dropna(inplace=True)
        # Choosing a specific table for return as in original code
        volatility_data2 = merge_data.loc[:, volatility_data.columns]
        return volatility_data2
    
    @staticmethod
    def model_train(table_name, time_frame, split_rate, cycle, b_size):
        stock_list = StockUtils.load_stock_number('stock_table_tw')
        for i in range(len(stock_list)):
            try:
                symbol = str(stock_list.iat[i])
                cost_data, Date_data = StockUtils.load_data_c(table_name, symbol)
                data = StockUtils.data_index(cost_data)
                data_normalize = StockUtils.normalize(data.copy(), Min=0, Max=1)
                
                prep = StockUtils.data_preprocesing(data_normalize, time_frame, split_rate, 
                                                   columns=data_normalize.columns.get_loc('Close'))
                if prep is None: continue
                X_train, Y_train, X_test, Y_test = prep
                
                model = stock_cost_AI.model_install(input_L=time_frame, input_d=len(data_normalize.columns))
                model.fit(X_train, Y_train, epochs=cycle, batch_size=b_size, validation_data=(X_test, Y_test))
                
                weights_path = f'E:/Infinity/mydjango/demo/stock_Django/stock_model/stock_model_weights_{symbol}.h5'
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                model.save_weights(weights_path)
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}")
                pass

    @staticmethod
    def pred_data(data, stock_number):
        data_index = StockUtils.data_index(data)
        data_copy = data_index.copy()
        model = stock_cost_AI.model_install(5, len(data_copy.columns))
        weights_path = f'E:/Infinity/mydjango/demo/stock_Django/stock_model/stock_model_weights_{stock_number}.h5'
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        
        data_normalize = StockUtils.normalize(data_copy, Min=0, Max=1)
        prep = StockUtils.data_preprocesing(data_normalize, 5, 0.98, data_normalize.columns.get_loc('Close'))
        if prep is None: return None
        X_train, Y_train, X_test, Y_test = prep
        
        model.fit(X_train, Y_train, epochs=100, batch_size=512, validation_data=(X_test, Y_test))
        prediction = model.predict(X_test)
        prediction_de = StockUtils.de_normalize(data_index, prediction, data_index.columns.get_loc('Close'))
        if len(prediction_de) == 0:
            return None
        return prediction_de[-1]
    
    @staticmethod
    def pred_cost(stock_number, pred_days, data_days=250):
        # Implementation based on views.py call which passes 3 args sometimes
        cost_data, cost_Date = StockUtils.load_data_c('stock_cost', stock_number)
        if cost_Date.empty:
            logger.warning(f"No cost data found for {stock_number}")
            return pd.DataFrame()
        start_date = cost_Date['Date'].iat[-1] + pd.Timedelta(+1, 'D')
        new_data = []
        for i in range(pred_days):
            p_data = stock_cost_AI.pred_data(cost_data, stock_number)
            if p_data is not None:
                new_data.extend(p_data)
            else:
                break
        
        new_date = StockUtils.date_create(start_date, len(new_data))
        new_data_df = pd.DataFrame(new_data, columns=['Close'])
        data_concat = pd.concat([new_date, new_data_df], axis=1)
        data_concat.set_index('Date', inplace=True)
        return data_concat

    @staticmethod
    def transfer_numeric(data):
        # Wrapper for StockUtils for backward compatibility if needed, 
        # but preferably update calls to StockUtils
        return StockUtils.transfer_numeric(data)

if __name__ == "__main__":
    test = stock_cost_AI()
    try:
        data = test.MultiModal_train('2330')
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")
