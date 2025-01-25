import mysql.connector
import pandas as pd
import mySQL_OP
import yfinance as yf
import datetime
from datetime import datetime
from datetime import timedelta


class stock_cost:
    
    def stock_TW(self, data):
        data_tw = []
        for i in range (data.shape[0]):
            data_a = data['有價證卷代號'] + '.TW'
            data_tw.append(data_a[i])
        return data_tw
    
    def cost_All_DL(self, data):
        stock_data = pd.DataFrame()
        stock_DL_fail = []
        for i in range (len(data)):
            stock_number = []
            try:
                stock_DL = yf.download(data[i], period = 'max', interval = '1d', auto_adjust = True)
            except Exception as e:
                print(e)
                try:
                    stock_DL = yf.download(data[i], period = '1d', interval = '1d', auto_adjust = True)
                except Exception as e:
                    print(e)
                    stock_DL_fail.append(data[i]) 
            stock_DL.drop_duplicates()
            stock_DL.dropna()
            for j in range (stock_DL.shape[0]):
                str1 = data[i].split('.')
                stock_number.append(str1[0])
            stock_DL.insert(0, column = 'number', value = stock_number)
            stock_DL.reset_index(inplace = True)
            stock_data = pd.concat([stock_data,stock_DL], ignore_index = False)
        return stock_data
    
    def update_all_DL(self, data):
        stock_data = pd.DataFrame()
        stock_DL_fail = []
        SQL = mySQL_OP.OP_Fun()
        get_old_data = SQL.get_cost_data('stock_cost')
        date = (get_old_data['Date'].iat[-1] + timedelta(days =1)).strftime('%Y-%m-%d')
        for i in range (len(data)):
            stock_number = []
            try:
                stock_DL = yf.download(data[i], start = date, period = 'max', interval = '1d', auto_adjust = True)
            except Exception as e:
                try:
                    stock_DL = yf.download(data[i], start = date, period = '1d', interval = '1d', auto_adjust = True)
                except Exception as e:
                    print(e)
                    stock_DL_fail.append(data[i])
            stock_DL.drop_duplicates()
            stock_DL.dropna()
            for j in range (stock_DL.shape[0]):
                str1 = data[i].split('.')
                stock_number.append(str1[0])
            stock_DL.insert(0, column = 'number', value = stock_number)
            stock_DL.reset_index(inplace = True)
            stock_data = pd.concat([stock_data,stock_DL], ignore_index = False)
        return stock_data

SQL = mySQL_OP.OP_Fun()
cost = stock_cost()
stock_number = SQL.get_cost_data(table_name = 'stock_table_tw')
stock_tw = cost.stock_TW(stock_number)
#stock_data = cost.update_all_DL(stock_tw)
stock_data = stock_cost().cost_All_DL(stock_tw)
#SQL.upload_all(stock_data, 'stock_tw_analyse', 'stock_cost')
