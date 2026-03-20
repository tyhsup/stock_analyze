import mysql.connector,time
import pandas as pd
import mySQL_OP
import yfinance as yf
import datetime
from datetime import datetime
from datetime import timedelta


class stock_cost:
    
    def stock_TW(data):
        data_tw = []
        for i in range (data.shape[0]):
            data_a = data['有價證卷代號'] + '.TW'
            data_tw.append(data_a[i])
        return data_tw
    
    def cost_All_DL():
        SQL = mySQL_OP.OP_Fun()
        number = list(SQL.get_cost_data(table_name = 'stock_table_tw').loc[:,'有價證卷代號'])
        for i in range (len(number)):
            number_TW = number[i] + '.TW'
            try:
                stock_DL = yf.download(number_TW, period = 'max', interval = '1d', auto_adjust = True)
            except Exception as e:
                print(e)
                try:
                    stock_DL = yf.download(number_TW, period = '5d', interval = '1d', auto_adjust = True)
                except Exception as e:
                    print(e)
                    SQL.delete_NaN_number(stock_number = number[i])
                    print(f'Delete stock number {number [i]}')
                    continue
            stock_DL.drop_duplicates()
            stock_DL.dropna()
            stock_DL.columns = stock_DL.columns.droplevel(1)
            stock_DL.insert(0, column = 'number', value = number[i])
            stock_DL.reset_index(inplace = True)
            SQL.upload_all(stock_DL, 'stock_tw_analyse', 'stock_cost_2')
    
    def update_all_cost():
        SQL = mySQL_OP.OP_Fun()
        number = list(SQL.get_cost_data(table_name = 'stock_table_tw').loc[:,'有價證卷代號'])
        for i in range (len(number)):
            number_TW = number[i] + '.TW'
            try:
                get_old_data = SQL.get_cost_data('stock_cost', stock_number = number[i])
                date = (get_old_data['Date'].iat[-1] + timedelta(days = 1)).strftime('%Y-%m-%d')
                stock_DL = yf.download(number_TW, start = date, period = 'max', interval = '1d', auto_adjust = True)
                time.sleep(1)
                print(f'upload stock {number[i]} done')
            except Exception as e:
                try:
                    print(f'upload stock {number[i]} fail, try to change period...')
                    get_old_data = SQL.get_cost_data('stock_cost', stock_number = number[i])
                    date = (get_old_data['Date'].iat[-1] + timedelta(days = 1)).strftime('%Y-%m-%d')
                    stock_DL = yf.download(number_TW, start = date, period = '5d', interval = '1d', auto_adjust = True)
                    time.sleep(1)
                    print(f'upload stock {number[i]} done')
                except Exception as e:
                    print(e)
                    SQL.delete_NaN_number(stock_number = number[i])
                    print(f'Download fail and Delete stock number {number [i]}')
                    continue
            stock_DL.drop_duplicates()
            stock_DL.dropna()
            stock_DL.columns = stock_DL.columns.droplevel(1)
            stock_DL.insert(0, column = 'number', value = number[i])
            stock_DL.reset_index(inplace = True)
            SQL.upload_all(stock_DL, 'stock_tw_analyse', 'stock_cost')

stock_cost.update_all_cost()
#stock_cost.cost_All_DL()



