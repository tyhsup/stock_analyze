import mysql.connector
import pandas as pd
import mySQL_OP
import yfinance as yf


class stock_cost():
    
    def stock_TW(self, data):
        data_tw = []
        for i in range (data.shape[0]):
            data_a = data.iloc[i] + '.TW'
            data_tw.append(data_a[0])
        return data_tw
    
    def cost_All_DL(self, data):
        stock_data = pd.DataFrame()
        stock_DL_fail = []
        try:
            for i in range (len(data)):
                stock_number = []
                stock_DL = yf.download(data[i], period = 'max', interval = '1d', auto_adjust = True)
                stock_DL.drop_duplicates()
                stock_DL.dropna()
                for j in range (stock_DL.shape[0]):
                    str1 = data[i].split('.')
                    stock_number.append(str1[0])
                stock_DL.insert(0, column = 'number', value = stock_number)
                stock_data = pd.concat([stock_data,stock_DL], ignore_index = False)
        except:
            stock_DL_fail.append(data[i])
        return stock_data

stock = mySQL_OP.OP_Fun()
stock_number = stock.sel_columns(table_name = 'stock_table_tw', columns_name = '有價證卷代號')
stock_tw = stock_cost().stock_TW(stock_number)
stock_vol= stock_cost().cost_All_DL(stock_tw)
stock.upload_all(stock_vol, 'stock_tw_analyse', 'stock_cost')