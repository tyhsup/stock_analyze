import mysql.connector
import pymysql
from sqlalchemy import create_engine
import pandas as pd

class OP_Fun:
    #設定connection 物件
    def connection(self):
        connection = mysql.connector.connect(host = '',
                                     port = '',
                                     user = '',
                                     password = '',
                                     database = '')
        return connection
    #初始化engine
    def C_engine(self):
        engine = create_engine('')
        return engine
    #上傳資料
    def upload_all(self, data, database, table_name):
        data.to_sql(name = str(table_name), con = self.C_engine(), schema = str(database), if_exists = 'append', index = False)
    #手動輸入要從mySQL 資料庫提取的資料
    def sel_cost_data(self, table_name ='NA', columns_name = 'NA', stock_number = 'NA', *args, **kwargs):
        connection = self.connection()
        cursor = connection.cursor()
        print('請輸入SQL table')
        table_name = input('SQL table:')
        print('請輸入要查看的股票代碼')
        stock_number = input('股票代碼:')
        if table_name == 'NA':
            print('please key in table name')
        else :
            if columns_name == 'NA' and stock_number == 'NA' :
                fun = f'SELECT * FROM {table_name} ;'
                cursor.execute(fun)
            elif columns_name == 'NA':
                fun = f'SELECT * FROM {table_name} WHERE number = %s;'
                number = (stock_number,)
                cursor.execute(fun, number)
            elif stock_number == 'NA' :
                fun = f'SELECT {columns_name} FROM {table_name} ;'
                cursor.execute(fun)
            else :
                fun = f'SELECT {columns_name} FROM {table_name} WHERE number = %s;'
                number = (stock_number,)
                cursor.execute(fun, number)
        cols = cursor.description
        col = []
        for i in cols:
            col.append(i[0])
        records = pd.DataFrame(cursor.fetchall(),columns = col)
        records.drop(records.columns[[-1]], axis =1, inplace = True)
        cursor.close()
        connection.close()
        return records
    #從mySQL資料庫提取資料
    def get_cost_data(self, table_name ='NA', columns_name = 'NA', stock_number = 'NA', *args, **kwargs):
        connection = self.connection()
        cursor = connection.cursor()
        if table_name == 'NA':
            print('please key in table name')
        else :
            if columns_name == 'NA' and stock_number == 'NA' :
                fun = f'SELECT * FROM {table_name} ;'
                cursor.execute(fun)
            elif columns_name == 'NA':
                fun = f'SELECT * FROM {table_name} WHERE number = %s;'
                number = (stock_number,)
                cursor.execute(fun, number)
            elif stock_number == 'NA' :
                fun = f'SELECT {columns_name} FROM {table_name} ;'
                cursor.execute(fun)
            else :
                fun = f'SELECT {columns_name} FROM {table_name} WHERE number = %s;'
                number = (stock_number,)
                cursor.execute(fun, number)
        cols = cursor.description
        col = []
        for i in cols:
            col.append(i[0])
        records = pd.DataFrame(cursor.fetchall(),columns = col)
        cursor.close()
        connection.close()
        return records

    #刪除無法download的股票代碼
    def delete_NaN_number(self, table_name ='stock_table_tw', columns_name = '有價證卷代號', stock_number = 'NA', *args, **kwargs):
        connection = self.connection()
        cursor = connection.cursor()
        delete_fun = f'DELETE FROM {table_name} WHERE {columns_name} = %s ;'
        number = (stock_number,)
        cursor.execute(delete_fun,number)
        connection.commit()
        cursor.close()
        connection.close()

        
        
    
    
    
    
    
