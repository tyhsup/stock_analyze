import mysql.connector
import pymysql
from sqlalchemy import create_engine
import pandas as pd

class OP_Fun:
    def connection(self):
        connection = mysql.connector.connect(host = '',
                                     port = '',
                                     user = '',
                                     password = '',
                                     database = '')
        return connection
    
    def C_engine(self):
        engine = create_engine('')
        return engine
    
    def upload_all(self, data, database, table_name):
        data.to_sql(name = str(table_name), con = self.C_engine(), schema = str(database), if_exists = 'append', index = False)
        
    def sel_cost_data(self, table_name ='NA', columns_name = 'NA', stock_number = 'NA', *args, **kwargs):
        connection = self.connection()
        cursor = connection.cursor()
        print('請輸入要分析的股票代碼')
        stock_number = input('股票代碼:')
        if table_name == 'NA':
            print('no select table')
        else :
            if columns_name == 'NA' :
                cursor.execute('SELECT * FROM' + '`' + table_name + '`' + 'WHERE number ='
                               + stock_number + ';')
            else :
                cursor.execute('SELECT' + '`' + columns_name + '`' + 'FROM' + '`' + table_name + '`' + 'WHERE number ='
                               + stock_number + ';')
        cols = cursor.description
        col = []
        for i in cols:
            col.append(i[0])
        records = pd.DataFrame(cursor.fetchall(),columns = col)
        records.drop(records.columns[[-1]], axis =1, inplace = True)
        cursor.close()
        connection.close()
        return records
    
    def get_cost_data(self, table_name ='NA', columns_name = 'NA', stock_number = 'NA', *args, **kwargs):
        connection = self.connection()
        cursor = connection.cursor()
        if table_name == 'NA':
            print('please key in table name')
        else :
            if columns_name == 'NA' and stock_number == 'NA' :
                cursor.execute('SELECT * FROM' + '`' + table_name + '`;')
            elif columns_name == 'NA':
                cursor.execute('SELECT * FROM' + '`' + table_name + '`' + 'WHERE number ='
                            + "'" + stock_number + "'" + ';')
            elif stock_number == 'NA' :
                cursor.execute('SELECT' + '`' + columns_name + '`' + 'FROM' + '`' + table_name + '`' + ';')
            else :
                cursor.execute('SELECT' + '`' + columns_name + '`' + 'FROM' + '`' + table_name + '`' + 'WHERE number ='
                               + "'" + stock_number + "'" + ';')
        cols = cursor.description
        col = []
        for i in cols:
            col.append(i[0])
        records = pd.DataFrame(cursor.fetchall(),columns = col)
        #records.drop(records.columns[[-1]], axis =1, inplace = True)
        cursor.close()
        connection.close()
        return records

        
        
    
    
    
    
    
