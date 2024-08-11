import mysql.connector
import pymysql
from sqlalchemy import create_engine
import pandas as pd

class OP_Fun:
    def cnn(self):
        connection = mysql.connector.connect(host = 'localhost',
                                     port = 'port number',
                                     user = 'user name',
                                     password = 'password',
                                     database = 'database name')
        return connection
    
    def C_engine():
        engine = create_engine('mysql+pymysql://user:password@localhost:port/database name')
        return engine
    
    def upload_all(self, data, database, table_name):
        data.to_sql(name = str(table_name), con = self.C_engine, schema = str(database), if_exists = 'append')
        
    def sel_cost_data(self, table_name ='NA', columns_name = 'NA', stock_number = 'NA', *args, **kwargs):
        connection = self.connection()
        cursor = connection.cursor()
        print('請輸入要查詢的股票代碼')
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
        records = pd.DataFrame(cursor.fetchall())
        cursor.close()
        connection.close()
        return records

        
        
    
    
    
    
    
