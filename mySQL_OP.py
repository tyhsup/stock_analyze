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
        
    def sel_columns(self, table_name, columns_name):
        connection = self.cnn()
        cursor = connection.cursor()
        cursor.execute('SELECT' + '`' + columns_name + '`' + 'FROM' + '`' + table_name + '`;')
        records = pd.DataFrame(cursor.fetchall())
        cursor.close()
        connection.close()
        return records

        
        
    
    
    
    
    
