import mysql.connector
import pymysql
from sqlalchemy import create_engine, text, types
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class OP_Fun:
    def __init__(self):
        self.db_config = {
            'host': 'your host',
            'port': '3306',
            'user': 'root',
            'password': 'please input password', # 請確認您的密碼
            'database': 'your database name'
        }
        self.engine = create_engine(
            f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}",
            pool_size=5, max_overflow=10, pool_recycle=3600
        )

    def upload_all(self, data, table_name):
        """原本的通用上傳函式，供 stock_cost 使用"""
        if data.empty:
            return

        dtype_dict = {}
        if 'number' in data.columns:
            dtype_dict['number'] = types.VARCHAR(20)
        
        date_col = 'Date' if 'Date' in data.columns else ('日期' if '日期' in data.columns else None)
        if date_col:
            dtype_dict[date_col] = types.DateTime() if 'Date' in data.columns else types.VARCHAR(20)

        temp_table = f"{table_name}_temp"
        try:
            data.to_sql(name=temp_table, con=self.engine, if_exists='replace', index=False, dtype=dtype_dict, method='multi')
            with self.engine.begin() as conn:
                check_table = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'")).fetchone()
                if not check_table:
                    conn.execute(text(f"CREATE TABLE {table_name} LIKE {temp_table}"))
                    if date_col and 'number' in data.columns:
                        index_name = f"idx_{table_name}_unique"
                        conn.execute(text(f"ALTER TABLE {table_name} ADD UNIQUE INDEX {index_name} ({date_col}, number)"))

                conn.execute(text(f"INSERT IGNORE INTO {table_name} SELECT * FROM {temp_table}"))
                conn.execute(text(f"DROP TABLE {temp_table}"))
            logger.info(f"成功更新 {table_name}")
        except Exception as e:
            logger.error(f"上傳至 {table_name} 失敗: {e}")
            raise

    def get_cost_data(self, table_name, columns_name='*', stock_number=None):
        """讀取資料函式"""
        try:
            if stock_number:
                query = f"SELECT {columns_name} FROM {table_name} WHERE number = :num"
                df = pd.read_sql(text(query), con=self.engine, params={'num': stock_number})
            else:
                query = f"SELECT {columns_name} FROM {table_name}"
                df = pd.read_sql(text(query), con=self.engine)
            return df
        except Exception as e:
            logger.error(f"讀取 {table_name} 失敗: {e}")
            return pd.DataFrame()
    
    def upload_investor_bulk(self, df, table_name='stock_investor'):
        if df.empty:
            return

        with self.engine.begin() as conn:
            # --- 新增：自動建表與索引檢查邏輯 ---
            check_table = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'")).fetchone()
            
            if not check_table:
                logger.info(f"偵測到資料表 {table_name} 不存在，正在執行初始化...")
                # 1. 先用 pandas 建立一個結構相同的表 (暫不寫入數據)
                # 我們利用 to_sql 建立一個空表，並定義好 number 欄位長度以利索引
                from sqlalchemy import types
                dtype_dict = {'number': types.VARCHAR(20), '日期': types.VARCHAR(20)}
                df.head(0).to_sql(name=table_name, con=self.engine, if_exists='replace', index=False, dtype=dtype_dict)
                
                # 2. 建立唯一索引 (日期 + number)，這是 UPSERT (ON DUPLICATE KEY) 運作的基礎
                index_name = f"idx_{table_name}_unique"
                conn.execute(text(f"ALTER TABLE `{table_name}` ADD UNIQUE INDEX `{index_name}` (`日期`, `number`)"))
                logger.info(f"已成功建立資料表 {table_name} 與唯一索引。")

            # --- 原有的批量寫入邏輯 ---
            cols = ", ".join([f"`{c}`" for c in df.columns])
            placeholders = ", ".join(["%s"] * len(df.columns))
            update_cols = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in df.columns if c not in ['日期', 'number']])
        
            sql = f"""
                INSERT INTO `{table_name}` ({cols}) 
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_cols}
            """
            data_list = [tuple(x) for x in df.values]
            
            try:
                conn.exec_driver_sql(sql, data_list)
                logger.info(f"三大法人批量寫入完成：{table_name} (共 {len(df)} 筆)")
            except Exception as e:
                logger.error(f"三大法人批量寫入 SQL 執行失敗: {e}")
                raise e
