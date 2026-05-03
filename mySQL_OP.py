import mysql.connector
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, types
import pandas as pd
import logging

# 加載環境變數
load_dotenv(dotenv_path='demo/stock_Django/.env')

logger = logging.getLogger(__name__)

class OP_Fun:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '3306'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'stock_tw_analyse')
        }
        self.engine = create_engine(
            f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}",
            pool_size=5, max_overflow=10, pool_recycle=3600
        )

    def upload_all(self, data, table_name):
        """優化後的通用上傳函式，支援 ON DUPLICATE KEY UPDATE 與自動欄位遷移"""
        if data.empty:
            return

        # 1. 欄位名稱標準化 (小寫)
        data.columns = [c.lower() for c in data.columns]
        
        # 處理日期欄位名稱 (DataFrame 端)
        if 'date' not in data.columns and '日期' in data.columns:
            data.rename(columns={'日期': 'date'}, inplace=True)
            
        # 轉換日期格式
        if 'date' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            except:
                pass

        with self.engine.begin() as conn:
            # 2. 檢查表是否存在
            check_table = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'")).fetchone()
            if not check_table:
                logger.info(f"正在初始化資料表 {table_name}...")
                dtype_dict = {'number': types.VARCHAR(20), 'date': types.DATE()}
                data.head(0).to_sql(name=table_name, con=self.engine, if_exists='replace', index=False, dtype=dtype_dict)
                # 建立唯一索引
                index_name = f"idx_{table_name}_unique"
                conn.execute(text(f"ALTER TABLE `{table_name}` ADD UNIQUE INDEX `{index_name}` (`date`, `number`)"))
            else:
                # 3. 自動遷移：處理亂碼或中文欄位名稱 (使用 INFORMATION_SCHEMA 確保準確性)
                # 取得第一個欄位名稱
                query_info = text("""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = :t AND TABLE_SCHEMA = :s AND ORDINAL_POSITION = 1
                """)
                res = conn.execute(query_info, {'t': table_name, 's': self.db_config['database']}).fetchone()
                
                if res:
                    first_col = res[0]
                    # 如果第一個欄位不是 'date'，且不是 'number' (排除特殊情況)
                    # 或者第一個欄位包含非 ASCII 字元
                    is_not_ascii = not all(ord(c) < 128 for c in first_col)
                    if first_col != 'date' and (is_not_ascii or first_col == '日期'):
                        safe_col_name = first_col.encode('utf-8', errors='ignore').decode('utf-8')
                        logger.info(f"正在遷移 {table_name}: 將欄位 '{safe_col_name}' 重新命名為 'date'...")
                        try:
                            # 使用 RENAME COLUMN (MySQL 8.0+)
                            conn.execute(text(f"ALTER TABLE `{table_name}` RENAME COLUMN `{first_col}` TO `date`"))
                            # 確保類型正確
                            conn.execute(text(f"ALTER TABLE `{table_name}` MODIFY COLUMN `date` DATE"))
                            # 強制提交 DDL 變更
                            conn.commit()
                        except Exception as e:
                            logger.warning(f"遷移失敗: {e}")

            # 4. 再次檢查欄位一致性 (確保 SQL 欄位存在於資料庫中)
            # 這裡簡單處理，只選取資料庫中已有的欄位
            cols_after = conn.execute(text(f"SHOW COLUMNS FROM `{table_name}`")).fetchall()
            db_cols = [c[0].lower() for c in cols_after]
            final_data_cols = [c for c in data.columns if c in db_cols]
            data = data[final_data_cols]

            # 5. 構建 UPSERT SQL
            cols_sql = ", ".join([f"`{c}`" for c in data.columns])
            placeholders = ", ".join(["%s"] * len(data.columns))
            update_cols = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in data.columns if c not in ['date', 'number']])
            
            sql = f"""
                INSERT INTO `{table_name}` ({cols_sql}) 
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_cols}
            """
            
            # 6. 執行批量寫入
            data_list = [tuple(x) for x in data.values]
            try:
                conn.exec_driver_sql(sql, data_list)
                logger.info(f"成功更新 {table_name} (共 {len(data)} 筆)")
            except Exception as e:
                logger.error(f"批量寫入 {table_name} 失敗: {e}")
                raise e

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
        """統一調用 upload_all 實作三大法人寫入"""
        self.upload_all(df, table_name)


