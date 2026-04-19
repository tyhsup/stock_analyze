import mysql.connector
import pymysql
from sqlalchemy import create_engine, text, types
import pandas as pd
import logging

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Use consistent .env loading relative to this file
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Try parent directory as fallback
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

class OP_Fun:
    # Use class-level variables to ensure singleton engine/pool
    _engine = None
    _db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '3306'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'stock_tw_analyse')
    }

    def __init__(self) -> None:
        if OP_Fun._engine is None:
            OP_Fun._engine = create_engine(
                f"mysql+pymysql://{self._db_config['user']}:{self._db_config['password']}@"
                f"{self._db_config['host']}:{self._db_config['port']}/{self._db_config['database']}?charset=utf8mb4",
                pool_size=10, 
                max_overflow=20, 
                pool_recycle=3600,
                pool_pre_ping=True
            )
        self.engine = OP_Fun._engine
        self.db_config = OP_Fun._db_config

    def upload_all(self, data: pd.DataFrame, table_name: str) -> None:
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

                # 採用 ON DUPLICATE KEY UPDATE 以符合標準並確保數據一致性
                cols_list = [f"`{c}`" for c in data.columns]
                # 排除唯一索引鍵 (Date, number)
                update_list = [f"{c}=VALUES({c})" for c in cols_list if c.replace("`", "") not in [date_col, 'number', '日期']]
                
                if update_list:
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(cols_list)}) SELECT {', '.join(cols_list)} FROM {temp_table} ON DUPLICATE KEY UPDATE {', '.join(update_list)}"
                else:
                    insert_sql = f"INSERT IGNORE INTO {table_name} SELECT * FROM {temp_table}"
                
                conn.execute(text(insert_sql))
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

    def _fix_investor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix garbled column names (Big5/UTF-8 mismatch) for stock_investor table."""
        if df.empty:
            return df
        
        # Mapping based on standard TWSE major institutional traders table format
        # Expanded to cover more variations and shifts
        mapping = {
            0: '日期',
            1: 'number',
            2: '證券名稱',
            3: '外陸資買進股數',
            4: '外陸資賣出股數',
            5: '外陸資買賣超股數(不含外資自營商)',
            6: '外資自營商買進股數',
            7: '外資自營商賣出股數',
            8: '外資自營商買賣超股數',
            9: '投信買進股數',
            10: '投信賣出股數',
            11: '投信買賣超股數',
            12: '自營商買賣超股數(合計)',
            13: '自營商買進股數(自行買賣)',
            14: '自營商賣出股數(自行買賣)',
            15: '自營商買賣超股數(自行買賣)',
            16: '自營商買進股數(避險)',
            17: '自營商賣出股數(避險)',
            18: '自營商買賣超股數(避險)',
            19: '三大法人買賣超股數'
        }
        
        # Apply mapping to existing columns safely
        new_cols = list(df.columns)
        for idx, name in mapping.items():
            if idx < len(new_cols):
                # Only overwrite if the current column name looks suspicious (mangled or generic)
                curr = str(new_cols[idx])
                if len(curr) < 2 or '?' in curr or any(ord(c) > 65533 for c in curr):
                    new_cols[idx] = name
                elif curr.strip() == '':
                    new_cols[idx] = name
        
        # Fallback: if columns still look like Column_1, Column_2 etc, just force names
        if len(new_cols) > 2 and 'number' not in [c.lower() for c in new_cols]:
            for idx, name in mapping.items():
                if idx < len(new_cols):
                    new_cols[idx] = name

        df.columns = new_cols
        return df

    def get_latest_investor_data(self, days=30):
        """
        Optimized method to fetch only the latest N distinct trading days of investor data.
        """
        try:
            # Table stock_investor has ~900k rows. Fetch only recent rows (last ~60 days * ~1000 tickers = 60000)
            # This balances performance and accuracy.
            query = "SELECT * FROM stock_investor ORDER BY 1 DESC LIMIT 100000"
            df = pd.read_sql(text(query), con=self.engine)
            df = self._fix_investor_columns(df)
            
            if not df.empty:
                # Filter to last 'days' unique dates
                unique_dates = sorted(df['日期'].unique(), reverse=True)[:int(days)]
                df = df[df['日期'].isin(unique_dates)]
                
            return df
        except Exception as e:
            logger.error(f"Failed to fetch latest investor data: {e}")
            return pd.DataFrame()

    def upload_investor_bulk(self, df, table_name='stock_investor'):
        if df.empty:
            return

        with self.engine.begin() as conn:
            # --- 自動建表與索引檢查邏輯 ---
            check_table = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'")).fetchone()
            
            if not check_table:
                logger.info(f"偵測到資料表 {table_name} 不存在，正在執行初始化...")
                from sqlalchemy import types
                dtype_dict = {'number': types.VARCHAR(20), '日期': types.VARCHAR(20)}
                df.head(0).to_sql(name=table_name, con=self.engine, if_exists='replace', index=False, dtype=dtype_dict)
                
                index_name = f"idx_{table_name}_unique"
                conn.execute(text(f"ALTER TABLE `{table_name}` ADD UNIQUE INDEX `{index_name}` (`日期`, `number`)"))
                logger.info(f"已成功建立資料表 {table_name} 與唯一索引。")

            # --- 批量寫入邏輯 ---
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

    def init_financial_tables(self):
        """初始化財報相關資料表"""
        queries = [
            # 股票基本資訊表 (TW)
            """
            CREATE TABLE IF NOT EXISTS stocks_tw (
                symbol VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100),
                market VARCHAR(20),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
            # 股票基本資訊表 (US)
            """
            CREATE TABLE IF NOT EXISTS stocks_us (
                symbol VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100),
                market VARCHAR(20),
                cik VARCHAR(20),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
            # 財報原始數據表 (TW) - 窄表格式
            """
            CREATE TABLE IF NOT EXISTS financial_raw_tw (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                year INT,
                quarter INT,
                statement_type ENUM('IS', 'BS', 'CF'),
                item_name VARCHAR(255),
                amount DECIMAL(32, 4),
                UNIQUE KEY idx_unique_item (symbol, year, quarter, statement_type, item_name)
            )
            """,
            # 財報原始數據表 (US) - 窄表格式
            """
            CREATE TABLE IF NOT EXISTS financial_raw_us (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                year INT,
                quarter INT,
                statement_type ENUM('IS', 'BS', 'CF'),
                item_name VARCHAR(255),
                amount DECIMAL(32, 4),
                UNIQUE KEY idx_unique_item (symbol, year, quarter, statement_type, item_name)
            )
            """,
            # 股價資料表 (US) - 若尚未建立
            """
            CREATE TABLE IF NOT EXISTS stock_cost_us (
                number VARCHAR(20),
                Date DATETIME,
                Open FLOAT,
                High FLOAT,
                Low FLOAT,
                Close FLOAT,
                Volume BIGINT,
                UNIQUE KEY idx_unique_date (Date, number)
            )
            """,
            # 修正現有表格：若已存在則調整長度
            "ALTER TABLE financial_raw_tw MODIFY COLUMN amount DECIMAL(32, 4)",
            "ALTER TABLE financial_raw_us MODIFY COLUMN amount DECIMAL(32, 4)"
        ]
        with self.engine.begin() as conn:
            for q in queries:
                conn.execute(text(q))
        logger.info("財報資料表初始化完成")

    def init_ai_prediction_tables(self):
        """初始化 AI 預測結果快取與軌跡資料表"""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS stock_ai_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20),
                date DATE COMMENT '預測產生的基準日',
                pred_day_1 FLOAT,
                pred_day_2 FLOAT,
                pred_day_3 FLOAT,
                pred_day_4 FLOAT,
                pred_day_5 FLOAT,
                trend_probability FLOAT COMMENT '綜合看漲機率',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY idx_unique_prediction (symbol, date)
            )
            """
        ]
        try:
            with self.engine.begin() as conn:
                for q in queries:
                    conn.execute(text(q))
            logger.info("AI 預測資料表 stock_ai_predictions 初始化完成")
        except Exception as e:
            logger.error(f"初始化 AI 預測資料表失敗: {e}")

    def bulk_upsert_raw_financials(self, df, market='tw'):
        """
        批量更新財報原始數據
        df 欄位應包含: symbol, year, quarter, statement_type, item_name, amount
        """
        if df.empty:
            return
        
        table_name = f'financial_raw_{market}'
        cols = ["symbol", "year", "quarter", "statement_type", "item_name", "amount"]
        
        # 資料清洗：過濾掉非有限數值 (NaN, Inf) 以防 MySQL 報錯
        import numpy as np
        df = df[np.isfinite(df['amount'])].copy()
        
        if df.empty:
            return
        
        sql = f"""
            INSERT INTO {table_name} (symbol, year, quarter, statement_type, item_name, amount)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE amount = VALUES(amount)
        """
        data_list = [tuple(x) for x in df[cols].values]
        
        with self.engine.begin() as conn:
            try:
                conn.exec_driver_sql(sql, data_list)
                logger.info(f"{market.upper()} 財報數據批量寫入完成：{len(df)} 筆")
            except Exception as e:
                logger.error(f"{market.upper()} 財報數據寫入失敗: {e}")
                raise e

    def upsert_stocks_info(self, df, market='tw'):
        """更新股票基本資訊"""
        if df.empty:
            return
            
        table_name = f'stocks_{market}'
        if market == 'tw':
            cols = ["symbol", "name", "market"]
            update_clause = "name = VALUES(name), market = VALUES(market)"
            placeholders = "%s, %s, %s"
        else:
            cols = ["symbol", "name", "market", "cik"]
            update_clause = "name = VALUES(name), market = VALUES(market), cik = VALUES(cik)"
            placeholders = "%s, %s, %s, %s"
            
        sql = f"""
            INSERT INTO {table_name} ({', '.join(cols)})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_clause}
        """
        data_list = [tuple(x) for x in df[cols].values]
        
        with self.engine.begin() as conn:
            try:
                conn.exec_driver_sql(sql, data_list)
                logger.info(f"{market.upper()} 股票資訊更新完成：{len(df)} 筆")
            except Exception as e:
                logger.error(f"{market.upper()} 股票資訊更新失敗: {e}")
                raise e

    def save_sentiment_embeddings(self, symbol: str, daily_embeddings: Dict[Any, Any]) -> None:
        """儲存語義向量到快取表 (Stock Sentiment Embeddings)"""
        if not daily_embeddings:
            return
            
        import numpy as np
        sql = """
            INSERT INTO stock_sentiment_embeddings (symbol, date, embedding_vector)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE embedding_vector = VALUES(embedding_vector)
        """
        
        data_list = []
        for d, emb in daily_embeddings.items():
            if isinstance(emb, np.ndarray):
                # Convert to float32 to save space and ensure consistency
                # 768 * 4 bytes = 3072 bytes
                binary_data = emb.astype(np.float32).tobytes()
                data_list.append((str(symbol), str(d), binary_data))

        if not data_list:
            return

        with self.engine.begin() as conn:
            try:
                conn.exec_driver_sql(sql, data_list)
                logger.info(f"成功快取 {symbol} 的 {len(data_list)} 筆語義向量")
            except Exception as e:
                logger.error(f"快取語義向量失敗 ({symbol}): {e}")

    def get_sentiment_embeddings(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """從快取表讀取特定範圍的語義向量"""
        import numpy as np
        # Explicitly conversion to string for SQL safety
        sql = "SELECT date, embedding_vector FROM stock_sentiment_embeddings WHERE symbol = :sym AND date >= :start AND date <= :end"
        
        results = {}
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text(sql), {"sym": str(symbol), "start": str(start_date), "end": str(end_date)}).fetchall()
                for row in res:
                    # row[0] might be a datetime.date object
                    d_obj = row[0]
                    d_str = d_obj.strftime('%Y-%m-%d') if hasattr(d_obj, 'strftime') else str(d_obj)
                    results[d_str] = np.frombuffer(row[1], dtype=np.float32)
            return results
        except Exception as e:
            logger.error(f"讀取語義向量快取失敗 ({symbol}): {e}")
            return {}