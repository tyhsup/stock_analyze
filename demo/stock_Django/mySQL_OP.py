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

        import threading
        temp_table = f"{table_name}_temp_{threading.get_ident()}"
        try:
            data.to_sql(name=temp_table, con=self.engine, if_exists='replace', index=False, dtype=dtype_dict, method='multi')
            with self.engine.begin() as conn:
                check_table = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'")).fetchone()
                if not check_table:
                    conn.execute(text(f"CREATE TABLE `{table_name}` LIKE `{temp_table}`"))
                    if date_col and 'number' in data.columns:
                        index_name = f"idx_{table_name}_unique"
                        conn.execute(text(f"ALTER TABLE `{table_name}` ADD UNIQUE INDEX `{index_name}` (`{date_col}`, `number`)"))

                # 採用 ON DUPLICATE KEY UPDATE 以符合標準並確保數據一致性
                cols_list = [f"`{c}`" for c in data.columns]
                # 排除唯一索引鍵 (Date, number)
                update_list = [f"{c}=VALUES({c})" for c in cols_list if c.replace("`", "") not in [date_col, 'number', '日期']]
                
                if update_list:
                    insert_sql = f"INSERT INTO `{table_name}` ({', '.join(cols_list)}) SELECT {', '.join(cols_list)} FROM `{temp_table}` ON DUPLICATE KEY UPDATE {', '.join(update_list)}"
                else:
                    insert_sql = f"INSERT IGNORE INTO `{table_name}` SELECT * FROM `{temp_table}`"
                
                conn.execute(text(insert_sql))
                conn.execute(text(f"DROP TABLE `{temp_table}`"))
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
            update_cols = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in df.columns if c not in ['日期', 'date', 'number']])
        
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
            # 股價表 (US) - 歷史收盤
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
            # 新增：三大法人買賣表 (TW) - 標準英文欄位
            """
            CREATE TABLE IF NOT EXISTS stock_investor_tw (
                date DATE NOT NULL,
                number VARCHAR(20) NOT NULL,
                name VARCHAR(100) DEFAULT NULL,
                foreign_buy DECIMAL(18, 4) DEFAULT 0.0000,
                foreign_sell DECIMAL(18, 4) DEFAULT 0.0000,
                foreign_net DECIMAL(18, 4) DEFAULT 0.0000,
                trust_buy DECIMAL(18, 4) DEFAULT 0.0000,
                trust_sell DECIMAL(18, 4) DEFAULT 0.0000,
                trust_net DECIMAL(18, 4) DEFAULT 0.0000,
                dealer_buy DECIMAL(18, 4) DEFAULT 0.0000,
                dealer_sell DECIMAL(18, 4) DEFAULT 0.0000,
                dealer_net DECIMAL(18, 4) DEFAULT 0.0000,
                total_net DECIMAL(18, 4) DEFAULT 0.0000,
                PRIMARY KEY (date, number),
                UNIQUE INDEX idx_investor_tw_unique (date, number)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            # 新增：融資融券信用交易餘額表 (TW)
            """
            CREATE TABLE IF NOT EXISTS stock_margin_balance (
                date DATE NOT NULL,
                number VARCHAR(20) NOT NULL,
                margin_purchase DECIMAL(18, 4) DEFAULT 0.0000,
                margin_sales DECIMAL(18, 4) DEFAULT 0.0000,
                margin_balance DECIMAL(18, 4) DEFAULT 0.0000,
                short_sale DECIMAL(18, 4) DEFAULT 0.0000,
                short_covering DECIMAL(18, 4) DEFAULT 0.0000,
                short_balance DECIMAL(18, 4) DEFAULT 0.0000,
                margin_utilization_rate DECIMAL(8, 4) DEFAULT 0.0000,
                short_utilization_rate DECIMAL(8, 4) DEFAULT 0.0000,
                PRIMARY KEY (date, number),
                UNIQUE INDEX idx_margin_unique (date, number)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            # 新增：集保所股權分散與大股東持股表 (TW)
            """
            CREATE TABLE IF NOT EXISTS stock_shareholder_distribution (
                date DATE NOT NULL,
                number VARCHAR(20) NOT NULL,
                class_1_to_5_ratio DECIMAL(8, 4) DEFAULT 0.0000,
                class_15_ratio DECIMAL(8, 4) DEFAULT 0.0000,
                total_shareholders INT DEFAULT 0,
                PRIMARY KEY (date, number),
                UNIQUE INDEX idx_distribution_unique (date, number)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """,
            # 修正欄位長度若已存在則變更
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

    def get_industry_investor_summary(self, days: int) -> pd.DataFrame:
        """
        獲取台灣股市個股近 days 天的籌碼與價格統計資訊，用於產業資金流向分析與多頭個股評分。
        """
        try:
            # 1. 取得最近的交易日期
            recent_dates_query = """
                SELECT DISTINCT date
                FROM stock_investor
                ORDER BY date DESC
                LIMIT :days
            """
            recent_dates_df = pd.read_sql(text(recent_dates_query), con=self.engine, params={'days': int(days)})
            if recent_dates_df.empty:
                return pd.DataFrame()
            
            recent_dates = recent_dates_df['date'].tolist()
            recent_dates_str = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in recent_dates]
            
            # 2. 讀取三大法人買賣超資料 (已限制日期範圍)
            df_investor = pd.read_sql(
                text("SELECT * FROM stock_investor WHERE date IN :dates"),
                con=self.engine,
                params={'dates': tuple(recent_dates_str)}
            )
            if df_investor.empty:
                return pd.DataFrame()
                
            df_investor = self._fix_investor_columns(df_investor)
            df_investor['date_str'] = df_investor['日期'].astype(str)
            
            # 3. 讀取產業別對照資料
            df_table_tw = pd.read_sql(
                text("SELECT 有價證卷代號 AS symbol, 產業別 AS industry FROM stock_table_tw"),
                con=self.engine
            )
            
            # 4. 讀取歷史股價資料 (已限制日期範圍)
            df_cost = pd.read_sql(
                text("SELECT number, Date, Close, Volume FROM stock_cost WHERE Date IN :dates"),
                con=self.engine,
                params={'dates': tuple(recent_dates_str)}
            )
            if df_cost.empty:
                return pd.DataFrame()
            df_cost['date_str'] = df_cost['Date'].astype(str).str.split(' ').str[0]
            df_cost['symbol_clean'] = df_cost['number'].str.replace('.TW', '', case=False, regex=False).str.replace('.TWO', '', case=False, regex=False)
            df_cost = df_cost.drop(columns=['number'])
            
            # 5. 讀取融資融券資料 (已限制日期範圍)
            df_margin = pd.read_sql(
                text("SELECT date, number, margin_balance, short_balance FROM stock_margin_balance WHERE date IN :dates"),
                con=self.engine,
                params={'dates': tuple(recent_dates_str)}
            )
            df_margin['date_str'] = df_margin['date'].astype(str)
            
            # 6. 在 Pandas 中進行合併 (避免資料庫無索引導致 full table JOIN 效能瓶頸)
            df_merged = pd.merge(df_investor, df_table_tw, left_on='number', right_on='symbol', how='inner')
            df_merged = pd.merge(df_merged, df_cost, left_on=['number', 'date_str'], right_on=['symbol_clean', 'date_str'], how='inner')
            df_merged = pd.merge(df_merged, df_margin, left_on=['number', 'date_str'], right_on=['number', 'date_str'], how='left')
            
            if df_merged.empty:
                return pd.DataFrame()
                
            # 7. 資料轉型與清洗
            from stock_Django.stock_utils import StockUtils
            industry_col = df_merged['industry'].copy()
            date_str_col = df_merged['date_str'].copy()
            
            df_merged = StockUtils.transfer_numeric(df_merged)
            df_merged['industry'] = industry_col
            df_merged['date_str'] = date_str_col
            
            # 8. 排序並計算時間序列指標
            df_sorted = df_merged.sort_values(by=['number', 'date_str'], ascending=True)
            
            # 計算連續買超天數 (三大法人買賣超股數 > 0)
            def calc_consecutive_buys(series):
                count = 0
                for val in reversed(series.tolist()):
                    if val > 0:
                        count += 1
                    else:
                        break
                return count
                
            consec_buys = df_sorted.groupby('number')['三大法人買賣超股數'].apply(calc_consecutive_buys)
            
            # 計算資券變動 (最新 - 最舊)
            margin_latest = df_sorted.groupby('number')['margin_balance'].last()
            margin_earliest = df_sorted.groupby('number')['margin_balance'].first()
            margin_change = margin_latest - margin_earliest
            
            short_latest = df_sorted.groupby('number')['short_balance'].last()
            short_earliest = df_sorted.groupby('number')['short_balance'].first()
            short_change = short_latest - short_earliest
            
            # 計算期間法人買賣超總金額 (三大法人買賣超股數 * Close / 1000)
            df_sorted['net_flow'] = df_sorted['三大法人買賣超股數'] * df_sorted['Close'] / 1000
            df_sorted['volume_value'] = df_sorted['Volume'] * df_sorted['Close']
            
            net_flow_sum = df_sorted.groupby('number')['net_flow'].sum()
            volume_value_sum = df_sorted.groupby('number')['volume_value'].sum()
            total_net_buy = df_sorted.groupby('number')['三大法人買賣超股數'].sum()
            total_volume = df_sorted.groupby('number')['Volume'].sum()
            
            # 取最後一天的股價與基本資料
            latest_rows = df_sorted.groupby('number').last().copy()
            
            # 替換為累計與衍生欄位
            latest_rows['consec_buys'] = consec_buys
            latest_rows['margin_change'] = margin_change
            latest_rows['short_change'] = short_change
            latest_rows['accum_net_flow'] = net_flow_sum
            latest_rows['accum_volume_value'] = volume_value_sum
            latest_rows['total_net_buy'] = total_net_buy
            latest_rows['total_volume'] = total_volume
            
            # 重設索引以方便後續處理
            return latest_rows.reset_index()
            
        except Exception as e:
            logger.error(f"get_industry_investor_summary 執行失敗: {e}")
            return pd.DataFrame()