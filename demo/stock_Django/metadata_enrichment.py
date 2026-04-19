import sys
import os
import shutil
import certifi
import yfinance as yf
import pandas as pd
import logging
from sqlalchemy import text
from stock_Django.mySQL_OP import OP_Fun

# Fix SSL Cert path issues on Windows with non-ASCII characters
safe_cert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cacert.pem')
try:
    if not os.path.exists(safe_cert_path):
        shutil.copy(certifi.where(), safe_cert_path)
    os.environ['SSL_CERT_FILE'] = safe_cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = safe_cert_path
    os.environ['CURL_CA_BUNDLE'] = safe_cert_path
except Exception as e:
    print(f"Warning: Could not set up safe cert path: {e}")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MetadataEnricher:
    def __init__(self):
        self.sql_op = OP_Fun()
        self._init_db_table()

    def _init_db_table(self):
        """建立 stock_metadata 資料表"""
        with self.sql_op.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_metadata (
                    symbol VARCHAR(20) PRIMARY KEY,
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    market_cap BIGINT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """))
            logger.info("Checked/Created stock_metadata table.")

    def enrich_us_stocks(self, limit=None):
        """從 stocks_us 抓取代碼並補全 metadata"""
        # 1. 取得所有美股代碼
        query = "SELECT symbol FROM stocks_us"
        if limit:
            query += f" LIMIT {limit}"
            
        us_stocks = pd.read_sql(text(query), con=self.sql_op.engine)
        if us_stocks.empty:
            logger.warning("No US stocks found in stocks_us table.")
            return

        symbols = us_stocks['symbol'].tolist()
        logger.info(f"Starting enrichment for {len(symbols)} US stocks...")

        for symbol in symbols:
            try:
                # yfinance 會自動處理 session，特別是具有特定要求的 curl_cffi 版本
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                metadata = {
                    'symbol': symbol,
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0)
                }
                
                self._upsert_metadata(metadata)
                logger.info(f"Updated metadata for {symbol}: {metadata['sector']}")
                
            except Exception as e:
                logger.error(f"Failed to enrich {symbol}: {e}")

    def _upsert_metadata(self, data):
        """更新或插入 Metadata"""
        with self.sql_op.engine.begin() as conn:
            sql = """
                INSERT INTO stock_metadata (symbol, sector, industry, market_cap)
                VALUES (:symbol, :sector, :industry, :market_cap)
                ON DUPLICATE KEY UPDATE 
                    sector=VALUES(sector), 
                    industry=VALUES(industry), 
                    market_cap=VALUES(market_cap)
            """
            conn.execute(text(sql), data)

if __name__ == "__main__":
    enricher = MetadataEnricher()
    # 第一次執行建議先測 10 檔觀察效果
    enricher.enrich_us_stocks(limit=10)
