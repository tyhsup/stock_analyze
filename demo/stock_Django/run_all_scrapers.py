import pandas as pd
from sqlalchemy import text
from mySQL_OP import OP_Fun
from scraper_us import get_us_stock_list, scrape_us_financials, SEC_HEADERS
from scraper_tw_pw import scrape_tw_financials_playwright
from scraper_utils import smart_delay
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def populate_us_data(limit=None):
    """自動化抓取美股財報 (5 年)"""
    op = OP_Fun()
    logger.info("Starting US Data Population...")
    
    # 1. 更新股票清單
    us_stocks = get_us_stock_list()
    op.upsert_stocks_info(us_stocks, market='us')
    
    # 取樣或全選
    stocks_to_process = us_stocks if limit is None else us_stocks.head(limit)
    start_year = 2024 - 5
    
    for idx, row in stocks_to_process.iterrows():
        symbol = str(row['symbol'])
        cik = str(row['cik'])
        
        # 檢查是否已有足夠資料 (斷點續傳)
        with op.engine.connect() as conn:
            check = conn.execute(text("SELECT COUNT(*) FROM financial_raw_us WHERE symbol = :s AND year >= :y"), 
                                {"s": symbol, "y": start_year}).scalar()
            if check > 500: # 粗略判斷是否已抓過
                logger.info(f"Skipping {symbol}, data already exists.")
                continue

        logger.info(f"Scraping US stock {symbol} ({int(idx)+1}/{len(stocks_to_process)})...")
        df = scrape_us_financials(symbol, cik)
        if not df.empty:
            df_filtered = df[df['year'] >= start_year]
            op.bulk_upsert_raw_financials(df_filtered, market='us')
            logger.info(f"Successfully uploaded {len(df_filtered)} items.")
        
        time.sleep(0.15) # SEC Rate Limit

def populate_tw_data(limit=None):
    """自動化抓取台股財報 (5 年)"""
    op = OP_Fun()
    logger.info("Starting TW Data Population...")
    
    with op.engine.connect() as conn:
        tw_stocks = pd.read_sql(text("SELECT symbol FROM stocks_tw"), conn)
    
    stocks_to_process = tw_stocks if limit is None else tw_stocks.head(limit)
    start_year = 2024 - 5
    
    for idx, sym_row in stocks_to_process.iterrows():
        symbol = sym_row['symbol']
        
        for year in range(start_year, 2024):
            for quarter in range(1, 5):
                # 檢查是否已存在
                with op.engine.connect() as conn:
                    check = conn.execute(text("SELECT id FROM financial_raw_tw WHERE symbol = :s AND year = :y AND quarter = :q LIMIT 1"), 
                                        {"s": symbol, "y": year, "q": quarter}).fetchone()
                    if check:
                        continue
                
                logger.info(f"Scraping TW stock {symbol} {year}Q{quarter}...")
                df = scrape_tw_financials_playwright(symbol, year, quarter)
                if not df.empty:
                    op.bulk_upsert_raw_financials(df, market='tw')
                    logger.info(f"Done.")
                
                # MOPS 防封鎖：延遲拉長
                smart_delay(10, 20)

if __name__ == "__main__":
    # 範例：抓取前 5 支美股與前 2 支台股
    populate_us_data(limit=None)
    # 台股較慢，範例僅跑 1 支
    populate_tw_data(limit=None)
