import pandas as pd
import yfinance as yf
import datetime
from datetime import timedelta
import time
import random
import logging
import mySQL_OP
from selenium import webdriver
from bs4 import BeautifulSoup
from sqlalchemy import text

# --- 日誌設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("stock_batch_update.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockCostManager:
    def __init__(self):
        self.sql = mySQL_OP.OP_Fun()

    def fetch_latest_stock_list(self):
        """抓取最新上市股票編號"""
        logger.info("正在從 ISIN 系統抓取最新上市股票清單...")
        url = 'http://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        options = webdriver.EdgeOptions()
        options.add_argument('--headless')
        
        try:
            driver = webdriver.Edge(options=options)
            driver.get(url)
            time.sleep(5) 
            soup = BeautifulSoup(driver.page_source, "lxml")
            tr = soup.find_all('tr')
            
            stock_numbers = []
            for row in tr:
                tds = [td.get_text().strip() for td in row.find_all("td")]
                # 篩選標準
                if len(tds) == 7 and '　' in tds[0]:
                    if '有價證券代號' in tds[0]: continue
                    data_number = tds[0].split('　')[0].strip()
                    if data_number.isdigit() and len(data_number) == 4:
                        stock_numbers.append(data_number)
            driver.quit()
            logger.info(f"成功取得最新上市股票清單，共 {len(stock_numbers)} 支")
            return stock_numbers
        except Exception as e:
            logger.error(f"即時抓取股票清單失敗: {e}")
            return []

    def check_table_empty(self, table_name):
        """檢查資料表狀態"""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            with self.sql.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                return result[0] == 0
        except Exception:
            return True

    def update_all_cost(self):
        """批量更新股價資料 (修正 Rate Limit 邏輯)"""
        stock_numbers = self.fetch_latest_stock_list()
        if not stock_numbers:
            logger.error("無法取得股票清單，終止更新作業")
            return

        total_stocks = len(stock_numbers)
        is_empty = self.check_table_empty('stock_cost')
        
        # 全量下載建議 batch 設小，日常則維持 10
        batch_size = 5 if is_empty else 10
        download_period = "max" if is_empty else "10d"
        
        logger.info(f"執行模式: {'全量下載' if is_empty else '日常更新'}")

        i = 0
        while i < total_stocks:
            batch_list = stock_numbers[i : i + batch_size]
            symbols = [f"{num}.TW" for num in batch_list]
            
            try:
                logger.info(f"進度 ({i+1}/{total_stocks}) 抓取: {symbols}")
                
                # 下載資料
                data = yf.download(
                    tickers=symbols, 
                    period=download_period, 
                    interval="1d", 
                    auto_adjust=True, 
                    group_by='ticker',
                    progress=False
                )

                # --- 關鍵修正：檢查 Rate Limit ---
                # yfinance 有時不會噴 Exception，而是回傳一個包含錯誤訊息的空 DataFrame
                if data.empty:
                    # 嘗試檢查 yfinance 是否有內部抓取錯誤
                    logger.warning(f"Yahoo 回傳資料為空。可能是觸發限速或該時段無交易。")
                    # 如果連續多組都空，通常是限速，強制休息
                    logger.error("疑似觸發限速 (Rate Limit)，強制休息 10 分鐘...")
                    time.sleep(600)
                    continue  # 不增加 i，休息完後重試同一組

                # 處理下載成功的資料
                for num in batch_list:
                    symbol = f"{num}.TW"
                    if len(batch_list) > 1:
                        if symbol not in data.columns.levels[0]: continue
                        stock_df = data[symbol].copy()
                    else:
                        stock_df = data.copy()
                    
                    stock_df = stock_df.dropna()
                    if stock_df.empty: continue

                    stock_df.reset_index(inplace=True)
                    stock_df.insert(0, 'number', num)
                    if isinstance(stock_df.columns, pd.MultiIndex):
                        stock_df.columns = stock_df.columns.get_level_values(0)

                    # 上傳至資料庫
                    self.sql.upload_all(stock_df, 'stock_cost')

                # 成功處理完一組，索引才增加
                i += batch_size
                time.sleep(random.uniform(3, 6))

            except Exception as e:
                error_msg = str(e)
                if 'Rate limited' in error_msg or '429' in error_msg or 'Too Many Requests' in error_msg:
                    logger.error(f"偵測到限速錯誤: {error_msg}。將休息 10 分鐘後重試...")
                    time.sleep(600)
                    # 不增加 i，讓迴圈重跑同一組
                else:
                    logger.error(f"發生未預期錯誤: {e}")
                    i += batch_size # 避開死循環，跳過此組

        logger.info("所有更新作業結束")

if __name__ == "__main__":
    manager = StockCostManager()
    manager.update_all_cost()
