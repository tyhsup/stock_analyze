import pandas as pd
import yfinance as yf
import datetime
from datetime import timedelta
import time
import random
import logging
import mySQL_OP
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from bs4 import BeautifulSoup
from sqlalchemy import text
import ssl

# 全域 SSL 猴子補丁
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        self.session = self._init_session()

    def _init_session(self):
        """初始化具備重試機制的 Session 並徹底禁用 SSL 驗證"""
        session = requests.Session()
        
        class NoVerifyAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                kwargs['cert_reqs'] = 'CERT_NONE'
                return super(NoVerifyAdapter, self).init_poolmanager(*args, **kwargs)
            def proxy_manager_for(self, *args, **kwargs):
                kwargs['cert_reqs'] = 'CERT_NONE'
                return super(NoVerifyAdapter, self).proxy_manager_for(*args, **kwargs)

        adapter = NoVerifyAdapter(max_retries=5)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        session.verify = False 
        return session

    def fetch_latest_stock_list(self):
        """抓取最新上市股票編號 (優化：使用 requests 與 curl 回退)"""
        logger.info("正在從 ISIN 系統抓取最新上市股票清單...")
        url = 'http://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        
        try:
            html_text = ""
            try:
                response = self.session.get(url, timeout=15, verify=False)
                response.encoding = 'big5'
                html_text = response.text
            except Exception as e:
                if "SSL" in str(e) or "timeout" in str(e).lower() or "connection" in str(e).lower():
                    logger.warning("標準請求失敗，嘗試使用 curl.exe...")
                    import subprocess
                    result = subprocess.run(['curl.exe', '-L', '-k', '-s', url], capture_output=True)
                    # ISIN 使用 Big5 編碼
                    html_text = result.stdout.decode('big5', errors='ignore')
                else:
                    raise e

            if not html_text:
                return []

            soup = BeautifulSoup(html_text, "lxml")
            tr = soup.find_all('tr')
            
            stock_numbers = []
            for row in tr:
                tds = [td.get_text().strip() for td in row.find_all("td")]
                if len(tds) == 7 and '　' in tds[0]:
                    if '有價證券代號' in tds[0]: continue
                    data_number = tds[0].split('　')[0].strip()
                    if data_number.isdigit() and len(data_number) == 4:
                        stock_numbers.append(data_number)
            
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
        """批量更新股價資料 (優化：Session 管理與限速處理)"""
        stock_numbers = self.fetch_latest_stock_list()
        if not stock_numbers:
            logger.error("無法取得股票清單，終止更新作業")
            return

        total_stocks = len(stock_numbers)
        is_empty = self.check_table_empty('stock_cost')
        
        batch_size = 5 if is_empty else 10
        download_period = "max" if is_empty else "10d"
        
        logger.info(f"執行模式: {'全量下載' if is_empty else '日常更新'}")

        i = 0
        consecutive_errors = 0
        
        while i < total_stocks:
            batch_list = stock_numbers[i : i + batch_size]
            symbols = [f"{num}.TW" for num in batch_list]
            
            try:
                logger.info(f"進度 ({i+1}/{total_stocks}) 抓取: {symbols}")
                
                # 下載資料 (使用 session)
                data = yf.download(
                    tickers=symbols, 
                    period=download_period, 
                    interval="1d", 
                    auto_adjust=True, 
                    group_by='ticker',
                    progress=False,
                    session=self.session,
                    timeout=30
                )

                if data.empty:
                    consecutive_errors += 1
                    logger.warning(f"Yahoo 回傳資料為空。連續錯誤次數: {consecutive_errors}")
                    if consecutive_errors >= 3:
                        wait_time = 600 * (2 ** (consecutive_errors - 3))
                        logger.error(f"疑似觸發嚴重限速，強制休息 {wait_time} 秒...")
                        time.sleep(wait_time)
                    else:
                        time.sleep(30)
                    continue 

                consecutive_errors = 0
                
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

                    # 欄位標準化
                    stock_df.columns = [c.lower() for c in stock_df.columns]
                    
                    # 上傳至資料庫 (Upsert)
                    self.sql.upload_all(stock_df, 'stock_cost')

                i += batch_size
                time.sleep(random.uniform(2, 5))

            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'Too Many Requests' in error_msg:
                    logger.error(f"偵測到限速錯誤: {error_msg}。休息 10 分鐘...")
                    time.sleep(600)
                else:
                    logger.error(f"發生未預期錯誤: {e}")
                    i += batch_size 

        logger.info("所有更新作業結束")

if __name__ == "__main__":
    manager = StockCostManager()
    manager.update_all_cost()


