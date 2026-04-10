import pandas as pd
import yfinance as yf
import datetime
from datetime import timedelta
import time
import random
import logging
try:
    from . import mySQL_OP
except (ImportError, ValueError):
    import mySQL_OP
try:
    from selenium import webdriver
    from bs4 import BeautifulSoup
except ImportError:
    webdriver = None
    BeautifulSoup = None
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

from typing import List, Dict, Any, Optional

class StockCostManager:
    def __init__(self) -> None:
        """初始化 StockCostManager，設定資料庫連線並初始化表格。"""
        self.sql = mySQL_OP.OP_Fun()
        # 初始化表格 (確保 stock_cost/stock_cost_us 與索引存在)
        self.sql.init_financial_tables()
        
        # 初始化持久化 Session (遵循 yfinance Best Practices/Domain Rules)
        from curl_cffi.requests import Session as CurlSession
        import requests
        self.curl_session = CurlSession(verify=False)
        self.http_session = requests.Session()
        
        # User-Agent 清單庫
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
        ]
        self._rotate_user_agent()

    def _rotate_user_agent(self) -> None:
        """輪替 Session 的 User-Agent 以規避單一指紋封鎖。"""
        ua = random.choice(self.user_agents)
        self.curl_session.headers.update({"User-Agent": ua})
        self.http_session.headers.update({"User-Agent": ua})
        logger.info(f"已輪替 User-Agent: {ua[:30]}...")

    def _is_otc_stock(self, code: str) -> bool:
        """查詢資料庫判斷代碼是否為上櫃股票。

        Args:
            code: 純數字股票代碼。

        Returns:
            True 表示上櫃(.TWO)，False 表示上市(.TW)。
        """
        try:
            with self.sql.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM stock_cost WHERE number = :num"),
                    {'num': f"{code}.TWO"}
                ).fetchone()
                if result and result[0] > 0:
                    return True
        except Exception:
            pass
        return False

    def _scrape_stock_codes(self, url: str) -> List[str]:
        """從 TWSE ISIN 頁面抓取 4 位數股票代碼的私有方法。

        Args:
            url (str): 要抓取的 TWSE 頁面 URL。

        Returns:
            List[str]: 抓取到的 4 位數股票代碼清單。
        """
        if not webdriver:
            logger.error("缺少 Selenium，無法執行抓取")
            return []
        
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
                if len(tds) == 7 and '　' in tds[0]:
                    if '有價證券代號' in tds[0]: continue
                    data_number = tds[0].split('　')[0].strip()
                    # 只抓取 4 位數的股票 (排除權證等)
                    if data_number.isdigit() and len(data_number) == 4:
                        stock_numbers.append(data_number)
            driver.quit()
            return stock_numbers
        except Exception as e:
            logger.error(f"抓取 {url} 股票清單失敗: {e}")
            return []

    def fetch_all_tw_stock_list(self) -> List[Dict[str, str]]:
        """抓取上市 (strMode=2) 與 上櫃 (strMode=4) 股票清單並整合。

        Returns:
            List[Dict[str, str]]: 包含股票代碼與字尾的字典清單，例如 [{'code': '2330', 'suffix': '.TW'}, ...]。
        """
        logger.info("正在抓取上市與上櫃股票清單...")
        
        listed_url = 'http://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        otc_url = 'http://isin.twse.com.tw/isin/C_public.jsp?strMode=4'
        
        listed_codes = self._scrape_stock_codes(listed_url)
        logger.info(f"成功取得上市股票清單，共 {len(listed_codes)} 支")
        
        otc_codes = self._scrape_stock_codes(otc_url)
        logger.info(f"成功取得上櫃股票清單，共 {len(otc_codes)} 支")
        
        all_stocks = []
        for code in listed_codes:
            all_stocks.append({"code": code, "suffix": ".TW"})
        for code in otc_codes:
            all_stocks.append({"code": code, "suffix": ".TWO"})
            
        return all_stocks

    def fetch_latest_stock_list(self):
        """保持向後相容性，原本只回傳上市代碼"""
        listed_url = 'http://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        return self._scrape_stock_codes(listed_url)

    def get_stock_stats(self, table_name: str) -> Dict[str, Dict[str, Any]]:
        """獲取資料庫中各股票的統計資料 (最後日期、最早日期、筆數)。

        Args:
            table_name (str): 要查詢的資料表名稱。

        Returns:
            Dict[str, Dict[str, Any]]: 以股票代碼為鍵，統計資料為值的字典。
        """
        query = f"""
            SELECT number, 
                   MAX(Date) as last_date, 
                   MIN(Date) as first_date, 
                   COUNT(*) as row_count 
            FROM {table_name} 
            GROUP BY number
        """
        try:
            with self.sql.engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
                return df.set_index('number').to_dict('index')
        except Exception as e:
            logger.warning(f"獲取 {table_name} 統計失敗 (可能表格尚無資料): {e}")
            return {}

    def _fetch_prices(self, tickers: List[str], period: Optional[str] = None, 
                      start: Optional[str] = None, end: Optional[str] = None,
                      retry_count: int = 0) -> pd.DataFrame:
        """私有輔助方法：嘗試多個來源抓取股價 (Resilient Fetch)。

        Args:
            tickers (List[str]): 股票代碼清單。
            period (Optional[str]): 抓取期間 (如 '1d', 'max')。
            start (Optional[str]): 開始日期 (YYYY-MM-DD)。
            end (Optional[str]): 結束日期 (YYYY-MM-DD)。
            retry_count (int): 目前重試次數。

        Returns:
            pd.DataFrame: 抓取到的股價資料。
        """
        # 1. 嘗試 yfinance (優先使用持久化 session)
        try:
            params = {
                "tickers": tickers,
                "interval": "1d",
                "auto_adjust": True,
                "group_by": 'ticker',
                "progress": False,
                "session": self.curl_session,
                "repair": True,        # Automatic price repair
                "timeout": 30          # 增加超時寬容度
            }
            if start: params["start"] = start
            if end: params["end"] = end
            if not start and period: params["period"] = period
            
            data = yf.download(**params)
            
            if not data.empty:
                if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
                    data.columns = pd.MultiIndex.from_product([[tickers[0]], data.columns])
                return data
            
        except Exception as e:
            msg = str(e)
            if any(key in msg for key in ["Rate limited", "429", "Too Many Requests"]):
                if retry_count < 2:
                    wait_time = random.randint(60, 120) * (retry_count + 1)
                    logger.warning(f"偵測到 YFRateLimitError for {tickers}。將休息 {wait_time}s 後更換 UA 並重試 ({retry_count+1}/2)")
                    time.sleep(wait_time)
                    self._rotate_user_agent()
                    return self._fetch_prices(tickers, period, start, end, retry_count + 1)
                else:
                    logger.error(f"達到最大重試次數，YF 依然限速 for {tickers}")
            else:
                logger.warning(f"yfinance Download Error for {tickers}: {e}")

        # 2. 如果 yf 失敗或為空，嘗試 yahooquery (備援)
        try:
            from yahooquery import Ticker
            # yahooquery 暫時手動傳入 verify=False
            t = Ticker(tickers, verify=False, session=self.http_session)
            
            yq_params = {"interval": '1d'}
            if start: yq_params["start"] = start
            if end: yq_params["end"] = end
            if not start and period: yq_params["period"] = period
            
            history = t.history(**yq_params)
            
            if not history.empty and isinstance(history, pd.DataFrame):
                logger.info(f"YahooQuery Fallback Success for {tickers}")
                history = history.reset_index()
                history = history.rename(columns={
                    'symbol': 'ticker', 'date': 'Date',
                    'adjclose': 'Close', 'adj_close': 'Close',
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'
                })
                pivoted = history.pivot(index='Date', columns='ticker')
                pivoted = pivoted.swaplevel(0, 1, axis=1).sort_index(axis=1)
                return pivoted
        except Exception as e2:
            logger.warning(f"YahooQuery Fallback Error for {tickers}: {e2}")
        
        # 3. 如果是台股，嘗試 TWSE 直接抓取 (最後手段)
        if isinstance(tickers, list) and len(tickers) == 1:
            sym = tickers[0]
            if sym.endswith('.TW'):
                try:
                    num = sym.split('.')[0]
                    target_date = start if start else datetime.datetime.now().strftime('%Y%m%d')
                    target_date = target_date.replace('-', '').replace('/', '')
                    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?stockNo={num}&date={target_date}"
                    
                    logger.info(f"Trying TWSE Direct Fallback for {sym}")
                    resp = self.http_session.get(url, timeout=15)
                    if resp.status_code == 200:
                        json_data = resp.json()
                        if json_data.get('stat') == 'OK':
                            data_list = json_data['data']
                            df_twse = pd.DataFrame(data_list, columns=json_data['fields'])
                            def convert_date(d_str):
                                parts = d_str.split('/')
                                return f"{int(parts[0]) + 1911}-{parts[1]}-{parts[2]}"
                            df_twse['Date'] = pd.to_datetime(df_twse['日期'].apply(convert_date))
                            for col in ['開盤價', '最高價', '最低價', '收盤價', '成交股數']:
                                df_twse[col] = df_twse[col].str.replace(',', '').apply(pd.to_numeric, errors='coerce')
                            df_twse = df_twse.rename(columns={'開盤價':'Open','最高價':'High','最低價':'Low','收盤價':'Close','成交股數':'Volume'})
                            df_twse = df_twse.dropna(subset=['Close'])[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                            df_twse['ticker'] = sym
                            pivoted = df_twse.pivot(index='Date', columns='ticker').swaplevel(0, 1, axis=1).sort_index(axis=1)
                            return pivoted
                except Exception as e_twse:
                    logger.error(f"TWSE Direct Fallback failed for {sym}: {e_twse}")

            elif sym.endswith('.TWO'):
                try:
                    num = sym.split('.')[0]
                    target_date = start if start else datetime.datetime.now().strftime('%Y/%m/%d')
                    target_date = target_date.replace('-', '/')
                    tpex_url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?d={target_date}&stkno={num}"
                    
                    logger.info(f"Trying TPEx Direct Fallback for {sym}")
                    resp = self.http_session.get(tpex_url, timeout=15)
                    if resp.status_code == 200:
                        json_data = resp.json()
                        if json_data.get('aaData'):
                            cols = ['日期', '成交仟股', '成交仟元', '開盤', '最高', '最低', '收盤', '漲跌', '筆數']
                            df_tpex = pd.DataFrame(json_data['aaData'], columns=cols[:len(json_data['aaData'][0])])
                            def convert_tpex_date(d_str):
                                parts = d_str.strip().split('/')
                                return f"{int(parts[0]) + 1911}-{parts[1]}-{parts[2]}"
                            df_tpex['Date'] = pd.to_datetime(df_tpex['日期'].apply(convert_tpex_date))
                            for col in ['開盤', '最高', '最低', '收盤', '成交仟股']:
                                if col in df_tpex.columns:
                                    df_tpex[col] = df_tpex[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
                            df_tpex = df_tpex.rename(columns={'開盤':'Open','最高':'High','最低':'Low','收盤':'Close','成交仟股':'Volume'})
                            df_tpex = df_tpex.dropna(subset=['Close'])[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                            df_tpex['ticker'] = sym
                            pivoted = df_tpex.pivot(index='Date', columns='ticker').swaplevel(0, 1, axis=1).sort_index(axis=1)
                            return pivoted
                except Exception as e_tpex:
                    logger.error(f"TPEx Direct Fallback failed for {sym}: {e_tpex}")

        logger.error(f"All price sources failed for {tickers}.")
        return pd.DataFrame()

    def update_single_ticker(self, ticker: str) -> bool:
        """更新單一股票的股價資料 (增量更新)。

        Args:
            ticker (str): 股票代碼。

        Returns:
            bool: 更新是否成功。
        """
        ticker = ticker.upper()
        # 1. 判斷市場與資料表
        is_tw = ticker.isdigit() or ".TW" in ticker or ".TWO" in ticker
        table_name = 'stock_cost' if is_tw else 'stock_cost_us'
        
        # 規範代號 (台股補全: 查表判斷上市或上櫃)
        if is_tw and not (ticker.endswith('.TW') or ticker.endswith('.TWO')):
            if self._is_otc_stock(ticker):
                ticker = f"{ticker}.TWO"
            else:
                ticker = f"{ticker}.TW"

        # 2. 獲取資料庫中的最後日期
        stats = self.get_stock_stats(table_name)
        start_date = None
        period_param = "30d"
        
        if ticker in stats:
            last_date = stats[ticker]['last_date']
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            # 如果最後日期是今天或未來，則不需要更新
            if last_date.date() >= datetime.date.today():
                logger.info(f"單一更新 {ticker}: 資料已是最新 ({last_date.date()})，跳過抓取")
                return True

            # 從最後日期的隔天開始抓
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(f"單一更新 {ticker}: 偵測到最後日期 {last_date.date()}，從 {start_date} 開始抓取")
        else:
            period_param = "max"
            logger.info(f"單一更新 {ticker}: 資料庫無紀錄，採用 max 模式")

        # 3. 抓取資料
        try:
            data = self._fetch_prices([ticker], period=period_param, start=start_date)
            if data.empty:
                logger.warning(f"單一更新 {ticker} 抓取失敗或無新資料")
                return False

            # 4. 資料處理與上傳
            # 資料結構標準化 (確保為 Ticker, Metric)
            if isinstance(data.columns, pd.MultiIndex):
                ticker_key = ticker # Already upper/checked
                if ticker_key in data.columns.get_level_values(0).unique():
                    stock_df = data[ticker_key].copy()
                else:
                    logger.warning(f"單一更新 {ticker}: 資料層級中找不到該代號")
                    return False
            else:
                stock_df = data.copy()

            stock_df = stock_df.loc[:, ~stock_df.columns.duplicated(keep='first')]
            stock_df = stock_df.dropna()
            
            if stock_df.empty:
                logger.info(f"單一更新 {ticker} 無新交易日資料")
                return True

            stock_df.reset_index(inplace=True)
            if 'index' in stock_df.columns: stock_df = stock_df.rename(columns={'index': 'Date'})
            stock_df.insert(0, 'number', ticker)
            
            # Ensure index is not MultiIndex now
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)

            final_cols = ['number', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            stock_df = stock_df[[c for c in final_cols if c in stock_df.columns]]

            self.sql.upload_all(stock_df, table_name)
            logger.info(f"單一更新 {ticker} 成功上傳 {len(stock_df)} 筆資料")
            return True
        except Exception as e:
            logger.error(f"單一更新 {ticker} 失敗: {e}")
            return False

    def _process_and_upload_batch(self, data: pd.DataFrame, symbols: List[str], table_name: str):
        """處理抓取到的批量資料並上傳。"""
        if data.empty:
            return

        for symbol in symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if symbol in data.columns.get_level_values(0).unique():
                        stock_df = data[symbol].copy()
                    else:
                        continue
                else:
                    stock_df = data.copy() if len(symbols) == 1 else pd.DataFrame()
                
                if stock_df.empty: continue
                
                stock_df = stock_df.loc[:, ~stock_df.columns.duplicated(keep='first')]
                stock_df = stock_df.dropna()
                if stock_df.empty: continue

                stock_df.reset_index(inplace=True)
                if 'index' in stock_df.columns: stock_df = stock_df.rename(columns={'index': 'Date'})
                stock_df.insert(0, 'number', symbol)
                if isinstance(stock_df.columns, pd.MultiIndex):
                    stock_df.columns = stock_df.columns.get_level_values(0)

                final_cols = ['number', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                stock_df = stock_df[[c for c in final_cols if c in stock_df.columns]]

                self.sql.upload_all(stock_df, table_name)
            except Exception as e:
                logger.error(f"處理 {symbol} 時發生錯誤: {e}")

    def update_all_cost(self):
        """批量更新台股股價 (整合上市與上櫃，增量更新 + 自動補全)"""
        all_stocks = self.fetch_all_tw_stock_list()
        if not all_stocks:
            logger.error("無法取得股票清單，終止更新作業")
            return

        stats = self.get_stock_stats('stock_cost')
        
        # 分離「新加入/需全量」與「已存在/需增量」的股票
        new_stocks = []
        existing_stocks = []
        
        for s_info in all_stocks:
            symbol = f"{s_info['code']}{s_info['suffix']}"
            if symbol not in stats:
                new_stocks.append(symbol)
            else:
                s_stat = stats[symbol]
                # 若資料筆數過少且開盤不到一年，視為新股
                days_diff = (datetime.datetime.now() - pd.to_datetime(s_stat['first_date'])).days
                if s_stat['row_count'] < 100 and days_diff < 365:
                    new_stocks.append(symbol)
                else:
                    existing_stocks.append(symbol)
        
        logger.info(f"台股分類: 全量 {len(new_stocks)} 支, 增量 {len(existing_stocks)} 支")

        # 1. 處理全量 (小 batch 避開封鎖)
        for i in range(0, len(new_stocks), 5):
            batch = new_stocks[i : i + 5]
            logger.info(f"[台股-全量] 處理中 ({i+1}/{len(new_stocks)}): {batch}")
            data = self._fetch_prices(batch, period="max")
            self._process_and_upload_batch(data, batch, 'stock_cost')
            time.sleep(random.uniform(3, 7))

        # 2. 處理增量 (大 batch 提升效率)
        for i in range(0, len(existing_stocks), 20):
            batch = existing_stocks[i : i + 20]
            min_last = min([pd.to_datetime(stats[s]['last_date']) for s in batch])
            start_param = (min_last + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if pd.to_datetime(start_param).date() >= datetime.date.today():
                continue
                
            logger.info(f"[台股-增量] 處理中 ({i+1}/{len(existing_stocks)}): 從 {start_param} 抓取 {batch}")
            data = self._fetch_prices(batch, start=start_param)
            self._process_and_upload_batch(data, batch, 'stock_cost')
            time.sleep(random.uniform(2, 5))

    def update_us_cost(self):
        """更新美股股價 (增量更新 + 自動補全)"""
        query = "SELECT symbol FROM stocks_us"
        try:
            with self.sql.engine.connect() as conn:
                us_symbols = pd.read_sql(text(query), conn)['symbol'].tolist()
            
            if not us_symbols: return
            
            stats = self.get_stock_stats('stock_cost_us')
            
            new_stocks = []
            existing_stocks = []
            for s in us_symbols:
                if s not in stats:
                    new_stocks.append(s)
                else:
                    s_stat = stats[s]
                    days_diff = (datetime.datetime.now() - pd.to_datetime(s_stat['first_date'])).days
                    if s_stat['row_count'] < 100 and days_diff < 365:
                        new_stocks.append(s)
                    else:
                        existing_stocks.append(s)
            
            logger.info(f"美股分類: 全量 {len(new_stocks)} 支, 增量 {len(existing_stocks)} 支")

            # 1. 全量
            for i in range(0, len(new_stocks), 10):
                batch = new_stocks[i : i + 10]
                logger.info(f"[美股-全量] 處理中 ({i+1}/{len(new_stocks)}): {batch}")
                data = self._fetch_prices(batch, period="max")
                self._process_and_upload_batch(data, batch, 'stock_cost_us')
                time.sleep(random.uniform(2, 4))

            # 2. 增量
            for i in range(0, len(existing_stocks), 30):
                batch = existing_stocks[i : i + 30]
                min_last = min([pd.to_datetime(stats[s]['last_date']) for s in batch])
                start_param = (min_last + timedelta(days=1)).strftime('%Y-%m-%d')
                
                if pd.to_datetime(start_param).date() >= datetime.date.today():
                    continue

                logger.info(f"[美股-增量] 處理中 ({i+1}/{len(existing_stocks)}): 從 {start_param} 抓取 {batch}")
                data = self._fetch_prices(batch, start=start_param)
                self._process_and_upload_batch(data, batch, 'stock_cost_us')
                time.sleep(random.uniform(1, 3))
                
            logger.info("美股價格更新完成")
        except Exception as e:
            logger.error(f"美股更新失敗: {e}")

if __name__ == "__main__":
    manager = StockCostManager()
    logger.info(">>> 啟動全市場價格更新作業 <<<")
    # 1. 更新台股
    manager.update_all_cost()
    # 2. 更新美股
    manager.update_us_cost()
    logger.info(">>> 全市場價格更新作業結束 <<<")