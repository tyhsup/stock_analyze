import re
import json
import time
import logging
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import yfinance as yf
from .mySQL_OP import OP_Fun

logger = logging.getLogger(__name__)

def get_slug(ticker: str) -> str:
    """透過 yfinance 獲取公司名稱，並推算 Macrotrends 的 slug"""
    try:
        ticker_ob = yf.Ticker(ticker)
        info = ticker_ob.info
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception as e:
        logger.warning(f"Failed to get longName from yfinance for {ticker}: {e}")
        name = ticker
        
    name = name.lower()
    # 移除常見的尾綴
    for suffix in [" inc.", " inc", " corporation", " corp.", " corp", " co.", " co", " ltd.", " ltd", " .com"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    # 移除非字母數字字元
    name = re.sub(r'[^a-z0-9\s-]', '', name)
    name = name.strip()
    # 將空白與減號轉為單一減號
    name = re.sub(r'[\s-]+', '-', name)
    return name

def parse_original_data_html(html: str) -> list:
    """從 HTML 中解析 originalData 變數"""
    match = re.search(r'var\s+originalData\s*=\s*(\[.*?\])\s*;', html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            logger.error(f"Failed to parse originalData JSON: {e}")
    return []

def scrape_macrotrends_via_selenium(url: str) -> list:
    """使用 Selenium 載入網頁並解析 originalData"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
    
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(8) # 等待 Cloudflare 挑戰與網頁渲染
        html = driver.page_source
        return parse_original_data_html(html)
    except Exception as e:
        logger.error(f"Selenium scrape failed for {url}: {e}")
        return []
    finally:
        if driver:
            driver.quit()

class MacrotrendsService:
    def __init__(self):
        self.op = OP_Fun()
        # 初始化財報相關的 MySQL Table
        self.op.init_financial_tables()

    def get_financials(self, symbol: str, stmt_type: str, frequency: str = "annual") -> list:
        """
        取得美股財務報表。
        優先讀取資料庫快取；若無，先爬取 Macrotrends，失敗則 Fallback 到 yfinance。
        """
        symbol = symbol.upper()
        # 報表類型轉寫
        stmt_map = {
            "income-statement": "IS",
            "balance-sheet": "BS",
            "cash-flow": "CF"
        }
        db_stmt_type = stmt_map.get(stmt_type, "IS")
        
        # 1. 查詢快取
        cached_df = self._get_cached_financials(symbol, db_stmt_type)
        if not cached_df.empty:
            logger.info(f"Cache hit for {symbol} {stmt_type}")
            return cached_df.to_dict(orient="records")
            
        # 2. 快取未命中，嘗試爬取 Macrotrends
        logger.info(f"Cache miss for {symbol} {stmt_type}. Fetching...")
        slug = get_slug(symbol)
        
        # 依頻率決定 URL (Macrotrends 季報通常在 url 加 ?freq=Q 或是用 UI。
        # 我們測試發現 URL 後綴直接使用原網址即可，如果季報爬取不到 originalData 則 Fallback)
        url = f"https://www.macrotrends.net/stocks/charts/{symbol}/{slug}/{stmt_type}"
        if frequency == "quarterly":
            url += "?freq=Q"
            
        logger.info(f"Navigating to Macrotrends: {url}")
        raw_data = scrape_macrotrends_via_selenium(url)
        
        df_parsed = pd.DataFrame()
        if raw_data:
            df_parsed = self._parse_macrotrends_raw_json(symbol, raw_data, db_stmt_type)
            
        # 3. 若爬取失敗或為空，執行 yfinance Fallback
        if df_parsed.empty:
            logger.warning(f"Macrotrends scrape failed or returned empty for {symbol}. Falling back to yfinance...")
            df_parsed = self._fetch_via_yfinance(symbol, stmt_type, frequency)
            
        # 4. 寫入快取與回傳
        if not df_parsed.empty:
            self.op.bulk_upsert_raw_financials(df_parsed, market="us")
            # 重新從資料庫讀取以保證回傳格式完全正確
            cached_df = self._get_cached_financials(symbol, db_stmt_type)
            return cached_df.to_dict(orient="records")
            
        return []

    def get_ratios(self, symbol: str, ratio_type: str) -> list:
        """
        取得美股財務比率歷史數據。
        優先讀取資料庫快取；若無，先爬取 Macrotrends，失敗則從 yfinance/歷史股價與財報中計算。
        """
        symbol = symbol.upper()
        # 本地快取我們先存放在與 financials 相同的 raw 表，但 item_name 標為比率名
        # statement_type 可以設為一個特殊的值比如 'IS'，但我們可以直接將比率對應到一個獨立的項目
        # 為防衝突，我們可以將 statement_type 設為 'IS'（比率通常也是損益相關的指標，或自定義）
        db_stmt_type = "IS" 
        
        # 1. 查詢快取
        cached_df = self._get_cached_ratios(symbol, ratio_type)
        if not cached_df.empty:
            return cached_df.to_dict(orient="records")
            
        # 2. 嘗試爬取 Macrotrends Ratios 網頁
        slug = get_slug(symbol)
        ratio_url_map = {
            "pe-ratio": "pe-ratio",
            "price-book": "price-book",
            "debt-equity": "debt-equity",
            "margins": "profit-margins" # Macrotrends 上通常是 profit-margins
        }
        url_metric = ratio_url_map.get(ratio_type, ratio_type)
        url = f"https://www.macrotrends.net/stocks/charts/{symbol}/{slug}/{url_metric}"
        
        logger.info(f"Navigating to Macrotrends Ratio: {url}")
        raw_data = scrape_macrotrends_via_selenium(url)
        
        df_parsed = pd.DataFrame()
        if raw_data:
            df_parsed = self._parse_macrotrends_raw_json(symbol, raw_data, db_stmt_type, is_ratio=True, ratio_name=ratio_type)
            
        # 3. Fallback 到 yfinance 計算比率
        if df_parsed.empty:
            logger.warning(f"Macrotrends ratio scrape failed for {symbol}. Falling back to yfinance estimation...")
            df_parsed = self._estimate_ratios_via_yfinance(symbol, ratio_type)
            
        # 4. 寫入快取與回傳
        if not df_parsed.empty:
            self.op.bulk_upsert_raw_financials(df_parsed, market="us")
            cached_df = self._get_cached_ratios(symbol, ratio_type)
            return cached_df.to_dict(orient="records")
            
        return []

    def _get_cached_financials(self, symbol: str, stmt_type: str) -> pd.DataFrame:
        query = """
            SELECT symbol, year, quarter, statement_type, item_name, amount 
            FROM financial_raw_us 
            WHERE symbol = :sym AND statement_type = :stmt
            ORDER BY year DESC, quarter DESC
        """
        try:
            with self.op.engine.connect() as conn:
                df = pd.read_sql(text(query), con=conn, params={"sym": symbol, "stmt": stmt_type})
            return df
        except Exception as e:
            logger.error(f"Read cache financials failed: {e}")
            return pd.DataFrame()

    def _get_cached_ratios(self, symbol: str, ratio_type: str) -> pd.DataFrame:
        # 比率我們以 item_name = ratio_type 快取在表中，quarter 設為 4 (年度) 或對應季度
        query = """
            SELECT symbol, CONCAT(year, '-', LPAD(quarter*3, 2, '0'), '-30') as date, item_name as ratio_name, amount as value
            FROM financial_raw_us
            WHERE symbol = :sym AND item_name = :ratio
            ORDER BY year DESC, quarter DESC
        """
        try:
            with self.op.engine.connect() as conn:
                df = pd.read_sql(text(query), con=conn, params={"sym": symbol, "ratio": ratio_type})
            return df
        except Exception as e:
            logger.error(f"Read cache ratios failed: {e}")
            return pd.DataFrame()

    def _parse_macrotrends_raw_json(self, symbol: str, raw_json: list, stmt_type: str, is_ratio: bool = False, ratio_name: str = "") -> pd.DataFrame:
        """解析 Macrotrends 的 originalData 格式並轉為 DataFrame"""
        all_items = []
        for row in raw_json:
            field_html = row.get("field_name", "")
            # 移除 HTML 標籤取得純文字
            item_name = re.sub(r'<[^>]*>', '', field_html).strip()
            if not item_name:
                continue
                
            if is_ratio:
                # 假如是比率網頁，只取對應的比率名稱
                item_name = ratio_name
                
            for key, val in row.items():
                if key in ["field_name", "popup_icon"]:
                    continue
                # Key 通常是日期 YYYY-MM-DD
                if re.match(r'^\d{4}-\d{2}-\d{2}$', key):
                    try:
                        date_parts = key.split("-")
                        year = int(date_parts[0])
                        month = int(date_parts[1])
                        # 簡單映射月份到季度
                        quarter = (month - 1) // 3 + 1
                        amount_val = float(val) if val is not None and str(val).strip() != "" else 0.0
                        
                        all_items.append({
                            "symbol": symbol,
                            "year": year,
                            "quarter": quarter,
                            "statement_type": stmt_type,
                            "item_name": item_name,
                            "amount": amount_val
                        })
                    except ValueError:
                        continue
        return pd.DataFrame(all_items)

    def _fetch_via_yfinance(self, symbol: str, stmt_type: str, frequency: str) -> pd.DataFrame:
        """透過 yfinance 獲取數據"""
        try:
            ticker = yf.Ticker(symbol)
            if stmt_type == "income-statement":
                df_yf = ticker.quarterly_financials if frequency == "quarterly" else ticker.financials
                stmt_code = "IS"
            elif stmt_type == "balance-sheet":
                df_yf = ticker.quarterly_balance_sheet if frequency == "quarterly" else ticker.balance_sheet
                stmt_code = "BS"
            elif stmt_type == "cash-flow":
                df_yf = ticker.quarterly_cashflow if frequency == "quarterly" else ticker.cashflow
                stmt_code = "CF"
            else:
                return pd.DataFrame()
                
            if df_yf is None or df_yf.empty:
                return pd.DataFrame()
                
            all_items = []
            for item in df_yf.index:
                row = df_yf.loc[item]
                for date_col, val in row.items():
                    if pd.isna(val):
                        continue
                    # date_col 通常是 Timestamp
                    year = date_col.year
                    month = date_col.month
                    quarter = (month - 1) // 3 + 1
                    
                    all_items.append({
                        "symbol": symbol,
                        "year": int(year),
                        "quarter": int(quarter),
                        "statement_type": stmt_code,
                        "item_name": str(item),
                        "amount": float(val)
                    })
            return pd.DataFrame(all_items)
        except Exception as e:
            logger.error(f"yfinance Fallback failed for {symbol}: {e}")
            return pd.DataFrame()

    def _estimate_ratios_via_yfinance(self, symbol: str, ratio_type: str) -> pd.DataFrame:
        """當爬不到比率時，用 yfinance 歷史數據與股價計算/估計歷史比率"""
        try:
            ticker = yf.Ticker(symbol)
            # 獲取歷史價格
            hist = ticker.history(period="5y")
            if hist.empty:
                return pd.DataFrame()
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
                
            # 獲取財報
            financials = ticker.financials # 年報
            if financials.empty:
                return pd.DataFrame()
                
            # 將比率估算為 DataFrame
            all_items = []
            
            if ratio_type == "pe-ratio":
                # PE Ratio = Price / EPS
                # 尋找 EPS
                eps_row = None
                for idx in financials.index:
                    if "diluted eps" in idx.lower() or "basic eps" in idx.lower() or "earnings per share" in idx.lower():
                        eps_row = financials.loc[idx]
                        break
                if eps_row is not None:
                    for date_col, eps_val in eps_row.items():
                        if pd.isna(eps_val) or eps_val == 0:
                            continue
                        # 找到該日期的收盤價
                        date_str = date_col.strftime('%Y-%m-%d')
                        if date_col in hist.index:
                            price = hist.loc[date_col]["Close"]
                        else:
                            # 尋找最接近的一天
                            price = hist.asof(date_col)["Close"] if hasattr(hist, "asof") else hist["Close"].iloc[0]
                        pe = float(price / eps_val)
                        
                        all_items.append({
                            "symbol": symbol,
                            "year": int(date_col.year),
                            "quarter": 4,
                            "statement_type": "IS",
                            "item_name": ratio_type,
                            "amount": pe
                        })
            elif ratio_type == "price-book":
                # Price to Book
                # 尋找 Book Value (Total Assets - Total Liabilities 或 Stockholders Equity)
                bs = ticker.balance_sheet
                shares = ticker.info.get("sharesOutstanding", 0)
                equity_row = None
                for idx in bs.index:
                    if "stockholders equity" in idx.lower() or "total equity" in idx.lower():
                        equity_row = bs.loc[idx]
                        break
                if equity_row is not None and shares > 0:
                    for date_col, equity_val in equity_row.items():
                        if pd.isna(equity_val):
                            continue
                        bps = equity_val / shares
                        if bps == 0:
                            continue
                        price = hist.asof(date_col)["Close"] if hasattr(hist, "asof") else hist["Close"].iloc[0]
                        pb = float(price / bps)
                        
                        all_items.append({
                            "symbol": symbol,
                            "year": int(date_col.year),
                            "quarter": 4,
                            "statement_type": "IS",
                            "item_name": ratio_type,
                            "amount": pb
                        })
            # 如果是其他比率或沒計算出結果，則直接使用 yfinance.info 的最新值
            if not all_items:
                info = ticker.info
                val = 0.0
                if ratio_type == "pe-ratio":
                    val = info.get("trailingPE") or info.get("forwardPE") or 0.0
                elif ratio_type == "price-book":
                    val = info.get("priceToBook") or 0.0
                elif ratio_type == "debt-equity":
                    val = (info.get("debtToEquity") or 0.0) / 100.0 # 轉換為比率
                elif ratio_type == "margins":
                    val = info.get("profitMargins") or 0.0
                
                # 快取最新一年的資料
                current_year = time.localtime().tm_year
                all_items.append({
                    "symbol": symbol,
                    "year": current_year,
                    "quarter": 4,
                    "statement_type": "IS",
                    "item_name": ratio_type,
                    "amount": float(val)
                })
                
            return pd.DataFrame(all_items)
        except Exception as e:
            logger.error(f"Estimate ratios via yfinance failed: {e}")
            return pd.DataFrame()

# Use text from sqlalchemy safely
from sqlalchemy import text
