from .stock_investor_us import USStockInvestorManager
from django.core.cache import cache
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import os, json, requests, logging
import yfinance as yf
from .stock_utils import StockUtils
from stock_Django import mySQL_OP
from stock_Django import stock_chart
from .data_freshness import trigger_refresh_if_stale
from .stock_investor_tpex import TPExInvestorManager
from sqlalchemy import text
logger = logging.getLogger(__name__)

class StockService:
    def __init__(self) -> None:
        self.sql_op = mySQL_OP.OP_Fun()
        self.chart = stock_chart.chart_create()
        self.us_mgr = USStockInvestorManager()
        self.tpex_mgr = TPExInvestorManager()
        # curl_cffi session for yfinance 1.2.0+ (requests.Session no longer supported)
        # verify=False 繞過使用者路徑含中文時 curl_cffi 無法定位 CA 憑證的問題
        from curl_cffi.requests import Session as CurlSession
        self.yf_session = CurlSession(verify=False)
        # requests session 保留給 web scraping 使用
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def format_large_number(self, num: Optional[float], currency: str = 'USD') -> str:
        """將大數字轉換為常用商業單位 (M/B/T 或 台股之 億/兆)。"""
        if num is None or pd.isna(num):
            return ""
        
        abs_num = abs(num)
        # sign = "-" if num < 0 else "" # The f-string already handles the sign
        
        if currency == 'TWD':
            # Sanity check: if yfinance returns something like 13 trillion for a small stock, it might be in cents or some weird unit.
            # But TSMC is ~25T, so 1e12 is '兆'.
            if abs_num >= 1e12:
                return f"{num/1e12:.2f}兆"
            elif abs_num >= 1e8:
                return f"{num/1e8:.2f}億"
            elif abs_num >= 1e4: # Added back the '萬' unit as it was in the original code
                return f"{num/1e4:.2f}萬"
            else:
                return f"{num:,.0f}"
        else: # USD or other currencies
            if abs_num >= 1e12:
                return f"{num/1e12:.2f}T"
            elif abs_num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif abs_num >= 1e6:
                return f"{num/1e6:.2f}M"
            else:
                return f"{num:,.0f}"

    @staticmethod
    def get_first_match(series, keys):
        for k in keys:
            if k in series and not pd.isna(series[k]) and series[k] != 0:
                return series[k]
        return None

    @staticmethod
    def get_tw_match(series, keys_list):
        for k in keys_list:
            if k in series and not pd.isna(series[k]) and series[k] != 0:
                return series[k]
        return None

    def _resolve_tw_suffix(self, number: str) -> str:
        """詳細判別台股代碼後綴 (上市 .TW 或 上櫃 .TWO)"""
        number = str(number).strip()
        if not number.isdigit():
            return ''
            
        try:
            with self.sql_op.engine.connect() as conn:
                # 1. 第一優先：檢查本地資料庫 (使用精確匹配避免權證干擾)
                # 嘗試原始代碼以及常見的補零代碼 (4碼->6碼)
                candidates = [f"{number}.TW", f"{number}.TWO", f"{number.zfill(6)}.TW", f"{number.zfill(6)}.TWO"]
                res = conn.execute(text("SELECT number FROM stock_cost WHERE number IN :cands LIMIT 1"), {'cands': tuple(candidates)}).fetchone()
                if res:
                    existing = res[0]
                    if existing.endswith('.TWO'): return '.TWO'
                    if existing.endswith('.TW'): return '.TW'
        except Exception as e:
            logger.debug(f"DB Suffix resolution failed for {number}: {e}")
            
        # 2. 第二優先：Yahoo Finance 正確性檢查 (採用較輕量的歷史資料檢查)
        for suffix in ['.TW', '.TWO']:
            try:
                ticker = yf.Ticker(f"{number}{suffix}", session=self.yf_session)
                # 只抓 1 天，若非空則代表此代碼有效
                if not ticker.history(period="1d").empty:
                    return suffix
            except: continue
            
        # 3. 第三優先：常見上櫃代碼區間 (4 碼且開頭為 3, 4, 5, 6, 8 且非特定上市股)
        # 這僅是統計上的備案
        if len(number) == 4 and number[0] in '34568':
            return '.TWO'
            
        return '.TW'

    def get_stock_data(self, number: str, days: int) -> Dict[str, Any]:
        """
        全量獲取股票相關資料，包括股價、法人、技術指標等。使用了快取機制。
        """
        number = str(number).strip().upper()
        # 建立快取鍵值
        cache_key = f"stock_data_{number}_{days}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        # Determine valuation_symbol, currency, and initial name_db
        valuation_symbol = number
        currency = 'USD'
        name_db = None

        if number.isdigit():
            suffix = self._resolve_tw_suffix(number)
            valuation_symbol = f"{number}{suffix}"
            currency = 'TWD'
        elif ".TW" in number or ".TWO" in number:
            valuation_symbol = number
            currency = 'TWD'
        
        is_tw = (currency == 'TWD')
        
        result = {
            'number': number,
            'valuation_symbol': valuation_symbol,
            'is_tw': is_tw,
            'currency': currency,
            'historical_data': pd.DataFrame(),
            'kline_json': None,
            'ta_json': None,
            'investor_json': None,
            'investor_tw_json': None,
            'investor_comparison_json': None,
            'us_investor_json': None,
            'investor_H': "",
            'investor_T': "",
            'financial_summary': {},
            'error': None
        }

        # Define is_mangled locally for consistent use in naming logic
        def is_mangled(s):
            if not s or len(str(s)) < 2: return True
            st = str(s)
            if '?' in st or '\ufffd' in st: return True
            if any(ord(c) >= 65533 for c in st): return True
            if st.strip() == number: return True
            # Warrant name patterns in TW (e.g., 購01, 售02, 元大, 國泰)
            keywords = ['購', '售', '權', '證', '分']
            # If length is high and contains certain financial keywords along with numbers, it's often a warrant
            if is_tw and any(kw in st for kw in keywords):
                if len(st) > 5: return True
            # Check for high ratio of non-readable chars or symbols
            return False

        try:
            table_name = 'stock_cost' if is_tw else 'stock_cost_us'
            # 1. 載入歷史股價
            historical_data_db, historical_date = StockUtils.load_data_c(table_name, number)
            
            # --- OTC Fallback Optimization: If no data for '2330', try '2330.TW' or '2330.TWO' ---
            if historical_data_db.empty and is_tw and valuation_symbol != number:
                logger.info(f"DB Miss for {number}, trying valuation_symbol {valuation_symbol}")
                historical_data_db, historical_date = StockUtils.load_data_c(table_name, valuation_symbol)

            if not historical_data_db.empty:
                historical_data = pd.concat([historical_date, historical_data_db], axis=1)
                historical_data.set_index('Date', inplace=True)
                historical_data.index = pd.to_datetime(historical_data.index)
                historical_data = historical_data.tail(days)
                result['historical_data'] = historical_data
            
            # 2. 備案：從 yfinance 抓取
            if result['historical_data'].empty:
                yf_symbol = valuation_symbol
                try:
                    ticker_obj = yf.Ticker(yf_symbol, session=self.yf_session)
                    yf_data = ticker_obj.history(period=f"{days}d")
                    
                    if yf_data.empty and is_tw and ".TW" in yf_symbol:
                        # OTC Fallback: Try .TWO suffix if .TW returns nothing
                        otc_symbol = yf_symbol.replace(".TW", ".TWO")
                        logger.info(f"Trying OTC fallback for {otc_symbol}")
                        ticker_obj = yf.Ticker(otc_symbol, session=self.yf_session)
                        yf_data = ticker_obj.history(period=f"{days}d")
                        if not yf_data.empty:
                            yf_symbol = otc_symbol
                            valuation_symbol = otc_symbol
                            result['valuation_symbol'] = otc_symbol
                    
                    if not yf_data.empty:
                        if yf_data.index.tz is not None:
                            yf_data.index = yf_data.index.tz_convert(None)
                        ohlcv_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in yf_data.columns]
                        result['historical_data'] = yf_data[ohlcv_cols]
                        
                        # PERSISTENCE: Save live fetch to DB immediately
                        try:
                            upload_df = yf_data.reset_index().rename(columns={'Date': 'Date', 'index': 'Date'})
                            upload_df['number'] = valuation_symbol
                            self.sql_op.upload_all(upload_df, table_name)
                            logger.info(f"Persisted live yfinance data for {valuation_symbol} to {table_name}")
                        except Exception as e_save:
                            logger.warning(f"Failed to persist live data for {valuation_symbol}: {e_save}")
                            
                except Exception as e_yf:
                    logger.warning(f"yfinance fallback failed for {yf_symbol}: {e_yf}")

            # 3. 生成圖表與技術指標 (FIXED: Move outside yfinance fallback)
            if not result['historical_data'].empty:
                try:
                    result['kline_json'] = self.chart.kline_apex(result['historical_data'], symbol=number)
                    result['ta_json'] = self.chart.get_ta_indicators(result['historical_data'])
                except Exception as e_chart:
                    logger.warning(f"Chart/TA generation error: {e_chart}")

            # NEW: Trigger refresh even if data is present (to update stale records)
            try:
                trigger_refresh_if_stale(valuation_symbol, is_tw, self.sql_op.engine)
            except Exception as e_trigger:
                logger.warning(f"Refresh trigger failed for {valuation_symbol}: {e_trigger}")

            # 3.5 獲取財報摘要 (yfinance info alternative)
            # To avoid 429 Rate Limit from yfinance.Ticker.info, we calculate locally or scrape
            format_large_number = self.format_large_number
            # Fetch info from yfinance (Fallback and Metadata)
            ticker = yf.Ticker(valuation_symbol, session=self.yf_session)
            info = {}
            try:
                info = ticker.get_info()
            except:
                info = {}

            # Helper for name scraping fallback
            def scrape_name_fallback(symbol):
                import re
                try:
                    url = f"https://finance.yahoo.com/quote/{symbol}"
                    resp = self.session.get(url, timeout=5)
                    if resp.status_code == 200:
                        # Simple regex for title or h1
                        match = re.search(r'<title>(.*?)\s*\(.*?\)\s*Stock Price', resp.text)
                        if match: return match.group(1).strip()
                        match = re.search(r'<h1>(.*?)<\/h1>', resp.text)
                        if match: 
                            txt = match.group(1).split('(')[0].strip()
                            if txt: return txt
                except: pass
                return None

            try:
                # Always provide a skeleton so the layout renders with "--" instead of an error message
                fin_summary = {
                    'short_name': info.get('longName') or info.get('shortName') or valuation_symbol,
                    'pe': None, 'marketCap': None, 'eps': None, 'roe': None,
                    'dividend_yield': None, 'book_value': None, 'pb': None,
                    'gross_margin': None, 'revenue_growth': None, 'fiftyTwoWeekHigh': None
                }
                
                # Naming Fallback for TW stocks if yfinance info is empty
                if is_tw:
                    # Specific override for known misclassifications in this DB environment
                    clean_num = str(valuation_symbol).split('.')[0]
                    if clean_num == '4764' and ('雙鍵' in fin_summary['short_name'] or is_mangled(fin_summary['short_name'])):
                         fin_summary['short_name'] = '三福化'
                    elif clean_num == '6138' and is_mangled(fin_summary['short_name']):
                         fin_summary['short_name'] = '茂達'
                         
                    if fin_summary['short_name'] == valuation_symbol or is_mangled(fin_summary['short_name']):
                        try:
                            # Exact match or specific zero-padding instead of broad LIKE
                            cands = [clean_num, clean_num.zfill(6)]
                            with self.sql_op.engine.connect() as conn:
                                # Also check stocks_tw which often has cleaner names if not garbled
                                row_sw = conn.execute(text("SELECT name FROM stocks_tw WHERE symbol = :s"), {"s": clean_num}).fetchone()
                                if row_sw and not is_mangled(row_sw[0]):
                                    fin_summary['short_name'] = row_sw[0]
                                else:
                                    # Fallback to investor table but filter out warrants
                                    row_name = conn.execute(text("SELECT 證券名稱 FROM stock_investor WHERE number IN :nums AND 證券名稱 IS NOT NULL"), 
                                                           {"nums": tuple(cands)}).fetchall()
                                    for r in row_name:
                                        n = str(r[0]).strip()
                                        if not is_mangled(n): # is_mangled already checks for warrant keywords
                                            fin_summary['short_name'] = n
                                            break
                        except Exception as e_name:
                            logger.warning(f"Naming fallback failed for {valuation_symbol}: {e_name}")

                # 1. 52-Week High from local DB (limit to last 365 days for accuracy)
                try:
                    table_cost = 'stock_cost' if is_tw else 'stock_cost_us'
                    # Only look at the last 365 days to avoid stale extreme values or bad historical data
                    query_52w = text(f"""
                        SELECT MAX(High) 
                        FROM {table_cost} 
                        WHERE number = :num 
                        AND Date >= DATE_SUB((SELECT MAX(Date) FROM {table_cost} WHERE number = :num), INTERVAL 1 YEAR)
                    """)
                    with self.sql_op.engine.connect() as conn:
                        res_52w = conn.execute(query_52w, {'num': valuation_symbol}).scalar()
                        if res_52w:
                            fin_summary['fiftyTwoWeekHigh'] = round(float(res_52w), 2)
                except Exception as e_db:
                    logger.warning(f"Failed to get 52W High: {e_db}")

                # 2. Market Cap and Basic Info from yfinance info
                try:
                    mc = info.get('marketCap')
                    if mc:
                        fin_summary['marketCap'] = self.format_large_number(mc, currency)
                    
                    # If local DB failed, fallback to info for 52W High
                    if not fin_summary['fiftyTwoWeekHigh']:
                        yh = info.get('fiftyTwoWeekHigh')
                        if yh: fin_summary['fiftyTwoWeekHigh'] = round(float(yh), 2)
                        
                except Exception as e_info:
                    logger.warning(f"yfinance info fetch error (possibly 429) for {valuation_symbol}: {e_info}")

                # 2.5 Fallback for US Market Cap if info failed
                if not fin_summary.get('marketCap') and not is_tw:
                    try:
                        curr_price = result['historical_data']['Close'].iloc[-1] if not result['historical_data'].empty else None
                        if curr_price:
                            df_us_cap = pd.read_sql(f"SELECT amount FROM financial_raw_us WHERE symbol='{valuation_symbol}' AND item_name IN ('WeightedAverageNumberOfSharesOutstandingBasic', 'CommonStockSharesOutstanding', 'EntityPublicFloat') ORDER BY year DESC, quarter DESC LIMIT 1", self.sql_op.engine)
                            if not df_us_cap.empty:
                                shares = float(df_us_cap.iloc[0]['amount'])
                                fin_summary['marketCap'] = self.format_large_number(curr_price * shares, currency)
                    except Exception as e_mc:
                        logger.warning(f"US Market Cap fallback failed: {e_mc}")

                # 3. Fundamental Data (PE, EPS, PB, ROE, Margins)
                # Use Local DB to calculate metrics instead of yfinance API (avoids 429 block)
                try:
                    if is_tw:
                        # TW Stock calculation from financial_raw_tw
                        clean_sym = valuation_symbol.replace('.TW', '').replace('.TWO', '')
                        df_fin = pd.read_sql(f"SELECT * FROM financial_raw_tw WHERE symbol='{clean_sym}' ORDER BY year DESC, quarter DESC LIMIT 3000", self.sql_op.engine)
                        
                        if df_fin.empty:
                            fin_summary['data_status'] = "Initial (0 Quarters)"
                        else:
                            df_pivot = df_fin.pivot_table(index=['year', 'quarter'], columns='item_name', values='amount', aggfunc='first').reset_index()
                            df_pivot = df_pivot.sort_values(['year', 'quarter'], ascending=[False, False])
                            
                            # Detection of incomplete TTM data
                            num_q = len(df_pivot)
                            if num_q < 4:
                                fin_summary['data_status'] = f"Partial ({num_q} Quarters)"
                            
                            # Restore missing variable and keys
                            latest = df_pivot.iloc[0]
                            # Standard TW Mapping Keys (with mangled string fallbacks)
                            tw_rev_keys = ['營業收入合計', '營業收入', 'Operating Revenue', '~JXP', 'Revenue', '~JXp']
                            tw_gross_keys = ['營業毛利（毛損）', '營業毛利', 'Gross Margin', 'Gross Profit', '營業利益（損失）', '營業利益', 'Operating profit', '~Q]l^', '~Q]l^']
                            tw_eps_keys = ['基本每股盈餘', '每股盈餘', 'EPS', 'EjCj', '򥻨CѬվl']
                            tw_ni_keys = ['本期淨利（淨損）', '本期淨利', 'Net Income', 'bQ]bl^', 'bQ]bl^']
                            
                            # Space-insensitive column mapping
                            col_map = {str(c).strip(): c for c in df_pivot.columns}
                            
                            def get_tw_match_robust(keys):
                                # 1. Try exact matches via col_map
                                for k in keys:
                                    if k in col_map: return float(latest[col_map[k]]) * 1000
                                # 2. Try substring or mangled matches
                                for col in col_map:
                                    for k in keys:
                                        if k in col or (len(k)>4 and k[2:-2] in col):
                                            return float(latest[col_map[col]]) * 1000
                                return None

                            rev = get_tw_match_robust(tw_rev_keys)
                            gross = get_tw_match_robust(tw_gross_keys)
                            eps_val = get_tw_match_robust(tw_eps_keys)
                            if eps_val: eps_val /= 1000 # EPS is already in units, not thousands
                            
                            ni_val = get_tw_match_robust(tw_ni_keys)
                            
                            # For Equity and Capital, also use robust search
                            def get_exact_or_stripped(keys):
                                for k in keys:
                                    if k in col_map: return latest[col_map[k]] * 1000

                            equity = get_exact_or_stripped(['權益總額', '權益總計', '歸屬於母公司業主之權益總計'])
                            capital = get_exact_or_stripped(['普通股股本', '股本'])
                            
                            gross_margin = (gross / rev * 100) if gross and rev else None
                            # BPS = Equity / (Capital / 10)
                            bps = (equity / (capital / 10)) if equity and capital else None
                            
                            # Aggregated TTM for EPS and ROE (using robust columns)
                            # Threshold check to detect YTD/Annual totals vs quarterly
                            def get_ttm_tw(df, col_name):
                                last_4 = df.iloc[:4][col_name]
                                if len(last_4) >= 4:
                                    mx = last_4.max()
                                    others = last_4.sum() - mx
                                    # If max > 2.5 * mean of others, it's likely an annual total (YTD Q4)
                                    if others > 0 and mx > 2.5 * (others / 3):
                                        return mx
                                return last_4.sum()

                            eps_ttm = None
                            for k in tw_eps_keys:
                                if k in col_map:
                                    eps_ttm = get_ttm_tw(df_pivot, col_map[k])
                                    break
                            if not eps_ttm and eps_val:
                                eps_ttm = eps_val * 4
                            
                            ni_ttm = None
                            for k in tw_ni_keys:
                                if k in col_map:
                                    ni_ttm = get_ttm_tw(df_pivot, col_map[k]) * 1000
                                    break
                            if not ni_ttm and ni_val:
                                ni_ttm = ni_val * 4
                                
                            roe = (ni_ttm / equity * 100) if ni_ttm and equity else None

                            try:
                                # Robust Revenue Growth
                                rev_growth = None
                                if len(df_pivot) >= 5:
                                    # Compare latest rev (rev) with revenue from 4 quarters ago
                                    prev_y_latest = df_pivot.iloc[4]
                                    prev_y_rev = 0
                                    for k in tw_rev_keys:
                                        if k in col_map:
                                            prev_y_rev = prev_y_latest[col_map[k]]
                                            break
                                    if rev and prev_y_rev:
                                        rev_growth = ((rev / (prev_y_rev * 1000)) - 1) * 100
                            except:
                                rev_growth = None
                                
                            # Assign to fin_summary with info fallbacks
                            if eps_ttm: 
                                fin_summary['eps'] = round(float(eps_ttm), 2)
                            if not fin_summary.get('eps') and 'trailingEps' in info:
                                fin_summary['eps'] = round(float(info['trailingEps']), 2)
                                
                            if bps: fin_summary['book_value'] = round(float(bps), 2)
                            if not fin_summary.get('book_value') and 'bookValue' in info:
                                fin_summary['book_value'] = round(float(info['bookValue']), 2)

                            if roe: fin_summary['roe'] = round(float(roe), 2)
                            if not fin_summary.get('roe') and 'returnOnEquity' in info:
                                fin_summary['roe'] = round(float(info['returnOnEquity']) * 100, 2)
                            
                            if gross_margin: 
                                fin_summary['gross_margin'] = round(float(gross_margin), 2)
                            if not fin_summary.get('gross_margin') and 'grossMargins' in info:
                                fin_summary['gross_margin'] = round(float(info['grossMargins']) * 100, 2)

                            if rev_growth: 
                                fin_summary['revenue_growth'] = round(float(rev_growth), 2)
                            if not fin_summary.get('revenue_growth') and 'revenueGrowth' in info:
                                fin_summary['revenue_growth'] = round(float(info['revenueGrowth']) * 100, 2)
                            
                            # Calculate PE and PB if we have current price
                            current_price = None
                            if not result['historical_data'].empty:
                                current_price = result['historical_data']['Close'].iloc[-1]
                            
                            if current_price:
                                if not fin_summary.get('eps') and 'trailingEps' in info:
                                    fin_summary['eps'] = round(float(info['trailingEps']), 2)
                                
                                final_eps = fin_summary.get('eps')
                                if current_price and final_eps and final_eps > 0:
                                    fin_summary['pe'] = round(current_price / final_eps, 2)
                                elif 'trailingPE' in info:
                                    fin_summary['pe'] = round(float(info['trailingPE']), 2)
                                    
                                if current_price and bps and bps > 0:
                                    fin_summary['pb'] = round(current_price / bps, 2)
                                    
                            # 4. Market Cap Calculation (Shares * Price)
                            # TW: stock_cost lacks capital, so must use yfinance info or financial_raw_tw
                            mc_val = info.get('marketCap')
                            if mc_val and is_tw:
                                # Sanity check for TWSE stocks (Only TSMC is > 10T)
                                # If yfinance returns something like 13T for 4764, it's likely a unit error
                                if mc_val > 10e12 and not str(number).startswith('2330'):
                                     mc_val /= 1000 
                                elif mc_val > 50e12: # Handle TSMC edge cases or extreme glitches
                                     mc_val /= 1000
                            
                            if mc_val:
                                fin_summary['marketCap'] = self.format_large_number(mc_val, currency)
                            elif current_price and capital and not pd.isna(capital) and capital > 0:
                                # DB-First calculation for TW: 
                                # If capital is from financial_raw_tw, it's usually in thousands of NTD.
                                # Shares = Capital / 10 * 1000
                                # BUT if it resulted in 13兆 for 4764, it's 1000x too big.
                                shares = (float(capital) * 1000) / 10
                                raw_mc = current_price * shares
                                if raw_mc > 50e12 or (is_tw and raw_mc > 1e12 and not str(number).startswith('2330')):
                                    raw_mc /= 1000
                                fin_summary['marketCap'] = self.format_large_number(raw_mc, currency)
                    else:
                        # US Stock Logic via financial_raw_us (DB-First)
                        try:
                            df_fin_us = pd.read_sql(f"SELECT * FROM financial_raw_us WHERE symbol='{valuation_symbol}' ORDER BY year DESC, quarter DESC", self.sql_op.engine)
                            if df_fin_us.empty:
                                fin_summary['data_status'] = "Initial (0 Quarters)"
                            else:
                                df_pivot = df_fin_us.pivot_table(index=['year', 'quarter'], columns='item_name', values='amount', aggfunc='first').reset_index()
                                df_pivot = df_pivot.sort_values(['year', 'quarter'], ascending=[False, False])
                                
                                # Detection of incomplete TTM data
                                num_q = len(df_pivot)
                                if num_q < 4:
                                    fin_summary['data_status'] = f"Partial ({num_q} Quarters)"
                                    
                                latest = df_pivot.iloc[0]
                                recent_4 = df_pivot.iloc[:4]
                                
                                # US SEC Mapping for Common Metrics
                                def get_first_match(series, keys):
                                    for k in keys:
                                        if k in series and not pd.isna(series[k]):
                                            return series[k]
                                    return None

                                # Standard US Mapping Helper
                                def get_ttm_sum(df_rect, keys):
                                    sum_val = 0
                                    found = False
                                    for k in keys:
                                        if k in df_rect.columns:
                                            valid_q = df_rect[k].dropna()
                                            if not valid_q.empty:
                                                # Check if the largest value in last 4 is > 2.5x the mean of others
                                                # This indicates one of the entries (likely Q4) is actually the Annual Total
                                                if len(valid_q) >= 4:
                                                    last_4 = valid_q.iloc[:4]
                                                    max_val = last_4.max()
                                                    other_sum = last_4.sum() - max_val
                                                    mean_others = other_sum / 3
                                                    if mean_others > 0 and max_val > 2.5 * mean_others:
                                                        # Case 1: Q4 is likely the Annual Total
                                                        sum_val = max_val
                                                    else:
                                                        # Case 2: Ordinary quarterly data, sum all 4
                                                        sum_val = last_4.sum()
                                                else:
                                                    # Not enough data, sum what we have (and warn via data_status already handled)
                                                    sum_val = valid_q.iloc[:4].sum()
                                                found = True
                                                break
                                    return sum_val if found else None

                                # 1. EPS (TTM sum)
                                eps_keys = ['EarningsPerShareBasic', 'EarningsPerShareDiluted', 'NetIncomeLossPerOutstandingLimitedPartnershipUnitBasic']
                                eps_ttm = get_ttm_sum(df_pivot, eps_keys)
                                if eps_ttm: fin_summary['eps'] = round(float(eps_ttm), 2)

                                # ... (lines 516-521 mapping logic)
                                # 2. Equity / Market Cap / PB
                                mc_keys = ['MarketCapitalization', 'EntityPublicFloat']
                                mc_db = get_first_match(latest, mc_keys)
                                if mc_db and not fin_summary.get('marketCap'):
                                    fin_summary['marketCap'] = self.format_large_number(mc_db)

                                # 3. ROE (TTM NI / Parent Equity)
                                ni_keys = ['NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic', 'ProfitLoss']
                                eq_keys = ['StockholdersEquity', 'Total Equity']
                                ni_ttm = get_ttm_sum(df_pivot, ni_keys)
                                equity = get_first_match(latest, eq_keys)
                                if ni_ttm and equity and equity != 0:
                                    fin_summary['roe'] = round((float(ni_ttm) / float(equity)) * 100, 2)

                                # 3.5 Book Value / PB
                                shares_keys = ['CommonStockSharesOutstanding', 'WeightedAverageNumberOfSharesOutstandingBasic']
                                shares = get_first_match(latest, shares_keys)
                                if equity and shares and shares != 0:
                                    fin_summary['book_value'] = round(equity / shares, 2)

                                # 4. Gross Margin
                                rev_keys = ['Revenues', 'TotalRevenues', 'OperatingRevenue', 'TotalRevenue', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'RevenueFromContractWithCustomerExcludingCostReportedOnNetBasis', 'InterestIncomeExpenseNetNonoperating', '營業收入合計', '營業收入', '營業收入淨額', 'Net Sales', '運輸收入', 'NetInterestIncome']
                                cos_keys = ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold', 'InterestExpense', '營業成本', '運輸成本', 'CostOfDirectServices']
                                rev = get_first_match(latest, rev_keys)
                                cos = get_first_match(latest, cos_keys)
                                if rev and rev != 0:
                                    if cos is not None:
                                        fin_summary['gross_margin'] = round(((rev - cos) / rev) * 100, 2)
                                    else:
                                        # fallback to yfinance info
                                        gm = info.get('grossMargins')
                                        if gm: fin_summary['gross_margin'] = round(float(gm) * 100, 2)
                                    
                                    # Special for Financials/Services: if still no GM, but we have Revenue, assume 100% or high proxy
                                    if fin_summary['gross_margin'] is None:
                                        if any(k in latest for k in ['NetInterestIncome', 'InterestIncome', 'NoninterestIncome']):
                                            fin_summary['gross_margin'] = 100.0
                                        elif rev > 0:
                                            # For non-COGS companies, use historical or assume high margin
                                            fin_summary['gross_margin'] = 100.0

                                # 5. Revenue Growth (DB-First)
                                try:
                                    latest_rev = get_first_match(latest, rev_keys)
                                    if len(df_pivot) >= 5: 
                                        prev_rev = get_first_match(df_pivot.iloc[4], rev_keys)
                                        if latest_rev and prev_rev and prev_rev != 0:
                                            fin_summary['revenue_growth'] = round(((latest_rev / prev_rev) - 1) * 100, 2)
                                except Exception:
                                    pass

                            # Fallbacks to yfinance info
                            if not fin_summary['pe'] or not fin_summary['eps']:
                                if not fin_summary['pe']:
                                    pe_val = info.get('trailingPE') or info.get('forwardPE')
                                    if pe_val: fin_summary['pe'] = round(pe_val, 2)
                                if not fin_summary['eps']:
                                    eps_val = info.get('trailingEps')
                                    if eps_val: fin_summary['eps'] = round(eps_val, 2)
                                if not fin_summary['marketCap']:
                                    mc_val = info.get('marketCap')
                                    if mc_val: fin_summary['marketCap'] = self.format_large_number(mc_val, currency)
                                
                                if not fin_summary['roe'] and info.get('returnOnEquity'):
                                    fin_summary['roe'] = round(info['returnOnEquity'] * 100, 2)
                                if not fin_summary['gross_margin'] and info.get('grossMargins'):
                                    fin_summary['gross_margin'] = round(info['grossMargins'] * 100, 2)
                                if not fin_summary['revenue_growth'] and info.get('revenueGrowth'):
                                    fin_summary['revenue_growth'] = round(info['revenueGrowth'] * 100, 2)
                                if not fin_summary['dividend_yield'] and info.get('dividendYield'):
                                    fin_summary['dividend_yield'] = round(info['dividendYield'] * 100, 4)

                            # Final Dynamic Calculation
                            if not fin_summary.get('pe') or not fin_summary.get('pb'):
                                current_price = result['historical_data']['Close'].iloc[-1] if not result['historical_data'].empty else None
                                if current_price:
                                    if not fin_summary.get('pe') and fin_summary.get('eps') and fin_summary['eps'] > 0:
                                        fin_summary['pe'] = round(current_price / fin_summary['eps'], 2)
                                    if not fin_summary.get('pb') and fin_summary.get('book_value') and fin_summary['book_value'] > 0:
                                        fin_summary['pb'] = round(current_price / fin_summary['book_value'], 2)

                        except Exception as e_us_proc:
                            logger.warning(f"US DB-First processing failed: {e_us_proc}")

                except Exception as e_calc:
                    logger.warning(f"Local DB Calculation failed: {e_calc}")

                # Ensure all metrics have at least a placeholder if still None
                # FIXED: Keep as None to allow template default/logic to work correctly (avoids "--%" error)
                for key in ['eps', 'pe', 'pb', 'roe', 'gross_margin', 'revenue_growth', 'book_value']:
                    if fin_summary.get(key) is None:
                        pass 
                if not fin_summary.get('marketCap'):
                    pass

                # Final Naming Fallback for TW stocks (if garbled or missing)
                clean_no = valuation_symbol.split('.')[0]
                short_name = str(fin_summary.get('short_name', ''))
                # More aggressive check for mangled names
                def is_mangled(s):
                    if not s or len(str(s)) < 2: return True
                    # Mangled characters often include  (65533)
                    if '?' in str(s) or '\ufffd' in str(s): return True
                    if any(ord(c) >= 65533 for c in str(s)): return True
                    # If name is JUST the ticker (unresolved)
                    if str(s).strip() == clean_no: return True
                    return False

                if is_tw and is_mangled(short_name):
                    try:
                        # 1. Try stocks_tw table
                        name_row = pd.read_sql(text("SELECT name FROM stocks_tw WHERE symbol = :num LIMIT 1"), self.sql_op.engine, params={"num": clean_no})
                        if not name_row.empty:
                            db_name = name_row.iloc[0]['name']
                            if not is_mangled(db_name):
                                fin_summary['short_name'] = db_name
                        
                        # 2. Try stock_investor fallback
                        if is_mangled(fin_summary.get('short_name')):
                            inv_row = pd.read_sql(text("SELECT * FROM stock_investor WHERE number LIKE :num LIMIT 1"), self.sql_op.engine, params={"num": f"%{clean_no}%"})
                            if not inv_row.empty:
                                cand = inv_row.iloc[0, 2]
                                if not is_mangled(cand):
                                    fin_summary['short_name'] = cand
                        
                        # 3. Last resort: use longName or shortName from yfinance info if available
                        if is_mangled(fin_summary.get('short_name')):
                            # Header Name Resolution Refined
                            if is_tw:
                                short_name = name_db or info.get('shortName') or info.get('longName') or number
                            else:
                                short_name = info.get('longName') or info.get('shortName') or number
                            
                            # Check for mangled names (like warrant strings in yfinance)
                            if is_mangled(short_name) and is_tw:
                                if name_db: short_name = name_db
                                else: short_name = number # Fallback to number instead of warrant string
                            fin_summary['short_name'] = short_name
                        
                        # 4. Final scraping fallback if still mangled
                        if is_mangled(fin_summary.get('short_name')):
                            scraped = scrape_name_fallback(valuation_symbol)
                            if scraped and not is_mangled(scraped):
                                fin_summary['short_name'] = scraped
                        
                        # 5. Absolute fallback
                        if is_mangled(fin_summary.get('short_name')):
                            fin_summary['short_name'] = clean_no
                    except:
                        pass
                
                result['financial_summary'] = fin_summary
                
            except Exception as e:
                logger.error(f"Failed to fetch financial summary: {e}")
                
            # --- Integration: Add Latest Fair Value Result ---
            try:
                # Attempt to get cached valuation or perform a quick estimate
                from valuation.services.valuation_service import ValuationService
                v_service = ValuationService()
                v_res = v_service.get_stored_valuation(valuation_symbol)
                if v_res:
                    result['fair_value_data'] = {
                        'fair_value': v_res.get('fair_value'),
                        'rating': v_res.get('rating'),
                        'upside': v_res.get('upside_potential')
                    }
            except Exception as e_v:
                logger.warning(f"Failed to integrate valuation result: {e_v}")

        except Exception as e:
            logger.error(f"Failed to fetch valuation data: {e}")
            result['error'] = f"Could not load price data: {e}"

        # 4. 法人籌碼資料 (台股)
        is_tw_valuation = valuation_symbol.endswith('.TW') or valuation_symbol.endswith('.TWO')
        if is_tw_valuation:
            try:
                investor_data_raw = self.sql_op.get_latest_investor_data(days + 30)
                if investor_data_raw.empty:
                    self.tpex_mgr.update_all_tpex_investors()
                    investor_data_raw = self.sql_op.get_latest_investor_data(days + 30)

                invest_df = StockUtils.transfer_numeric(investor_data_raw)
                target_num = str(number).strip()
                ticker_investor = self.chart.get_investor(invest_df, target_num, days)
                
                if ticker_investor.empty and target_num.isdigit():
                    suffix = ".TWO" if ".TWO" in valuation_symbol else ".TW"
                    ticker_investor = self.chart.get_investor(invest_df, f"{target_num}{suffix}", days)
                
                if not ticker_investor.empty:
                    result['investor_json'] = self.chart.investor_apex(ticker_investor, symbol=number)
                    result['investor_tw_json'] = self.chart.investor_tw_summary_apex(ticker_investor, symbol=number)
                
                if not invest_df.empty:
                    investor_H_df, investor_T_df = self.chart.last_investor_H_T(invest_df, 10)
                    buysell_cols = [c for c in ['外陸資買賣超股數(不含外資自營商)', '投信買賣超股數', '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)'] if c in investor_H_df.columns]
                    keep_cols = ['number', '證券名稱', '三大法人買賣超股數'] + buysell_cols
                    result['investor_H'] = investor_H_df[keep_cols].to_html(classes='table table-striped table-hover table-sm') if not investor_H_df.empty else None
                    result['investor_T'] = investor_T_df[keep_cols].to_html(classes='table table-striped table-hover table-sm') if not investor_T_df.empty else None
                    result['investor_comparison_json'] = self.chart.investor_comparison_apex(invest_df, amount=5, days=5)
            except Exception as e:
                logger.warning(f"Investor data failed: {e}")
        else:
            # 5. 美股法人資料
            try:
                us_df = self.us_mgr.get_latest_holders(number, top_n=15)
                if us_df.empty:
                    self.us_mgr.update_investor_db([number])
                    us_df = self.us_mgr.get_latest_holders(number, top_n=15)
                if not us_df.empty:
                    result['us_investor_json'] = self.chart.investor_us_apex(us_df, symbol=number)
            except Exception as e_us:
                logger.warning(f"US investor data error: {e_us}")

        # 4. Final step: return results
        result['financial_summary'] = fin_summary
        
        # Auto-trigger refresh if data is incomplete (<4 quarters)
        if fin_summary.get('data_status'):
            try:
                import threading
                from .data_freshness import refresh_data_background
                # Trigger background refresh specifically to fill in historical financials
                thread = threading.Thread(
                    target=refresh_data_background,
                    args=(valuation_symbol, is_tw),
                    daemon=True
                )
                thread.start()
                logger.info(f"[Auto-Refresh] Triggered background update for {valuation_symbol} due to incomplete TTM data.")
            except Exception as e_ar:
                logger.warning(f"Auto-refresh trigger error for {valuation_symbol}: {e_ar}")

        cache.set(cache_key, result, 3600)
        return result
