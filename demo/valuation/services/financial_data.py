import pandas as pd
import numpy as np
import logging
import sys
import os
from sqlalchemy import text
import datetime
import requests
import pickle
import time
from pathlib import Path

# Adapt imports for Django environment
from stock_Django.mySQL_OP import OP_Fun
# Adapt imports for Django environment
from stock_Django.mySQL_OP import OP_Fun

logger = logging.getLogger(__name__)

class FinancialDataLoader:
    def __init__(self, ticker_symbol):
        # 判斷市場 (邏輯：包含 .TW 或 純數字 為台灣，否則為美國)
        temp_symbol = ticker_symbol.upper()
        if ".TW" in temp_symbol or ".TWO" in temp_symbol or temp_symbol.isdigit():
            # 台灣市場
            self.market = 'tw'
            self.symbol = temp_symbol.replace(".TW", "").replace(".TWO", "")
            # 定義完整的 yfinance 搜尋代號 (優先使用 .TW 如果是純數字)
            if ".TWO" in temp_symbol:
                self.full_symbol = f"{self.symbol}.TWO"
            else:
                self.full_symbol = f"{self.symbol}.TW"
        else:
            # 美國市場
            self.market = 'us'
            self.symbol = temp_symbol
            self.full_symbol = temp_symbol
            
        self.op = OP_Fun()
        self.raw_data = None
        
        # curl_cffi session for yfinance 1.2.0+ (requests.Session no longer supported)
        from curl_cffi.requests import Session as CurlSession
        self.yf_session = CurlSession(verify=False)
        # requests session 保留給一般 HTTP 請求使用
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        })
        self.cache_dir = Path(__file__).parent / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

    def ensure_data_freshness(self):
        """檢查資料是否為最新，若非最新則自動抓取更新"""
        table_name = f"financial_raw_{self.market}"
        # 取得資料庫中該股最晚的年份與季度，以及總季數
        query = f"SELECT MAX(year * 10 + quarter) as latest_val, COUNT(DISTINCT year * 10 + quarter) as q_count FROM {table_name} WHERE symbol = :symbol"
        with self.op.engine.connect() as conn:
            row = conn.execute(text(query), {"symbol": self.symbol}).fetchone()
            latest_db_val = row[0] if row and row[0] else 0
            q_count = row[1] if row and row[1] else 0
            
        now = datetime.datetime.now()
        cur_year = now.year
        cur_month = now.month
        
        # 簡單判斷邏輯：根據目前月份推算應該要有的最新财報
        if self.market == 'tw':
            if cur_month >= 4: target_year, target_q = cur_year - 1, 4
            if cur_month >= 6: target_year, target_q = cur_year, 1
            if cur_month >= 9: target_year, target_q = cur_year, 2
            if cur_month >= 12: target_year, target_q = cur_year, 3
            if cur_month < 4: target_year, target_q = cur_year - 1, 3
        else:
            if cur_month in [1, 2]: target_year, target_q = cur_year - 1, 3
            elif cur_month in [3, 4, 5]: target_year, target_q = cur_year - 1, 4
            elif cur_month in [6, 7, 8]: target_year, target_q = cur_year, 1
            elif cur_month in [9, 10, 11]: target_year, target_q = cur_year, 2
            else: target_year, target_q = cur_year, 3

        target_val = target_year * 10 + target_q
        
        if latest_db_val < target_val or q_count < 4:
            logger.info(f"Data for {self.symbol} ({self.market}) is stale or incomplete (Latest: {latest_db_val}, Count: {q_count}, Target: {target_val}). Updating...")
            try:
                if self.market == 'us':
                    from stock_Django.scraper_us import scrape_us_financials, get_us_stock_list
                    stocks_info = get_us_stock_list()
                    row = stocks_info[stocks_info['symbol'] == self.symbol]
                    if not row.empty:
                        cik = row.iloc[0]['cik']
                        df = scrape_us_financials(self.symbol, cik)
                        if not df.empty:
                            df_filtered = df[df['year'] >= (cur_year - 5)]
                            self.op.bulk_upsert_raw_financials(df_filtered, market='us')
                else:
                    from stock_Django.scraper_tw_pw import scrape_tw_financials_playwright, scrape_tw_financials_multi
                    if latest_db_val == 0 or q_count < 4:
                        # 全新股票或資料不全：抓取過去 5 年 (20 季)
                        logger.info(f"Incomplete TW stock data detected: {self.symbol} (Count: {q_count}). Fetching history...")
                        periods = []
                        for y in range(cur_year - 5, cur_year + 1):
                            for q in range(1, 5):
                                if (y * 10 + q) <= target_val:
                                    periods.append((y, q))
                        # 限縮數量，抓最近 12 季
                        df = scrape_tw_financials_multi(self.symbol, periods[-12:])
                    else:
                        # 僅更新缺失的一季
                        df = scrape_tw_financials_playwright(self.symbol, target_year, target_q)
                    
                    if not df.empty:
                        self.op.bulk_upsert_raw_financials(df, market='tw')
            except Exception as e:
                logger.error(f"Failed to auto-update {self.symbol}: {e}")

    def get_full_financials(self):
        """從 MySQL 讀取原始數據並轉換為 DataFrame 格式"""
        self.ensure_data_freshness()
        
        table_name = f"financial_raw_{self.market}"
        query = f"SELECT year, quarter, statement_type, item_name, amount FROM {table_name} WHERE symbol = :symbol ORDER BY year DESC, quarter DESC"
        
        with self.op.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"symbol": self.symbol})
        
        if df.empty:
            logger.error(f"No financial data found in local MySQL for {self.symbol}. Please run the scraper first.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Successfully loaded {len(df)} financial items from local MySQL for {self.symbol}")
        
        # 統一單位：台股資料在 MOPS 是「千元」，美股 SEC 通常是「元」
        # 我們統一轉換為「元」(Absolute Units)
        if self.market == 'tw':
            df['amount'] = df['amount'] * 1000
            
        self.raw_data = df
        
        # 轉換為 DCF 需要的格式 (橫表)
        # 對於美股 SEC 資料，標籤通常不重複，我們直接 Pivot 整張表
        if self.market == 'us':
            df['date'] = df['year'].astype(str) + "Q" + df['quarter'].astype(str)
            # US 數據可能有多重標籤，樞紐分析時取第一筆
            pivoted_all = df.pivot_table(index='date', columns='item_name', values='amount', aggfunc='first')
            pivoted_all = pivoted_all.sort_index()
            # 為了相容性，讓 IS, BS, CF 都回傳整張表 (在 extract_projection_start 中 mapping 會處理)
            return pivoted_all, pivoted_all, pivoted_all
            
        dfs = {}
        for st in ['IS', 'BS', 'CF']:
            st_df = df[df['statement_type'] == st].copy()
            if not st_df.empty:
                st_df['date'] = st_df['year'].astype(str) + "Q" + st_df['quarter'].astype(str)
                pivoted = st_df.pivot_table(index='date', columns='item_name', values='amount', aggfunc='first')
                dfs[st] = pivoted.sort_index()
            else:
                dfs[st] = pd.DataFrame()
                
        return dfs.get('IS'), dfs.get('BS'), dfs.get('CF')

    def _get_financials_from_yfinance(self):
        """
        Fallback: fetch financial statements directly from yfinance when DB is empty.
        Returns (is_df, bs_df, cf_df) in the same pivoted format as get_full_financials.
        """
        # Try cache first
        cached = self._load_from_cache('financials')
        if cached:
            return cached['is'], cached['bs'], cached['cf']
            
        try:
            ticker = yf.Ticker(self.full_symbol, session=self.yf_session)
            
            # Fetch annual financials (more stable than quarterly for DCF)
            income_stmt = ticker.financials       # Annual income statement
            balance_sheet = ticker.balance_sheet  # Annual balance sheet
            cash_flow = ticker.cashflow           # Annual cash flow
            
            def transpose_and_rename(df):
                """Transpose yfinance df (items as rows, dates as cols) to (dates as rows, items as cols)"""
                if df is None or df.empty:
                    return pd.DataFrame()
                df_t = df.T.copy()
                # Convert column index to string year labels
                df_t.index = [str(d.year) + "Q4" for d in df_t.index]
                df_t.index.name = 'date'
                return df_t
            
            is_df = transpose_and_rename(income_stmt)
            bs_df = transpose_and_rename(balance_sheet)
            cf_df = transpose_and_rename(cash_flow)
            
            if is_df.empty and bs_df.empty:
                logger.error(f"yfinance returned no financial data for {self.full_symbol}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            logger.info(f"Successfully loaded financial data from yfinance for {self.full_symbol}")
            
            # Save to cache
            self._save_to_cache({'is': is_df, 'bs': bs_df, 'cf': cf_df}, 'financials')
            
            return is_df, bs_df, cf_df
            
        except Exception as e:
            logger.error(f"yfinance financial data fallback failed for {self.full_symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _save_to_cache(self, data, data_type):
        try:
            with open(self.cache_dir / f"{self.full_symbol}_{data_type}.pkl", 'wb') as f:
                pickle.dump({'data': data, 'timestamp': time.time()}, f)
        except: pass

    def _load_from_cache(self, data_type, max_age_hours=24):
        path = self.cache_dir / f"{self.full_symbol}_{data_type}.pkl"
        if not path.exists(): return None
        try:
            with open(path, 'rb') as f:
                cached = pickle.load(f)
            return cached['data'] if (time.time() - cached['timestamp'])/3600 < max_age_hours else None
        except: return None

    def extract_projection_start(self):
        is_df, bs_df, _ = self.get_full_financials()
        if is_df.empty or bs_df.empty: return None
        
        # 計算 TTM (最近四季累計) 對於損益表項目
        # 如果不足 4 季，則進行年化 (例如 2 季 * 2)
        recent_is = is_df.iloc[-4:]
        num_q = len(recent_is)
        multiplier = 4.0 / num_q if num_q > 0 else 4.0
        
        def get_ttm(df: pd.DataFrame, keys: list) -> float:
            if df.empty: return 0.0
            # 重要修正：美股或台股有多個可選標籤時，採取「優先順序填補」，而非全部加總
            # 避免同時加總了 Revenues 和 TotalRevenue 導致數值翻倍
            combined = pd.Series(np.nan, index=df.index)
            for k in keys:
                if k in df.columns:
                    # 使用 fillna 確保優先順序：前面的 key 優先
                    combined = combined.fillna(df[k])
            
            quarterly_sum = combined.fillna(0)
            if quarterly_sum.empty: return 0.0
            
            # 關鍵修正：美股資料中，有些 Q4 或年度項目其實是全年度加總 (例如 COST)
            # 如果最近四期中存在一個顯著巨大的數值 (例如 > 平均值的 2.5 倍)，視為年度數據
            if self.market == 'us' and len(quarterly_sum) >= 4:
                last_4 = quarterly_sum.iloc[-4:]
                max_val = last_4.max()
                mean_others = (last_4.sum() - max_val) / 3
                if mean_others > 0 and max_val > 2.5 * mean_others:
                    # 視為此 max_val 本身就是 TTM/年度數據
                    return max_val
            
            # 判讀是否為年報 (資料間隔約 365 天)
            is_annual = False
            if len(df) >= 2:
                try:
                    # 使用 pandas 的 diff 計算日期差異
                    # 注意：df.index 在此處通常是 YYYYQX 格式字串，轉換為 Datetime 判斷間隔
                    dates = pd.to_datetime([str(i).replace('Q1','-03-31').replace('Q2','-06-30').replace('Q3','-09-30').replace('Q4','-12-31') for i in df.index], errors='coerce')
                    intervals = dates.to_series().diff().dt.days.dropna()
                    if not intervals.empty and intervals.mean() > 300:
                        is_annual = True
                except:
                    # Fallback check
                    is_annual = all(len(str(idx)) == 4 or str(idx).endswith('Q4') for idx in df.index)
            
            if is_annual:
                # 若為年報，取最新一期即代表年度數據
                val = quarterly_sum.iloc[-1]
            else:
                # 若為季報，則加總最近四期 (即 TTM)
                val = quarterly_sum.iloc[-4:].sum()
            
            # 關鍵修正：對於美股，SEC 資料單位通常是 Absolute (1) 或 Millions (1,000,000)
            # 如果數值異常小 (例如營收 < 100 萬且市場是美股)，可能是單位問題
            if self.market == 'us' and val < 1e7: # 假設營收都大於 1000 萬美金
                 # Check if the DB table has very small numbers which implies Million units
                 val *= 1_000_000
            return val

        # 資產負債表取最近一期 (Point in Time)
        last_bs = bs_df.iloc[-1]

        # 定義映射對照表 (包含中文 MOPS 欄位名稱 和 英文 yfinance 欄位名稱)
        mapping = {
            'tw': {
                # yfinance 英文欄位名稱 (fallback) + MOPS 中文欄位名稱 (DB)
                'revenue': ["Total Revenue", "Revenue", "Operating Revenue", "營業收入合計", "營業收入", "營業收入淨額", "Net Sales", "運輸收入"],
                'ebit': ["Operating Income", "EBIT", "Operating Income Loss", "營業利益（損失）", "營業利益", "營業利益淨額", "Operating Profit", "運輸利益"],
                'net_income': ["Net Income", "Net Income Common Stockholders", "本期淨利（負擔）", "本期淨利", "歸屬於母公司業主之本期淨利（損）", "本期淨利（淨損）", "Net Profit"],
                'cash': ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "現金及約約資金", "現金及約當現金", "現金", "約當現金", "流動資產-現金及約當現金"],
                'accounts_receivable': ["Accounts Receivable", "Net Receivables", "應收帳款淨額", "應收帳款", "應收票據及帳款", "流動應收帳款", "應收帳款及票據"],
                'inventory': ["Inventory", "存貨", "存貨淨額", "流動資產-存貨"],
                'accounts_payable': ["Accounts Payable", "應付帳款", "應付票據及帳款", "流動應付帳款", "應付帳款及票據"],
                'net_ppe': ["Net PPE", "Property Plant And Equipment Net", "不動產、廠房及設備", "固定資產", "非流動資產-不動產、廠房及設備"],
                'total_debt': ["Total Debt", "Long Term Debt", "Short Long Term Debt", "短期借款", "應付短期票券", "一年內到期長期負債", "長期借款", "應付公司債", "流動/非流動負債-借款", "非流動負債-借款"],
                'share_capital': ["Common Stock", "Common Stock Equity", "股本", "普通股股本", "歸屬於母公司業主之權益總計", "權益總計"],
                'retained_earnings': ["Retained Earnings", "保留盈餘", "累積盈餘"],
                'da': ["Reconciled Depreciation", "Depreciation And Amortization", "Depreciation Amortization Depletion", "折舊及攤銷", "折舊及攤銷費用", "折舊費用", "攤銷費用", "折舊", "攤銷", "折舊及攤銷合計", "折舊及攤銷金額", "折舊、折耗及攤銷"],
                'capex': ["Capital Expenditure", "Capital Expenditures", "取得不動產、廠房及設備", "取得不動產、廠房及設備（不含租賃資產）", "取得資產及設備", "取得不動產、廠房及設備之支出"]
            },
            'us': {
                'revenue': ["Revenues", "RevenueFromContractWithCustomerExcludingCostReportedOnNetBasis", "TotalRevenue", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax", "TotalRevenues"],
                'ebit': ["OperatingIncomeLoss", "OperatingIncome", "EBIT"],
                'net_income': ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
                'cash': ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents", "Cash", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
                'accounts_receivable': ["AccountsReceivableNetCurrent", "AccountsReceivableNet", "ReceivablesNetCurrent"],
                'inventory': ["InventoryNet", "InventoryNetCurrent", "Inventory"],
                'accounts_payable': ["AccountsPayableCurrent", "AccountsPayable"],
                'net_ppe': ["PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipmentNetNoncurrent", "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"],
                'total_debt': ["ShortTermBorrowings", "LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebt", "CommercialPaper", "LongTermDebtAndCapitalLeaseObligations"],
                'share_capital': ["StockholdersEquity", "CommonStockValue", "AdditionalPaidInCapital", "CommonStockIncludingAdditionalPaidInCapital", "CommonStocksIncludingAdditionalPaidInCapital"], 
                'retained_earnings': ["RetainedEarningsAccumulatedDeficit", "RetainedEarnings"],
                'da': ["DepreciationAndAmortization", "DepreciationDepletionAndAmortization", "Depreciation", "AmortizationOfIntangibleAssets", "DepreciationAmortizationAndAccretionNet"],
                'capex': ["CapitalExpenditure", "CapitalExpenditures", "PaymentsToAcquirePropertyPlantAndEquipment", "AcquisitionOfProductiveAssets"]
            }
        }
        
        m = mapping.get(self.market, {})
        
        def get_v_bs(series, keys, label=None):
            for k in keys:
                if k in series and not pd.isna(series[k]):
                    return series[k]
            return 0

        try:
            is_df, bs_df, cf_df = self.get_full_financials()
            if is_df is None or bs_df is None or is_df.empty or bs_df.empty:
                logger.warning(f"Insufficient financial data for {self.symbol} ({self.market})")
                return {}
            
            recent_is = is_df.iloc[-4:] if not is_df.empty else pd.DataFrame()
            recent_cf = cf_df.iloc[-4:] if not cf_df.empty else pd.DataFrame()
            last_bs = bs_df.iloc[-1] if not bs_df.empty else pd.Series()
            
            ebit_ttm = get_ttm(recent_is, m.get('ebit', []))
            # D&A prefers CF if available
            da_ttm = get_ttm(recent_cf, m.get('da', []))
            if da_ttm <= 0:
                da_ttm = get_ttm(recent_is, m.get('da', []))
            
            return {
                'revenue': get_ttm(recent_is, m.get('revenue', [])),
                'ebit': ebit_ttm,
                'ebitda': ebit_ttm + da_ttm,
                'net_income': get_ttm(recent_is, m.get('net_income', [])),
                'cash': get_v_bs(last_bs, m.get('cash', []), 'cash'),
                'accounts_receivable': get_v_bs(last_bs, m.get('accounts_receivable', []), 'accounts_receivable'),
                'inventory': get_v_bs(last_bs, m.get('inventory', []), 'inventory'),
                'accounts_payable': get_v_bs(last_bs, m.get('accounts_payable', []), 'accounts_payable'),
                'net_ppe': get_v_bs(last_bs, m.get('net_ppe', []), 'net_ppe'),
                'total_debt': get_v_bs(last_bs, m.get('total_debt', []), 'total_debt'),
                'share_capital': get_v_bs(last_bs, m.get('share_capital', []), 'share_capital'),
                'retained_earnings': get_v_bs(last_bs, m.get('retained_earnings', []), 'retained_earnings'),
                'shares_outstanding': self._get_shares_outstanding(),
                'capex': abs(get_ttm(recent_cf, m.get('capex', []))),
                'depreciation': abs(da_ttm)
            }
        except Exception as e:
            logger.error(f"Mapping Error: {e}")
            return None

    def calculate_historical_ratios(self):
        is_df, bs_df, cf_df = self.get_full_financials()
        if is_df is None or is_df.empty: return {
            'tax_rate': 0.20, 'ebit_margin': 0.15, 
            'capex_as_pct_revenue': 0.05, 'da_as_pct_revenue': 0.05,
            'ar_as_pct_revenue': 0.1, 'inv_as_pct_revenue': 0.1, 'ap_as_pct_revenue': 0.05
        }
        
        # 使用完整的 mapping 查找
        m = self._get_full_mapping()
        
        recent_is = is_df.iloc[-4:] # 取最近 4 季 (年化基準)
        recent_cf = cf_df.iloc[-4:] if not cf_df.empty else pd.DataFrame()
        last_bs = bs_df.iloc[-1]
        
        def get_sum(df, keys):
            if df.empty: return pd.Series([0]*1, index=[0])
            cols = [k for k in keys if k in df.columns]
            return df[cols].sum(axis=1) if cols else pd.Series([0]*len(df), index=df.index)

        rev_series = get_sum(recent_is, m.get('revenue', []))
        ebit_series = get_sum(recent_is, m.get('ebit', []))
        # D&A 優先從 CF 抓取，否則從 IS 抓取
        da_series = get_sum(recent_cf, m.get('da', []))
        if da_series.sum() <= 0:
            da_series = get_sum(recent_is, m.get('da', []))
        
        capex_series = get_sum(recent_cf, m.get('capex', []))
        
        total_rev = rev_series.sum()
        if total_rev <= 0:
            return {
                'tax_rate': 0.20, 'ebit_margin': 0.15, 
                'capex_as_pct_revenue': 0.05, 'da_as_pct_revenue': 0.05,
                'ar_as_pct_revenue': 0.1, 'inv_as_pct_revenue': 0.1, 'ap_as_pct_revenue': 0.05
            }

        # 核心利潤率與支出率 (基於合計以平滑季度變動)
        ebit_margin = ebit_series.sum() / total_rev
        da_ratio = da_series.sum() / total_rev
        # CapEx 在現金流量表通常是負值，取絕對值
        # 關鍵修正：對於資本支出極大的股票，不應直接採納單季/單年的極端值，限制在營收的 50% 以內
        capex_ratio = abs(capex_series.sum()) / total_rev
        
        # 防止異常值 - 針對不同行業調整範圍 (如台積電折舊非常大)
        ebit_margin = max(min(float(ebit_margin), 0.7), -0.1) # 限制獲利率在合理區間
        da_ratio = max(min(float(da_ratio), 0.5), 0.01) 
        capex_ratio = max(min(float(capex_ratio), 0.6), 0.01) # 限制資本支出比例上限
        
        # 營運資本比率 (基於最新推算年化營收)
        # 偵測是年報還是季報：如果最近一期的 index 都是 Q4，視為年報
        last_rev = rev_series.iloc[-1]
        is_annual = all(str(idx).endswith('Q4') for idx in recent_is.index)
        last_rev_annual = last_rev if is_annual else (last_rev * 4)
        
        def get_bs_val(series, keys):
            for k in keys:
                if k in series.index and not pd.isna(series[k]):
                    return series[k]
            return 0

        ar = get_bs_val(last_bs, m.get('accounts_receivable', []))
        inv = get_bs_val(last_bs, m.get('inventory', []))
        ap = get_bs_val(last_bs, m.get('accounts_payable', []))
        
        return {
            'tax_rate': 0.20, # 暫定 20%
            'ebit_margin': ebit_margin,
            'da_as_pct_revenue': da_ratio,
            'capex_as_pct_revenue': capex_ratio,
            'ar_as_pct_revenue': ar / last_rev_annual if last_rev_annual > 0 else 0.1,
            'inv_as_pct_revenue': inv / last_rev_annual if last_rev_annual > 0 else 0.1,
            'ap_as_pct_revenue': ap / last_rev_annual if last_rev_annual > 0 else 0.05
        }

    def _get_full_mapping(self):
        """Helper to get the mapping without repeating code"""
        # (This is a simplified version of the mapping in extract_projection_start)
        # Ideally, this should be a class-level constant, but let's keep it simple for now.
        return {
            'tw': {
                'revenue': ["Total Revenue", "Operating Revenue", "營業收入合計", "營業收入", "營業收入淨額"],
                'ebit': ["Operating Income", "EBIT", "營業利益（損失）", "營業利益", "營業利益淨額"],
                'da': ["Depreciation And Amortization", "折舊及攤銷", "折舊及攤銷費用", "折舊費用", "攤銷費用", "折舊", "攤銷", "折舊及攤銷合計"],
                'capex': ["Capital Expenditure", "Capital Expenditures", "取得不動產、廠房及設備", "取得資產及設備"],
                'accounts_receivable': ["Accounts Receivable", "應收帳款淨額", "應收帳款"],
                'inventory': ["Inventory", "存貨", "存貨淨額"],
                'accounts_payable': ["Accounts Payable", "應付帳款"]
            },
            'us': {
                'revenue': ["Revenues", "TotalRevenue", "SalesRevenueNet"],
                'ebit': ["OperatingIncomeLoss"],
                'da': ["DepreciationAndAmortization", "Depreciation"],
                'capex': ["CapitalExpenditure", "CapitalExpenditures"],
                'accounts_receivable': ["AccountsReceivableNetCurrent"],
                'inventory': ["InventoryNetCurrent"],
                'accounts_payable': ["AccountsPayableCurrent"]
            }
        }.get(self.market, {})

    def get_historical_growth_rates(self):
        is_df, _, _ = self.get_full_financials()
        if is_df is None or is_df.empty: return pd.Series([0.05])
        
        m_rev = {
            'tw': ["營業收入合計", "營業收入"],
            'us': ["Revenues", "TotalRevenue", "RevenueFromContractWithCustomerExcludingCostReportedOnNetBasis", "SalesRevenueNet"]
        }.get(self.market, [])
        
        cols = [k for k in m_rev if k in is_df.columns]
        if not cols: return pd.Series([0.05])
        
        # 獲取營收並排序
        rev = is_df[cols].sum(axis=1).sort_index()
        
        # 關鍵修正：移除 0 或負值，避免 inf
        rev = rev[rev > 0]
        if rev.empty: return pd.Series([0.05])

        # 如果只有兩筆且時間間隔過大，需要年化
        if len(rev) == 2:
            # 解析日期，例如 2023Q3 vs 2025Q3
            d1 = rev.index[0]
            d2 = rev.index[1]
            try:
                q1 = int(d1[:4]) * 4 + int(d1[-1])
                q2 = int(d2[:4]) * 4 + int(d2[-1])
                num_quarters = q2 - q1
                if num_quarters > 0:
                    total_growth = rev.iloc[1] / rev.iloc[0]
                    # 年化成長率：(total_growth ^ (4 / num_quarters)) - 1
                    annual_growth = (total_growth ** (4 / num_quarters)) - 1
                    # 限制在合理範圍 [-50%, +100%]
                    annual_growth = max(min(annual_growth, 1.0), -0.5)
                    return pd.Series([annual_growth])
            except: pass

        if len(rev) >= 4:
            growth = rev.pct_change(4).dropna()
        else:
            growth = rev.pct_change().dropna()
            
        # 清洗 growth：移除 inf 並限制範圍
        import numpy as np
        growth = growth.replace([np.inf, -np.inf], np.nan).dropna()
        # 限制在合理範圍 [-50%, +200%]，防止極端值毀掉模型
        growth = growth.clip(lower=-0.5, upper=2.0)

        return growth if not growth.empty else pd.Series([0.05])

    def _get_shares_outstanding(self):
        """估計發行股數 (絕對單位)"""
        table_name = f"financial_raw_{self.market}"
        
        if self.market == 'tw':
            query = f"SELECT item_name, amount FROM {table_name} WHERE symbol = :symbol AND statement_type = 'BS' AND (item_name LIKE '%%股本%%' OR item_name LIKE '%%CommonStock%%') ORDER BY year DESC, quarter DESC LIMIT 10"
        else:
            # 美股優先找 SharesOutstanding 或 WeightedAverage
            query = f"SELECT item_name, amount FROM {table_name} WHERE symbol = :symbol AND (item_name LIKE '%%CommonStockSharesOutstanding%%' OR item_name LIKE '%%WeightedAverageNumberOfSharesOutstandingBasic%%') ORDER BY year DESC, quarter DESC LIMIT 5"

        with self.op.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"symbol": self.symbol})
            
        if not df.empty:
            if self.market == 'tw':
                for _, row in df.iterrows():
                    if any(x in row['item_name'] for x in ["普通股", "股本"]):
                        return (row['amount'] * 1000) / 10 
            else:
                # 美股直接回傳第一筆找到的數目
                return float(df.iloc[0]['amount'])

        return 1e9 # 預設值

    def get_currency(self):
        return 'TWD' if self.market == 'tw' else 'USD'

    def get_market_price(self):
        """從資料庫抓取最新一期收盤價，確保不受重複日期影響"""
        table_name = "stock_cost" if self.market == 'tw' else "stock_cost_us"
        
        # 使用 Date DESC, id DESC (若有) 確保拿到最後存入的一筆
        query = text(f"""
            SELECT Close 
            FROM {table_name} 
            WHERE TRIM(number) = :symbol 
            ORDER BY Date DESC
            LIMIT 1
        """)
        try:
            with self.op.engine.connect() as conn:
                result = conn.execute(query, {"symbol": self.full_symbol}).fetchone()
                if result:
                    return float(result[0])
        except Exception as e:
            logger.error(f"Error fetching market price for {self.full_symbol}: {e}")
            
        return 0

    def ensure_price_freshness(self):
        """
        不再主動檢查，因為 Dashboard 的 Analyze 按鈕已負責更新。
        此處僅保留作為介面相容性。
        """
        pass

    def get_historical_multiples(self):
        """從資料庫數據估算歷史平均 P/E 與 EV/EBITDA"""
        # 1. 獲取財務數據
        start_data = self.extract_projection_start()
        if not start_data: return {'pe': 15, 'ev_ebitda': 10}
        
        # 2. 獲取最近幾年的淨利與 EBITDA (簡化：取 TTM 並參考目前的市值)
        current_price = self.get_market_price()
        if current_price <= 0: return {'pe': 15, 'ev_ebitda': 10}
        
        eps = start_data['net_income'] / max(start_data['shares_outstanding'], 1)
        current_pe = current_price / eps if eps > 0 else 15
        
        ebitda = start_data['ebitda']
        market_cap = current_price * start_data['shares_outstanding']
        ev = market_cap + (start_data['total_debt'] - start_data['cash'])
        current_ev_ebitda = ev / ebitda if ebitda > 0 else 10
        
        # 限制在合理範圍
        avg_pe = max(min(current_pe, 30), 10)
        avg_ev_ebitda = max(min(current_ev_ebitda, 20), 8)
        
        return {
            'pe': avg_pe,
            'ev_ebitda': avg_ev_ebitda,
            'current_pe': current_pe,
            'current_ev_ebitda': current_ev_ebitda
        }
