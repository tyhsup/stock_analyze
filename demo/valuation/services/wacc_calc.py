import pandas as pd
import logging
from sqlalchemy import text
from stock_Django.mySQL_OP import OP_Fun

logger = logging.getLogger(__name__)

class WACCCalculator:
    def __init__(self, ticker_symbol):
        # 判斷市場 (邏輯：包含 .TW 或 純數字 為台灣，否則為美國)
        temp_symbol = ticker_symbol.upper()
        if ".TW" in temp_symbol or ".TWO" in temp_symbol or temp_symbol.isdigit():
            # 台灣市場
            self.market = 'tw'
            self.symbol = temp_symbol.replace(".TW", "").replace(".TWO", "")
            if ".TWO" in temp_symbol:
                self.ticker_symbol = f"{self.symbol}.TWO"
            else:
                self.ticker_symbol = f"{self.symbol}.TW"
        else:
            # 美國市場
            self.market = 'us'
            self.symbol = temp_symbol
            self.ticker_symbol = temp_symbol
            
        self.op = OP_Fun()
        self.info = self._get_info_from_local()
        
    def _get_info_from_local(self):
        """從本地資料庫獲取市值、負債等關鍵數據"""
        logger.info(f"Fetching WACC market data for {self.symbol} from local DB...")
        
        # 1. 獲取最新股價 (stock_cost 使用帶有 .TW 的完整代碼)
        price_table = "stock_cost" if self.market == 'tw' else "stock_cost_us"
        query_price = text(f"SELECT Close FROM {price_table} WHERE TRIM(number) = :symbol ORDER BY Date DESC LIMIT 1")
        
        # 2. 獲取股本 (financial_raw_tw 使用純數字代碼)
        fin_table = f"financial_raw_{self.market}"
        if self.market == 'tw':
            query_shares = text(f"SELECT amount FROM {fin_table} WHERE symbol = :symbol AND statement_type = 'BS' AND (item_name LIKE '%%普通股%%' OR item_name LIKE '%%股本%%') ORDER BY year DESC, quarter DESC LIMIT 1")
        else:
            query_shares = text(f"SELECT amount FROM {fin_table} WHERE symbol = :symbol AND (item_name LIKE '%%CommonStockSharesOutstanding%%' OR item_name LIKE '%%WeightedAverageNumberOfSharesOutstandingBasic%%') ORDER BY year DESC, quarter DESC LIMIT 1")
            
        # 3. 獲取總負債
        if self.market == 'tw':
            query_debt = text(f"SELECT amount FROM {fin_table} WHERE symbol = :symbol AND statement_type = 'BS' AND (item_name LIKE '%%負債總額%%' OR item_name LIKE '%%負債總計%%') ORDER BY year DESC, quarter DESC LIMIT 1")
        else:
            query_debt = text(f"SELECT amount FROM {fin_table} WHERE symbol = :symbol AND item_name LIKE '%%TotalLiabilities%%' ORDER BY year DESC, quarter DESC LIMIT 1")

        info = {
            'beta': 1.0,
            'marketCap': None,
            'totalDebt': 0,
            'currency': 'TWD' if self.market == 'tw' else 'USD',
            'currentPrice': 0,
            'sharesOutstanding': 100e6
        }

        try:
            with self.op.engine.connect() as conn:
                # 執行查詢
                # 注意：price 查詢使用 self.ticker_symbol (2330.TW), fin 查詢使用 self.symbol (2330)
                p_row = conn.execute(query_price, {"symbol": self.ticker_symbol}).fetchone()
                s_row = conn.execute(query_shares, {"symbol": self.symbol}).fetchone()
                d_row = conn.execute(query_debt, {"symbol": self.symbol}).fetchone()
                
                if p_row:
                    info['currentPrice'] = float(p_row[0])
                
                if s_row:
                    raw_shares = float(s_row[0])
                    if self.market == 'tw':
                        # 台股股本單位為「千元」，除以面額 10 元 = 股數
                        info['sharesOutstanding'] = (raw_shares * 1000) / 10
                    else:
                        info['sharesOutstanding'] = raw_shares
                
                if d_row:
                    raw_debt = float(d_row[0])
                    info['totalDebt'] = raw_debt * 1000 if self.market == 'tw' else raw_debt
                
                # 計算市值
                if info['currentPrice'] > 0:
                    info['marketCap'] = info['currentPrice'] * info['sharesOutstanding']
                
        except Exception as e:
            logger.error(f"Error fetching local WACC data for {self.symbol}: {e}")

        # 保底市值
        if info['marketCap'] is None:
            info['marketCap'] = 10e9
            
        return info

    def get_risk_free_rate(self):
        """返回固定無風險利率 (4.2%)，避免 API 阻擋"""
        return 0.042

    def calculate_cost_of_equity(self):
        rf = self.get_risk_free_rate()
        beta = self.info.get('beta', 1.0)
        return rf + (beta * 0.055) # CAPM (ERP = 5.5%)

    def calculate_cost_of_debt(self):
        """從本地數據庫估算債務成本"""
        total_debt = self.info.get('totalDebt', 0)
        fin_table = f"financial_raw_{self.market}"
        
        # 嘗試抓取利息支出 (Interest Expense)
        if self.market == 'tw':
            query_int = text(f"SELECT amount FROM {fin_table} WHERE symbol = :symbol AND statement_type = 'IS' AND (item_name LIKE '%%利息支出%%' OR item_name LIKE '%%InterestExpense%%') ORDER BY year DESC, quarter DESC LIMIT 1")
        else:
            query_int = text(f"SELECT amount FROM {fin_table} WHERE symbol = :symbol AND item_name LIKE '%%InterestExpense%%' ORDER BY year DESC, quarter DESC LIMIT 1")
            
        try:
            with self.op.engine.connect() as conn:
                int_row = conn.execute(query_int, {"symbol": self.symbol}).fetchone()
                if int_row and total_debt > 0:
                    interest = abs(float(int_row[0]))
                    if self.market == 'tw': interest *= 1000
                    rd = interest / total_debt
                    if 0.01 < rd < 0.20:
                        return rd
        except: pass
        
        return 0.045 # 預設值

    def calculate_wacc(self, custom_tax_rate=0.21):
        mcap = self.info.get('marketCap', 10e9)
        debt = self.info.get('totalDebt', 0)
        v = mcap + debt
        
        if v <= 0:
            return {"WACC": 0.08, "Cost of Debt (Rd)": 0.045}

        re = self.calculate_cost_of_equity()
        rd = self.calculate_cost_of_debt()
        wacc = (mcap/v * re) + (debt/v * rd * (1 - custom_tax_rate))
        
        return {"WACC": wacc, "Cost of Debt (Rd)": rd}
