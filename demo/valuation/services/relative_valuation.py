import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RelativeValuator:
    def __init__(self, ticker_symbol, start_data, current_price, currency='USD'):
        """
        ticker_symbol: 股票代號
        start_data: 從 FinancialDataLoader.extract_projection_start() 獲取的數據
        current_price: 當前市價
        currency: 幣別
        """
        self.ticker = ticker_symbol
        self.data = start_data
        self.price = current_price
        self.currency = currency
        
        # 提取核心數值
        self.net_income = start_data.get('net_income', 0)
        self.ebitda = start_data.get('ebitda', 0)
        self.shares = max(start_data.get('shares_outstanding', 1), 1)
        self.total_debt = start_data.get('total_debt', 0)
        self.cash = start_data.get('cash', 0)
        self.net_debt = self.total_debt - self.cash
        
        # 計算現行指標 (Current Multiples)
        self.eps = self.net_income / self.shares
        self.pe_ratio = self.price / self.eps if self.eps > 0 else float('inf')
        
        self.market_cap = self.price * self.shares
        self.ev = self.market_cap + self.net_debt
        self.ev_ebitda = self.ev / self.ebitda if self.ebitda > 0 else float('inf')

    def calculate_implied_fair_value(self, target_pe=None, target_ev_ebitda=None):
        """
        基於目標倍數計算隱含價值
        target_pe: 目標本益比 (如 20x)
        target_ev_ebitda: 目標 EV/EBITDA (如 12x)
        """
        results = {}
        
        # 1. 基於 P/E
        if target_pe:
            implied_price_pe = self.eps * target_pe
            results['pe_approach'] = {
                'target_multiple': target_pe,
                'implied_price': implied_price_pe
            }
            
        # 2. 基於 EV/EBITDA
        if target_ev_ebitda:
            # EV = EBITDA * multiple
            # Equity Value = EV - Net Debt
            # Price = Equity Value / Shares
            implied_ev = self.ebitda * target_ev_ebitda
            implied_equity_value = implied_ev - self.net_debt
            implied_price_ev = implied_equity_value / self.shares
            results['ev_ebitda_approach'] = {
                'target_multiple': target_ev_ebitda,
                'implied_price': implied_price_ev
            }
            
        return results

    def get_summary(self):
        return {
            'ticker': self.ticker,
            'current_price': self.price,
            'currency': self.currency,
            'eps_ttm': self.eps,
            'ebitda_ttm': self.ebitda,
            'pe_ratio': self.pe_ratio,
            'ev_ebitda': self.ev_ebitda,
            'market_cap': self.market_cap,
            'enterprise_value': self.ev
        }
