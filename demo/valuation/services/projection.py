import pandas as pd
import numpy as np

class FinancialProjector:
    def __init__(self, last_historical_year, assumptions, projection_years=5):
        """
        last_historical_year: Dictionary 或 Series，包含上一年度實際數據
        assumptions: Assumptions 物件，包含未來增長率、利潤率等
        """
        self.last_hist = last_historical_year
        self.assumptions = assumptions
        self.years = projection_years
        self.projected_statements = [] # 儲存每年的預測結果

    def run_projection(self):
        # 初始化：將上一年的歷史數據作為起點
        previous_year = self.last_hist
        
        for i in range(self.years):
            # 獲取第 i 年的特定假設 (如果假設隨年份變化)
            # 這裡簡化為直接調用 assumptions
            
            # 1. 預測損益表
            is_proj = self._project_income_statement(previous_year, i)
            
            # 2. 預測資產負債表 (部分) - 營運資本與長期資產
            bs_proj = self._project_balance_sheet_items(previous_year, is_proj, i)
            
            # 3. 預測現金流量表 & 完成資產負債表平衡
            final_year_data = self._project_cash_flow_and_balance(previous_year, is_proj, bs_proj)
            
            # 儲存結果
            self.projected_statements.append(final_year_data)
            
            # 更新 previous_year 為剛預測完的這一年，供下一次迭代使用
            previous_year = final_year_data
            
        return pd.DataFrame(self.projected_statements)

    # --- 以下是內部邏輯方法 ---
    def _project_income_statement(self, prev, year_idx):
        growth_rate = self.assumptions.revenue_growth_rate[year_idx]
        ebit_margin = self.assumptions.ebit_margin
        tax_rate = self.assumptions.tax_rate
        
        # 1. Revenue
        revenue = prev['revenue'] * (1 + growth_rate)
        
        # 2. EBIT (Operating Income)
        # 簡化模型通常直接用 EBIT Margin，複雜模型會拆分 COGS 和 OpEx
        # 2. EBIT 預測
        ebit = revenue * self.assumptions.ebit_margin
            
        # 3. D&A (折舊與攤銷)
        # 通常與 CapEx 掛鉤，或作為營收的百分比
        depreciation = revenue * self.assumptions.depreciation_as_pct_revenue
        
        # 4. Interest Expense (利息費用)
        # 注意：為了避免循環引用 (Circular Reference)，通常使用「期初債務」計算利息
        interest_expense = prev['total_debt'] * self.assumptions.cost_of_debt
        
        # 5. EBT & Net Income
        ebt = ebit - interest_expense
        taxes = ebt * tax_rate
        net_income = ebt - taxes
        
        return {
            'revenue': revenue,
            'ebit': ebit,
            'depreciation': depreciation,
            'interest_expense': interest_expense,
            'taxes': taxes,
            'net_income': net_income
        }
    
    def _project_balance_sheet_items(self, prev, is_data, year_idx):
        # 1. Working Capital Items (基於周轉天數或營收百分比)
        # Accounts Receivable (應收帳款)
        ar = is_data['revenue'] * self.assumptions.ar_as_pct_revenue
        
        # Accounts Payable (應付帳款) - 通常基於 COGS，這裡簡化基於 Revenue
        ap = is_data['revenue'] * self.assumptions.ap_as_pct_revenue
        
        # Inventory (存貨)
        inventory = is_data['revenue'] * self.assumptions.inv_as_pct_revenue
        
        # 2. Long Term Assets (PP&E)
        capex = is_data['revenue'] * self.assumptions.capex_as_pct_sales
        # 期末 PP&E = 期初 PP&E + CapEx - Depreciation
        net_ppe = prev['net_ppe'] + capex - is_data['depreciation']
        
        # 3. Debt (債務)
        # 假設債務保持不變 (除非有還款計畫)，實際模型中可設為變量
        total_debt = prev['total_debt'] 
        
        # 4. Equity (股東權益，不含當期淨利累積)
        # Retained Earnings = 期初 RE + Net Income (這部分在下一步整合)
        share_capital = prev['share_capital'] # 假設股本不變
        
        return {
            'accounts_receivable': ar,
            'inventory': inventory,
            'accounts_payable': ap,
            'net_ppe': net_ppe,
            'capex': capex, # 雖然不是 BS 存量，但需要傳遞給 CFS
            'total_debt': total_debt,
            'share_capital': share_capital
        }
    
    def _project_cash_flow_and_balance(self, prev, is_data, bs_data):
        # 1. 計算營運資本變動 (Change in Working Capital)
        # 增加資產 = 現金流出 (-)，增加負債 = 現金流入 (+)
        delta_ar = bs_data['accounts_receivable'] - prev['accounts_receivable']
        delta_inv = bs_data['inventory'] - prev['inventory']
        delta_ap = bs_data['accounts_payable'] - prev['accounts_payable']
        
        change_in_wc = delta_ar + delta_inv - delta_ap # 注意這裡的符號邏輯，這是"佔用"的現金
        
        # 2. 自由現金流計算 (FCFF 用於 DCF，但這裡我們先算現金餘額)
        # Cash Flow from Operations (CFO)
        cfo = is_data['net_income'] + is_data['depreciation'] - change_in_wc
        
        # Cash Flow from Investing (CFI)
        cfi = -bs_data['capex']
        
        # Cash Flow from Financing (CFF)
        # 假設無新發債或還款，無股利
        cff = 0 
        
        # 3. 現金變動與期末餘額 (The Plug)
        net_change_in_cash = cfo + cfi + cff
        ending_cash = prev['cash'] + net_change_in_cash
        
        # 4. 完成資產負債表平衡檢查
        retained_earnings = prev['retained_earnings'] + is_data['net_income']
        
        total_assets = ending_cash + bs_data['accounts_receivable'] + bs_data['inventory'] + bs_data['net_ppe']
        total_liabilities_equity = bs_data['accounts_payable'] + bs_data['total_debt'] + bs_data['share_capital'] + retained_earnings
        
        # 理論上 total_assets 應該等於 total_liabilities_equity
        # 在這裡，我們把計算出的所有數據合併
        full_data = {**is_data, **bs_data}
        full_data['cash'] = ending_cash
        full_data['retained_earnings'] = retained_earnings
        full_data['change_in_wc'] = change_in_wc # 儲存供 DCF 計算用
        full_data['check_balance'] = total_assets - total_liabilities_equity # 應該接近 0
        
        return full_data
