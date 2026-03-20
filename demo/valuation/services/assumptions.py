class Assumptions:
    def __init__(self):
        # --- 預測期假設 (5-10 年) ---
        self.revenue_growth_rate = [0.15, 0.12, 0.10, 0.08, 0.05]  # 每年的營收增長率
        self.ebit_margin = 0.20  # 預測期的平均 EBIT Margin
        self.tax_rate = 0.25     # 有效稅率
        
        # --- 資產負債表與現金流假設 ---
        self.depreciation_as_pct_revenue = 0.03 # 折舊佔營收比
        
        # 營運資本假設 (Working Capital)
        self.ar_as_pct_revenue = 0.15   # 應收帳款佔營收比
        self.inv_as_pct_revenue = 0.10  # 存貨佔營收比
        self.ap_as_pct_revenue = 0.10   # 應付帳款佔營收比

        self.capex_as_pct_sales = 0.03  # 資本支出佔營收比

        # --- WACC 假設 (用於折現率) ---
        self.risk_free_rate = 0.03       # 無風險利率 (Rf)
        self.equity_beta = 1.25          # 股權 Beta
        self.market_risk_premium = 0.06  # 市場風險溢酬 (MRP)
        self.cost_of_debt = 0.05         # 債務成本 (Rd)
        self.target_debt_to_equity_ratio = 0.4  # 目標資本結構 D/E 比例

        # --- 永續期假設 (Terminal Value) ---
        self.perpetuity_growth_rate = 0.03 # g
