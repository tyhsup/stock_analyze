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

    def validate_terminal_growth(self, risk_free_rate: float = 0.03) -> float:
        """
        動態檢驗並約束永續增長率不得超過無風險利率 (g <= Rf)。
        若超出上限，自動 clamp 並記錄警告。
        """
        rf = float(risk_free_rate if risk_free_rate is not None else 0.03)
        if self.perpetuity_growth_rate > rf:
            import logging
            logging.getLogger(__name__).warning(
                "永續增長率 g (%.2f%%) 超過無風險利率 (%.2f%%)，自動約束至上限。",
                self.perpetuity_growth_rate * 100, rf * 100
            )
            self.perpetuity_growth_rate = rf
        elif self.perpetuity_growth_rate < 0:
            self.perpetuity_growth_rate = 0.0
        return float(self.perpetuity_growth_rate)

    def calculate_reinvestment_rate(self, roic: float, g: float = None) -> float:
        """
        依據 ROIC 與永續成長率 g 計算穩定期再投資率 (Reinvestment Rate = g / ROIC)。
        
        邊界防禦：
        - 若 ROIC <= 0 且 g > 0，強制再投資率為 1.0 (100% 再投資以維持增長，防止無本萬利假設)。
        - 若 g <= 0，再投資率為 0.0。
        - 正常情況 clamp 於 [0.0, 1.0]。
        """
        eff_g = float(g if g is not None else self.perpetuity_growth_rate)
        eff_roic = float(roic if roic is not None else 0.10)

        if eff_g <= 0:
            return 0.0

        if eff_roic <= 0:
            # 虧損或零資本回報卻預期增長，必須投入全部利潤
            return 1.0

        rr = eff_g / eff_roic
        return float(max(min(rr, 1.0), 0.0))

    def calculate_terminal_fcff(self, nopat_n: float, roic: float, g: float = None) -> float:
        """
        計算終值期第一年自由現金流 (Terminal FCFF_n+1)。
        公式：NOPAT_n * (1 + g) * (1 - Reinvestment_Rate)
        """
        eff_g = float(g if g is not None else self.perpetuity_growth_rate)
        rr = self.calculate_reinvestment_rate(roic, eff_g)
        terminal_nopat = float(nopat_n) * (1.0 + eff_g)
        return float(terminal_nopat * (1.0 - rr))

