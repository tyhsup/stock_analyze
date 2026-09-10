import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger("ValuationEngine.Sensitivity")


class SensitivityAnalyzer:
    """
    加權平均資本成本 (WACC) 與永續成長率 (g) 雙向敏感度矩陣分析器。
    
    生成 2D 估值熱力圖矩陣數據，支援自適應中心基準與 Gordon 模型防禦。
    """

    @classmethod
    def generate_matrix(
        cls,
        base_revenue: float,
        growth_rates: List[float],
        ebit_margin: float,
        tax_rate: float,
        reinvestment_rate: float,
        base_wacc: float,
        base_g: float,
        net_debt: float,
        shares_outstanding: float,
        risk_free_rate: float = 0.03,
        is_mid_year: bool = False
    ) -> Dict[str, Any]:
        """
        生成 WACC (縱軸) x g (橫軸) 雙向估值矩陣。
        """
        shares = max(float(shares_outstanding), 1.0)
        debt = float(net_debt)
        t_rate = max(min(float(tax_rate), 0.5), 0.0)
        rr = max(min(float(reinvestment_rate), 1.0), 0.0)
        rf = max(float(risk_free_rate), 0.02)
        margin = max(float(ebit_margin), 0.01)

        # 1. 構建 WACC 區間 (Base ± 2.0%, 步長 0.5%)
        w_center = max(float(base_wacc), 0.04)
        wacc_deltas = [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02]
        wacc_values = []
        for dw in wacc_deltas:
            w = round(w_center + dw, 4)
            if 0.03 <= w <= 0.25:
                wacc_values.append(w)
        wacc_values = sorted(list(set(wacc_values)))

        # 2. 構建 g 區間 (0.5% ~ 3.0%, 不得超過 Rf)
        g_candidates = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
        g_values = [g for g in g_candidates if g <= rf]
        if not g_values:
            g_values = [min(0.01, rf)]
        g_values = sorted(list(set(g_values)))

        # 3. 預先計算 5 年未折現現金流
        years = len(growth_rates) if growth_rates else 5
        revenues = []
        cur_rev = float(base_revenue)
        fcfs = []

        for t in range(years):
            g_t = growth_rates[t] if t < len(growth_rates) else 0.05
            cur_rev *= (1.0 + g_t)
            revenues.append(cur_rev)
            ebit_t = cur_rev * margin
            nopat_t = ebit_t * (1.0 - t_rate)
            fcff_t = nopat_t * (1.0 - rr)
            fcfs.append(fcff_t)

        nopat_last = (revenues[-1] * margin) * (1.0 - t_rate)

        # 4. 生成 2D 估值矩陣
        matrix: List[List[float]] = []

        for w_val in wacc_values:
            row: List[float] = []
            for g_val in g_values:
                # 戈登模型防禦：確保分母 >= 0.5%
                denom = max(w_val - g_val, 0.005)

                # 5 年現金流折現
                pv_fcf_sum = 0.0
                for i, fcf in enumerate(fcfs):
                    t_exp = (i + 0.5) if is_mid_year else (i + 1.0)
                    pv_fcf_sum += fcf / ((1.0 + w_val) ** t_exp)

                # 終端價值與折現
                term_fcff = nopat_last * (1.0 + g_val) * (1.0 - rr)
                tv = term_fcff / denom
                tv_exp = (years - 0.5) if is_mid_year else float(years)
                pv_tv = tv / ((1.0 + w_val) ** tv_exp)

                # 企業價值與股權價值
                ev = pv_fcf_sum + pv_tv
                equity_val = max(ev - debt, 0.0)
                price = equity_val / shares
                row.append(round(float(price), 2))
            matrix.append(row)

        # 找出最接近基準值的行列索引
        wacc_idx = min(range(len(wacc_values)), key=lambda i: abs(wacc_values[i] - w_center))
        g_idx = min(range(len(g_values)), key=lambda i: abs(g_values[i] - float(base_g)))

        return {
            "wacc_labels": [f"{round(w * 100, 2)}%" for w in wacc_values],
            "g_labels": [f"{round(g * 100, 2)}%" for g in g_values],
            "matrix": matrix,
            "base_coordinates": {
                "wacc_index": int(wacc_idx),
                "g_index": int(g_idx)
            }
        }
