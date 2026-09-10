import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger("ValuationEngine.MonteCarlo")


class MonteCarloSimulator:
    """
    高精度 NumPy 向量化蒙地卡羅 DCF 估值模擬器。
    
    支援隨機變數分佈：
    - 營收增長率 (Revenue Growth): 正態分佈 (Normal Distribution)
    - 營業利益率 (EBIT Margin): 三角分佈 (Triangular Distribution)
    - 加權平均資本成本 (WACC): 正態分佈 (Normal Distribution, 下限 3%, 上限 25%)
    - 永續成長率 (g): 均勻分佈 (Uniform Distribution, 強制 g <= Rf 且 WACC - g >= 0.5%)
    """

    def __init__(self, num_simulations: int = 10000, seed: Optional[int] = 42):
        # 防禦：模擬次數限制於 1,000 ~ 50,000 次，避免記憶體與計算暴走
        self.num_simulations = min(max(num_simulations, 1000), 50000)
        self.seed = seed

    def run_simulation(
        self,
        base_revenue: float,
        base_growth: float,
        growth_std: float,
        ebit_margin: float,
        tax_rate: float,
        base_wacc: float,
        wacc_std: float,
        risk_free_rate: float,
        reinvestment_rate: float,
        net_debt: float,
        shares_outstanding: float,
        current_price: float,
        projection_years: int = 5
    ) -> Dict[str, Any]:
        """
        執行向量化隨機抽樣與估值分佈統計。
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        n = self.num_simulations
        shares = max(float(shares_outstanding), 1.0)
        debt = float(net_debt)
        cur_p = max(float(current_price), 0.01)

        # 1. 隨機變數生成 (Vectorized Sampling)
        # 營收複合年增長率 (CAGR) 抽樣
        g_rev_samples = np.random.normal(loc=base_growth, scale=max(growth_std, 0.01), size=n)
        g_rev_samples = np.clip(g_rev_samples, -0.40, 0.80)

        # EBIT Margin 三角分佈抽樣 (low=0.75*base, mode=base, high=1.25*base)
        m_base = max(float(ebit_margin), 0.02)
        m_low = max(m_base * 0.70, 0.01)
        m_high = min(m_base * 1.30, 0.85)
        margin_samples = np.random.triangular(left=m_low, mode=m_base, right=m_high, size=n)

        # WACC 正態分佈抽樣 (Clamp 至 [0.035, 0.22])
        wacc_samples = np.random.normal(loc=base_wacc, scale=max(wacc_std, 0.005), size=n)
        wacc_samples = np.clip(wacc_samples, 0.035, 0.22)

        # 永續成長率 g 均勻抽樣 (強制 g <= Rf 且 WACC - g >= 0.005)
        rf_eff = max(float(risk_free_rate), 0.02)
        g_raw = np.random.uniform(low=0.01, high=min(0.035, rf_eff), size=n)
        g_samples = np.minimum(g_raw, rf_eff)
        g_samples = np.minimum(g_samples, wacc_samples - 0.005)
        g_samples = np.maximum(g_samples, 0.0)

        # 2. 向量化 5 年現金流折現投影
        t_rate = max(min(float(tax_rate), 0.5), 0.0)
        rr = max(min(float(reinvestment_rate), 1.0), 0.0)
        years = projection_years

        pv_fcff_sum = np.zeros(n)
        rev_prev = np.full(n, max(float(base_revenue), 1.0))

        for t in range(1, years + 1):
            rev_t = rev_prev * (1.0 + g_rev_samples)
            ebit_t = rev_t * margin_samples
            nopat_t = ebit_t * (1.0 - t_rate)
            # 扣除再投資額後之 FCF
            fcff_t = nopat_t * (1.0 - rr)
            # 依各路徑的 WACC 進行折現
            discount_factor = (1.0 + wacc_samples) ** t
            pv_fcff_sum += (fcff_t / discount_factor)
            rev_prev = rev_t

        # 3. 終端價值 (Terminal Value) 向量化計算
        ebit_last = rev_prev * margin_samples
        nopat_last = ebit_last * (1.0 - t_rate)
        terminal_fcff = nopat_last * (1.0 + g_samples) * (1.0 - rr)
        denom = np.maximum(wacc_samples - g_samples, 0.005)
        tv = terminal_fcff / denom
        pv_tv = tv / ((1.0 + wacc_samples) ** years)

        # 4. 企業價值 (EV) 與每股股權價值
        simulated_ev = pv_fcff_sum + pv_tv
        simulated_equity = np.maximum(simulated_ev - debt, 0.0)
        simulated_prices = simulated_equity / shares

        # 極端值防禦剪裁 (防止溢位或 Inf)
        simulated_prices = np.clip(simulated_prices, 0.0, cur_p * 15.0)

        # 5. 統計指標聚合 (明確轉為原生 Python float，保證 JSON 序列化)
        mean_price = float(np.mean(simulated_prices))
        std_price = float(np.std(simulated_prices))
        pcts = np.percentile(simulated_prices, [5, 10, 25, 50, 75, 90, 95])
        
        # 現價在模擬分佈中的百分位 (0% ~ 100%)
        current_percentile = float(np.mean(simulated_prices < cur_p) * 100.0)

        # 生成 20-bin 直方圖數據供前端繪製分佈曲線
        counts, bin_edges = np.histogram(simulated_prices, bins=20)
        histogram_bins = []
        for i in range(len(counts)):
            histogram_bins.append({
                "bin_start": round(float(bin_edges[i]), 2),
                "bin_end": round(float(bin_edges[i + 1]), 2),
                "count": int(counts[i])
            })

        return {
            "num_simulations": int(n),
            "statistics": {
                "mean": round(mean_price, 2),
                "std": round(std_price, 2),
                "median": round(float(pcts[3]), 2),
                "p10": round(float(pcts[1]), 2),
                "p25": round(float(pcts[2]), 2),
                "p50": round(float(pcts[3]), 2),
                "p75": round(float(pcts[4]), 2),
                "p90": round(float(pcts[5]), 2),
                "ci_90_lower": round(float(pcts[0]), 2),
                "ci_90_upper": round(float(pcts[6]), 2),
                "current_price_percentile": round(current_percentile, 1)
            },
            "histogram": histogram_bins,
            "sampled_paths": [round(float(p), 2) for p in simulated_prices[:200]]  # 取樣 200 個點供前端散布圖
        }
