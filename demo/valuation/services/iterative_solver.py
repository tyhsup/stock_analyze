import logging
from enum import Enum
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger("ValuationEngine.IterativeSolver")


class FCFType(str, Enum):
    """
    現金流型別枚舉，用於 DCF 計算中的嚴格型別綁定防護。
    """
    FCFF = "FCFF"  # Free Cash Flow to Firm (企業自由現金流，對應 WACC)
    FCFE = "FCFE"  # Free Cash Flow to Equity (股權自由現金流，對應 Ke)


class ValuationError(ValueError):
    """估值引擎基礎異常類別。"""
    pass


class ConvergenceError(RuntimeError):
    """當數值迭代求解器在限制步數內無法收斂時拋出。"""
    pass


class InvalidInputError(ValuationError):
    """輸入參數格式、範圍或數據類型不符預期時拋出。"""
    pass


class TSMCalculator:
    """
    庫藏股法 (Treasury Stock Method, TSM) 完全稀釋股數計算器。
    
    支援雙軌制：
    - Level 1: 結構化期權/權證計算。當市價 P > 行權價 K (價內) 時，
      新增淨股數 = Options * (1 - K / P)。價外則不稀釋。
    - Level 2: 官方申報稀釋股數 Fallback。當缺少期權明細或價格異常時，
      直接採用官方申報的加權平均完全稀釋股數。
    """

    @staticmethod
    def calculate_diluted_shares(
        basic_shares: float,
        options_count: float = 0.0,
        strike_price: float = 0.0,
        current_price: float = 0.0,
        reported_diluted_shares: float = 0.0
    ) -> float:
        """
        計算完全稀釋股數，具備嚴格的邊界防禦與型別保護。
        
        :param basic_shares: 基本流通在外股數 (Basic Shares Outstanding)
        :param options_count: 未行使期權/認股權證總量
        :param strike_price: 加權平均行權價格 (Strike Price, K)
        :param current_price: 當前估計/市場價格 (Stock Price, P)
        :param reported_diluted_shares: 財報官方申報的稀釋股數 (Fallback)
        :return: 完全稀釋股數 (原生 Python float)
        """
        if basic_shares is None or basic_shares <= 0:
            raise ValuationError("基本流通股數 (basic_shares) 必須大於 0。")

        # 確保型別轉換為 float 避免 NumPy 類型洩漏
        b_shares = float(basic_shares)
        opts = float(options_count or 0.0)
        strike = float(strike_price or 0.0)
        price = float(current_price or 0.0)
        rep_diluted = float(reported_diluted_shares or 0.0)

        # 邊界防禦：價格極小值、非有限數 (NaN/Inf) 或零值
        if not np.isfinite(price) or price <= 1e-6:
            logger.warning(
                "檢測到異常或無效股價 (%.4f)，啟用 Level 2 官方申報稀釋股數 Fallback。", price
            )
            return float(max(b_shares, rep_diluted))

        # Level 1: 具備期權數量且行權價大於 0 時的 TSM 計算
        if opts > 0 and strike > 0:
            if price > strike:
                # 價內期權 (In-the-Money)：公司將行權所得資金全數在市場以市價買回股票
                proceeds = opts * strike
                shares_repurchased = proceeds / price
                net_dilution = opts - shares_repurchased
                diluted = b_shares + max(net_dilution, 0.0)
            else:
                # 價外期權 (Out-of-the-Money)：理性投資人不行權，不產生稀釋效應
                diluted = b_shares
        else:
            # 缺少詳細期權行權價，啟用 Level 2 官方申報稀釋股數
            diluted = max(b_shares, rep_diluted)

        # 最終防護：稀釋股數絕不可小於基本股數，並確保轉換為原生 float
        return float(max(diluted, b_shares))


class IterativeDCFSolver:
    """
    資本結構與股權市值循環依賴迭代求解器 (Fixed-point Iteration Solver)。
    
    解決核心循環論證：
    - 股價 P -> 稀釋股數 S -> 股權市值 E = P * S -> 資本結構 (We, Wd) -> WACC -> 折現企業價值 EV -> 股價 P'
    
    內建阻尼平滑 (Damping Factor, alpha=0.5) 防止高槓桿下的雙態震盪，
    並於每次迭代動態反饋更新 TSM 稀釋股數。
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-4,
        damping_factor: float = 0.5,
        min_wacc: float = 0.03,
        max_wacc: float = 0.20
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.damping_factor = max(min(damping_factor, 1.0), 0.1)
        self.min_wacc = min_wacc
        self.max_wacc = max_wacc

    def solve(
        self,
        initial_price: float,
        basic_shares: float,
        total_debt: float,
        cash: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float,
        discount_fn: Any,
        options_count: float = 0.0,
        strike_price: float = 0.0,
        reported_diluted_shares: float = 0.0,
        debt_adjust: float = 0.0,
        wacc_premium: float = 0.0,
        operating_lease_liability: float = 0.0,
        preferred_stock: float = 0.0,
        minority_interest: float = 0.0,
        pension_deficit: float = 0.0,
        net_debt: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        執行資本結構迭代求解。

        :param initial_price: 初始迭代種子價格 (通常為市場當前股價)
        :param basic_shares: 基本流通股數
        :param total_debt: 總附息債務
        :param cash: 現金與約當現金
        :param cost_of_equity: 股權成本 Ke
        :param cost_of_debt: 稅前債務成本 Rd
        :param tax_rate: 有效稅率
        :param discount_fn: 依給定 WACC 計算 (Enterprise Value, Terminal Value, PV_TV) 之回調函式
        :param options_count: 未行使期權量
        :param strike_price: 期權行權價格
        :param reported_diluted_shares: 官方申報稀釋股數
        :param debt_adjust: 債務等價物調整項
        :param wacc_premium: 機構風險貼水 (小數形式，例如 0.01 代表 1%)
        :param operating_lease_liability: 資本化租賃負債
        :param preferred_stock: 特別股
        :param minority_interest: 少數股權
        :param pension_deficit: 未提撥退休金赤字
        :param net_debt: 若預先透過 EVToEquityBridge 計算淨負債，可直接傳入覆蓋
        :return: 包含收斂價格、完全稀釋股數、最終 WACC、迭代次數的結果字典
        """
        if initial_price <= 0:
            raise InvalidInputError("初始價格必須大於 0。")

        # 完整淨債務計算 (支援 EVToEquityBridge 注入)
        if net_debt is not None:
            raw_net_debt = float(net_debt)
        else:
            raw_net_debt = float(
                total_debt - cash + debt_adjust + operating_lease_liability
                + preferred_stock + minority_interest + pension_deficit
            )
        ke = float(cost_of_equity)
        kd_after_tax = float(cost_of_debt * (1.0 - tax_rate))
        # 資本結構中將附息債務與資本化租賃負債納入槓桿考量
        debt_val = max(float(total_debt + operating_lease_liability), 0.0)

        current_p = float(initial_price)
        current_wacc = 0.08  # 初始預設值
        final_diluted_shares = float(basic_shares)
        converged = False
        iteration_count = 0

        for i in range(1, self.max_iter + 1):
            iteration_count = i

            # 1. 動態 TSM 反饋：依當前迭代股價計算完全稀釋股數
            diluted_shares = TSMCalculator.calculate_diluted_shares(
                basic_shares=basic_shares,
                options_count=options_count,
                strike_price=strike_price,
                current_price=current_p,
                reported_diluted_shares=reported_diluted_shares
            )

            # 2. 計算動態股權市值 (Equity Value) 與企業總價值 (Firm Value)
            equity_val = max(current_p * diluted_shares, 1e-4)
            firm_val = equity_val + debt_val

            # 3. 計算動態資本結構權重
            we = equity_val / firm_val
            wd = debt_val / firm_val

            # 4. 更新當期 WACC (含 WACC Premium 與上下限 Clamp)
            raw_wacc = (we * ke) + (wd * kd_after_tax) + float(wacc_premium)
            current_wacc = max(min(raw_wacc, self.max_wacc), self.min_wacc)

            # 5. 呼叫外部折現邏輯，計算當期理論 Enterprise Value
            ev_calc, tv_calc, pv_tv_calc = discount_fn(current_wacc)

            # 6. 計算理論股權價值與每股隱含價格
            implied_equity_val = ev_calc - raw_net_debt
            price_calc = max(implied_equity_val / diluted_shares, 0.0)

            # 7. 阻尼因子平滑更新 (Damping Factor)
            next_p = (self.damping_factor * price_calc) + ((1.0 - self.damping_factor) * current_p)

            # 8. 收斂檢查 (相對偏差 < tol)
            rel_diff = abs(next_p - current_p) / max(current_p, 1e-4)
            if rel_diff < self.tol:
                converged = True
                current_p = next_p
                final_diluted_shares = diluted_shares
                break

            current_p = next_p
            final_diluted_shares = diluted_shares

        if not converged:
            logger.warning(
                "IterativeDCFSolver 未能在 %d 次迭代內收斂 (最終相對偏差: %.6f)，啟用安全降級回傳。",
                self.max_iter, rel_diff
            )

        # 最終輸出均嚴格轉換為 Python 原生 float，杜絕 JSON 序列化失敗
        return {
            "implied_price": float(current_p),
            "diluted_shares": float(final_diluted_shares),
            "converged_wacc": float(current_wacc),
            "is_converged": bool(converged),
            "iterations": int(iteration_count)
        }
