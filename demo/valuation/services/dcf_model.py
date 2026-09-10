import logging
from typing import List, Dict, Any, Optional
from .assumptions import Assumptions
from .iterative_solver import FCFType, ValuationError

logger = logging.getLogger("ValuationEngine.DCFModel")


class DCFModel:
    """
    具備嚴格型別硬綁定防護的 DCF 折現模型。
    
    規則約束：
    - FCFF (Firm Free Cash Flow) 必須配對 WACC 進行企業價值 (EV) 折現。
    - FCFE (Equity Free Cash Flow) 必須配對 Ke (Cost of Equity) 進行股權價值 (Equity Value) 折現。
    若出現型別錯配，主動拋出 TypeError，防止模型混用。
    """

    def __init__(
        self,
        fcf_type: FCFType = FCFType.FCFF,
        discount_rate_type: str = "WACC",
        discount_rate: float = 0.08,
        discount_convention: str = "end_of_year"
    ):
        self.fcf_type = fcf_type
        self.discount_rate_type = discount_rate_type.upper()
        self.discount_rate = float(discount_rate)
        self.discount_convention = discount_convention
        self._validate_type_binding()

    def _validate_type_binding(self):
        """
        強制型別與折現率配對檢驗。
        """
        if self.fcf_type == FCFType.FCFF and self.discount_rate_type not in ["WACC", "COST_OF_CAPITAL"]:
            raise TypeError(
                f"型別綁定錯誤：FCFF 估值必須使用加權平均資金成本 (WACC)，不可使用 '{self.discount_rate_type}'。"
            )
        if self.fcf_type == FCFType.FCFE and self.discount_rate_type not in ["KE", "COST_OF_EQUITY"]:
            raise TypeError(
                f"型別綁定錯誤：FCFE 估值必須使用股權成本 (Ke)，不可使用 '{self.discount_rate_type}'。"
            )

    def discount_cash_flows(
        self,
        cash_flows: List[float],
        terminal_value: float,
        rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        折現現金流並計算現值。

        :param cash_flows: 預測期各期現金流量串流
        :param terminal_value: 終值 (Terminal Value)
        :param rate: 指定折現率 (若無則使用初始化值)
        :return: {'pv_cash_flows': float, 'pv_terminal_value': float, 'total_pv': float}
        """
        r = float(rate if rate is not None else self.discount_rate)
        if r <= 0:
            raise ValuationError(f"折現率必須大於 0 (收到: {r})。")

        is_mid_year = (self.discount_convention == "mid_year")
        pv_cfs = 0.0
        n_years = len(cash_flows)

        for i, cf in enumerate(cash_flows):
            t = (i + 0.5) if is_mid_year else (i + 1.0)
            pv_cfs += float(cf) / ((1.0 + r) ** t)

        t_tv = (n_years - 0.5) if is_mid_year else float(n_years)
        pv_tv = float(terminal_value) / ((1.0 + r) ** t_tv)

        return {
            "pv_cash_flows": float(pv_cfs),
            "pv_terminal_value": float(pv_tv),
            "total_pv": float(pv_cfs + pv_tv)
        }
