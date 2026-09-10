import math
import logging
from decimal import Decimal
from typing import Dict, Any, List

logger = logging.getLogger("ValuationEngine.EVBridge")


class EVToEquityBridge:
    """
    企業價值 (Enterprise Value, EV) 至股權價值 (Equity Value) 完整橋樑計算器。
    
    橋接公式 (Damodaran & Institutional Standard):
    Equity Value = Enterprise Value (EV)
                 + 現金及約當現金 (Cash & Equivalents)
                 - 總附息債務 (Total Debt)
                 - 資本化租賃負債 (Capitalized Lease Liabilities, ASC 842 / IFRS 16)
                 - 特別股 (Preferred Stock)
                 - 少數股權 / 非控制權益 (Minority Interest)
                 - 未提撥退休金赤字 (Unfunded Pension Deficit)
                 - 其他債務等價物 (Other Debt-like Items)
    """

    @staticmethod
    def _to_decimal(value: Any, default: str = "0.0") -> Decimal:
        """安全轉換各類型數值為 Decimal，防止 NaN、None 與二進位浮點數精度外洩"""
        if value is None:
            return Decimal(default)
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return Decimal(default)
            return Decimal(str(round(value, 4)))
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal(default)

    @classmethod
    def calculate_bridge(
        cls,
        enterprise_value: float,
        cash: float = 0.0,
        total_debt: float = 0.0,
        operating_lease_liability: float = 0.0,
        preferred_stock: float = 0.0,
        minority_interest: float = 0.0,
        pension_deficit: float = 0.0,
        debt_like_items: float = 0.0
    ) -> Dict[str, Any]:
        """
        執行高精度橋接運算，並輸出結構化明細與防禦性結果。
        """
        d_ev = cls._to_decimal(enterprise_value)
        d_cash = max(cls._to_decimal(cash), Decimal("0.0"))
        d_debt = max(cls._to_decimal(total_debt), Decimal("0.0"))
        d_leases = max(cls._to_decimal(operating_lease_liability), Decimal("0.0"))
        d_pref = max(cls._to_decimal(preferred_stock), Decimal("0.0"))
        d_minority = max(cls._to_decimal(minority_interest), Decimal("0.0"))
        d_pension = max(cls._to_decimal(pension_deficit), Decimal("0.0"))
        d_debt_like = max(cls._to_decimal(debt_like_items), Decimal("0.0"))

        # 總扣除項（廣義總負債與股權優先要求權）
        d_total_deductions = d_debt + d_leases + d_pref + d_minority + d_pension + d_debt_like
        # 廣義淨負債 (Net Debt Bridge)
        d_net_debt = d_total_deductions - d_cash

        # 理論股權價值
        d_equity_value = max(d_ev - d_net_debt, Decimal("0.0"))

        # 建立明細項目清單 (符合 UI 零破壞與審計對帳)
        items: List[Dict[str, Any]] = [
            {
                "key": "enterprise_value",
                "label": "企業價值 (EV)",
                "amount": float(d_ev),
                "category": "BASE",
                "description": "DCF 營運現金流與終端價值之折現總額"
            },
            {
                "key": "cash",
                "label": "現金及約當現金",
                "amount": float(d_cash),
                "category": "ADD",
                "description": "非營運性資產加回"
            },
            {
                "key": "total_debt",
                "label": "總附息債務",
                "amount": float(d_debt),
                "category": "DEDUCT",
                "description": "短期借款、應付公司債與長期借款"
            },
            {
                "key": "operating_lease_liability",
                "label": "資本化租賃負債",
                "amount": float(d_leases),
                "category": "DEDUCT",
                "description": "IFRS 16 / ASC 842 資本化使用權資產對應之租賃負債"
            },
            {
                "key": "preferred_stock",
                "label": "特別股",
                "amount": float(d_pref),
                "category": "DEDUCT",
                "description": "清算受償順位高於普通股之特別股股本"
            },
            {
                "key": "minority_interest",
                "label": "少數股權 / 非控制權益",
                "amount": float(d_minority),
                "category": "DEDUCT",
                "description": "合併報表中不歸屬於母公司普通股股東之子公司權益"
            },
            {
                "key": "pension_deficit",
                "label": "未提撥退休金赤字",
                "amount": float(d_pension),
                "category": "DEDUCT",
                "description": "確定福利計畫義務現值超出計畫資產公允價值之短絀"
            },
            {
                "key": "debt_like_items",
                "label": "其他債務等價物",
                "amount": float(d_debt_like),
                "category": "DEDUCT",
                "description": "訴訟準備、未決稅務或機構調校調整項"
            }
        ]

        return {
            "enterprise_value": float(d_ev),
            "equity_value": float(d_equity_value),
            "cash": float(d_cash),
            "total_debt": float(d_debt),
            "operating_lease_liability": float(d_leases),
            "preferred_stock": float(d_pref),
            "minority_interest": float(d_minority),
            "pension_deficit": float(d_pension),
            "debt_like_items": float(d_debt_like),
            "total_deductions": float(d_total_deductions),
            "net_debt": float(d_net_debt),
            "items": items
        }
