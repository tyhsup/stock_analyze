import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# Phase 6: WACC 異常值防護閥 (WACC Anomaly Protection Guardrail)
# ==============================================================================

# 金融常數與邊界定義 (依據 NotebookLM 與巴菲特/高盛估值標準)
MIN_WACC_LIMIT: float = 0.03   # 3.0% 最低資本成本底線，防止分母趨零產生極端高估
MAX_WACC_LIMIT: float = 0.20   # 20.0% 最高資本成本上限，防止過度折現導致企業價值歸零
MIN_RF_LIMIT: float = 0.005    # 0.5% 無風險利率最低底線
MAX_RF_LIMIT: float = 0.080    # 8.0% 無風險利率最高上限
MIN_WACC_RF_SPREAD: float = 0.005  # 0.5% WACC 與無風險利率之最小價差 (Equity/Debt Risk Premium)
DEFAULT_FALLBACK_WACC: float = 0.085  # 8.5% 歷史長期加權平均預設資本成本


@dataclass
class WACCGuardrailResult:
    """
    WACC 異常值防護閥輸出結構體。
    所有數值強制轉為原生 Python 型別，杜絕 JSON 序列化失敗。
    """
    wacc: float
    raw_wacc: float
    base_wacc: float
    rf: float
    is_flagged: bool
    flag_reasons: List[str] = field(default_factory=list)
    is_clamped: bool = False
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wacc": float(self.wacc),
            "raw_wacc": float(self.raw_wacc),
            "base_wacc": float(self.base_wacc),
            "rf": float(self.rf),
            "is_flagged": bool(self.is_flagged),
            "flag_reasons": list(self.flag_reasons),
            "is_clamped": bool(self.is_clamped),
            "fallback_used": bool(self.fallback_used),
        }


class WACCGuardrail:
    """
    WACC 異常值防護閥服務。
    負責檢驗並修正加權平均資本成本 (WACC)、動態無風險利率 (Rf) 與終端增長率 (g) 之邏輯邊界。
    """

    @classmethod
    def validate_rf(cls, rf: float, market: str = 'tw') -> Tuple[float, bool, List[str]]:
        """
        動態無風險利率邊界校驗。
        :param rf: 原始無風險利率 (例如 0.042)
        :param market: 市場別 ('tw' 或 'us')
        :return: (clamped_rf, is_flagged, reasons)
        """
        reasons = []
        is_flagged = False
        safe_rf = float(rf)

        # 1. 數值異常檢查 (NaN / None / 負數)
        if safe_rf is None or safe_rf != safe_rf:
            safe_rf = 0.015 if market == 'tw' else 0.042
            is_flagged = True
            reasons.append("RISK_FREE_RATE_INVALID_NAN")
            return safe_rf, is_flagged, reasons

        # 2. 邊界檢測 [0.5%, 8.0%]
        if safe_rf < MIN_RF_LIMIT:
            logger.warning(f"[WACCGuardrail] Rf ({safe_rf:.2%}) 低於合理下限 {MIN_RF_LIMIT:.2%}，自動調整。")
            safe_rf = MIN_RF_LIMIT
            is_flagged = True
            reasons.append(f"RISK_FREE_RATE_BELOW_LOWER_BOUND_{MIN_RF_LIMIT:.1%}")
        elif safe_rf > MAX_RF_LIMIT:
            logger.warning(f"[WACCGuardrail] Rf ({safe_rf:.2%}) 超過合理上限 {MAX_RF_LIMIT:.2%}，自動調整。")
            safe_rf = MAX_RF_LIMIT
            is_flagged = True
            reasons.append(f"RISK_FREE_RATE_ABOVE_UPPER_BOUND_{MAX_RF_LIMIT:.1%}")

        return safe_rf, is_flagged, reasons

    @classmethod
    def validate_and_clamp(
        cls,
        raw_wacc: float,
        base_wacc: float = None,
        rf: float = 0.042,
        g: float = 0.03,
        market: str = 'tw'
    ) -> WACCGuardrailResult:
        """
        對計算出的 WACC 執行完整異常防護閥檢查、Clamp 截斷與標記。
        """
        flag_reasons = []
        is_flagged = False
        is_clamped = False

        # 1. 處理無效數值 (NaN / None / Inf)
        try:
            val_wacc = float(raw_wacc)
            if val_wacc != val_wacc or val_wacc == float('inf') or val_wacc == float('-inf'):
                raise ValueError("WACC is NaN or Inf")
        except Exception as e:
            logger.error(f"[WACCGuardrail] WACC 輸入數值異常: {e}，啟用 Fallback")
            return cls.fallback_wacc(reason=f"INVALID_NUMERICAL_VALUE: {str(e)}")

        base_val = float(base_wacc) if base_wacc is not None else val_wacc

        # 2. 校驗無風險利率
        validated_rf, rf_flagged, rf_reasons = cls.validate_rf(rf, market=market)
        if rf_flagged:
            is_flagged = True
            flag_reasons.extend(rf_reasons)

        final_wacc = val_wacc

        # 3. 核心區間檢查 [3.0%, 20.0%]
        if final_wacc < MIN_WACC_LIMIT:
            logger.warning(
                f"[WACCGuardrail] WACC ({final_wacc:.2%}) 低於核心下限 {MIN_WACC_LIMIT:.1%}，強制截斷至 {MIN_WACC_LIMIT:.1%}"
            )
            final_wacc = MIN_WACC_LIMIT
            is_flagged = True
            is_clamped = True
            flag_reasons.append(f"WACC_BELOW_LOWER_BOUND_{MIN_WACC_LIMIT:.0%}")
        elif final_wacc > MAX_WACC_LIMIT:
            logger.warning(
                f"[WACCGuardrail] WACC ({final_wacc:.2%}) 超出核心上限 {MAX_WACC_LIMIT:.0%}，強制截斷至 {MAX_WACC_LIMIT:.0%}"
            )
            final_wacc = MAX_WACC_LIMIT
            is_flagged = True
            is_clamped = True
            flag_reasons.append(f"WACC_ABOVE_UPPER_BOUND_{MAX_WACC_LIMIT:.0%}")

        # 4. 資本成本溢酬邊界：WACC 必須 >= Rf + 0.5%
        if final_wacc < (validated_rf + MIN_WACC_RF_SPREAD):
            logger.warning(
                f"[WACCGuardrail] WACC ({final_wacc:.2%}) 未達無風險利率價差要求 (Rf {validated_rf:.2%} + {MIN_WACC_RF_SPREAD:.2%})"
            )
            is_flagged = True
            flag_reasons.append("WACC_BELOW_RISK_FREE_SPREAD")
            # 在不超過上限的前提下微調
            adjusted_wacc = min(validated_rf + MIN_WACC_RF_SPREAD, MAX_WACC_LIMIT)
            if adjusted_wacc > final_wacc:
                final_wacc = adjusted_wacc
                is_clamped = True

        # 5. 終端永續增長率勾稽：WACC 必須 >= g + 0.5% (戈登增長模型分母防呆)
        try:
            val_g = float(g)
            if final_wacc <= (val_g + 0.005):
                logger.warning(
                    f"[WACCGuardrail] WACC ({final_wacc:.2%}) 小於或接近永續增長率 g ({val_g:.2%})，觸發估值分母保護告警"
                )
                is_flagged = True
                flag_reasons.append("WACC_INSUFFICIENT_SPREAD_OVER_G")
        except Exception:
            pass

        return WACCGuardrailResult(
            wacc=float(round(final_wacc, 6)),
            raw_wacc=float(round(val_wacc, 6)),
            base_wacc=float(round(base_val, 6)),
            rf=float(round(validated_rf, 6)),
            is_flagged=is_flagged,
            flag_reasons=flag_reasons,
            is_clamped=is_clamped,
            fallback_used=False
        )

    @classmethod
    def fallback_wacc(cls, reason: str = "WACC_CALCULATION_EXCEPTION") -> WACCGuardrailResult:
        """
        當運算發生致命錯誤時之安全保底機制 (Fallback)。
        回傳預設資本成本 (8.5%) 並標記異常。
        """
        logger.error(f"[WACCGuardrail] 啟用 Fallback WACC: {DEFAULT_FALLBACK_WACC:.1%}, 原因: {reason}")
        return WACCGuardrailResult(
            wacc=DEFAULT_FALLBACK_WACC,
            raw_wacc=DEFAULT_FALLBACK_WACC,
            base_wacc=DEFAULT_FALLBACK_WACC,
            rf=0.042,
            is_flagged=True,
            flag_reasons=[reason, "FALLBACK_WACC_APPLIED"],
            is_clamped=True,
            fallback_used=True
        )
