# -*- coding: utf-8 -*-
"""
SCD Type 2 (Slowly Changing Dimension Type 2) 歷史版本控制服務
維護估值假設快照 (WACC, g, ROIC, 稅率, 利潤率等) 與估值結果歷史可追溯性。
實作悲觀鎖 (select_for_update) 與資料庫事務 (transaction.atomic) 防禦併發競態條件。
"""
import datetime
import json
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple

from django.db import transaction
from django.utils import timezone
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Q

logger = logging.getLogger(__name__)


def _sanitize_for_json(data: Any) -> Any:
    """遞迴清理並保留數值精度，安全轉換為 JSON 可序列化結構"""
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_sanitize_for_json(v) for v in data]
    elif isinstance(data, Decimal):
        return str(data)
    elif hasattr(data, 'item'):  # numpy 標量型別防禦
        return data.item()
    return data


class AssumptionHistoryService:
    """SCD Type 2 歷史版本控制核心服務"""

    @classmethod
    def save_assumptions_scd2(
        cls,
        symbol: str,
        market: str,
        assumptions: Dict[str, Any],
        valuation_snapshot: Optional[Dict[str, Any]] = None,
        change_reason: str = "自動計算更新",
        effective_date: Optional[datetime.date] = None,
        is_flagged: bool = False,
        flag_reasons: Optional[List[str]] = None,
    ) -> Tuple[Any, bool]:
        """
        儲存或升版估值假設 (SCD Type 2)。
        若既有有效版本 (is_current=True) 存在：
          - 若 assumptions 內容無實質變化，則回傳現有版本 (is_new=False)。
          - 若有變動，以原子交易 + 悲觀鎖更新舊版本 (is_current=False, end_date=今日)，
            並建立新版本 (version=舊+1, is_current=True, end_date=None)。
        """
        from valuation.models import ValuationAssumptionHistory

        symbol = str(symbol).strip().upper()
        market = str(market).strip().upper()
        eff_date = effective_date or timezone.localdate()
        clean_assump = _sanitize_for_json(assumptions)
        clean_snapshot = _sanitize_for_json(valuation_snapshot or {})
        reasons_list = list(flag_reasons or [])

        with transaction.atomic():
            # 使用悲觀鎖 select_for_update 防禦併發競態條件
            current_record = (
                ValuationAssumptionHistory.objects.select_for_update()
                .filter(symbol=symbol, market=market, is_current=True)
                .first()
            )

            if current_record:
                # 檢查假設內容是否完全一致 (以排序之 JSON 字串精確比對)
                existing_json = json.dumps(current_record.assumptions, sort_keys=True, cls=DjangoJSONEncoder)
                new_json = json.dumps(clean_assump, sort_keys=True, cls=DjangoJSONEncoder)

                if existing_json == new_json:
                    # 假設無實質變更，若估值快照或異常標記有更新則就地補充
                    update_fields = []
                    if clean_snapshot and clean_snapshot != current_record.valuation_snapshot:
                        current_record.valuation_snapshot = clean_snapshot
                        update_fields.append('valuation_snapshot')
                    if is_flagged != current_record.is_flagged:
                        current_record.is_flagged = is_flagged
                        update_fields.append('is_flagged')
                    if reasons_list != current_record.flag_reasons:
                        current_record.flag_reasons = reasons_list
                        update_fields.append('flag_reasons')
                    if update_fields:
                        current_record.save(update_fields=update_fields)
                    return current_record, False

                # 歸檔舊版本：標記為歷史紀錄，設定失效日為新版本之生效日
                current_record.is_current = False
                current_record.end_date = eff_date
                current_record.save(update_fields=['is_current', 'end_date'])
                next_version = current_record.version + 1
            else:
                next_version = 1

            new_record = ValuationAssumptionHistory.objects.create(
                symbol=symbol,
                market=market,
                version=next_version,
                effective_date=eff_date,
                end_date=None,
                is_current=True,
                assumptions=clean_assump,
                valuation_snapshot=clean_snapshot,
                change_reason=change_reason,
                is_flagged=is_flagged,
                flag_reasons=reasons_list,
            )
            logger.info(
                f"[SCD2] 成功建立 {symbol} ({market}) 估值假設版本 v{next_version}，生效日：{eff_date} (is_flagged={is_flagged})"
            )
            return new_record, True

    @classmethod
    def get_current(cls, symbol: str, market: str) -> Optional[Any]:
        """查詢特定股票之當前有效假設快照"""
        from valuation.models import ValuationAssumptionHistory

        symbol = str(symbol).strip().upper()
        market = str(market).strip().upper()
        return ValuationAssumptionHistory.objects.filter(
            symbol=symbol, market=market, is_current=True
        ).first()

    @classmethod
    def get_as_of_date(cls, symbol: str, market: str, target_date: datetime.date) -> Optional[Any]:
        """
        Time-travel 查詢：取得指定歷史日期當下生效之假設版本。
        時間邊界條件：effective_date <= target_date AND (end_date IS NULL OR end_date > target_date)
        """
        from valuation.models import ValuationAssumptionHistory

        symbol = str(symbol).strip().upper()
        market = str(market).strip().upper()
        return ValuationAssumptionHistory.objects.filter(
            Q(symbol=symbol, market=market) &
            Q(effective_date__lte=target_date) &
            (Q(end_date__isnull=True) | Q(end_date__gt=target_date))
        ).order_by('-version').first()

    @classmethod
    def get_history(
        cls,
        symbol: str,
        market: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Any]:
        """分頁查詢歷史版本列表 (預設依版本由新至舊排序)"""
        from valuation.models import ValuationAssumptionHistory

        symbol = str(symbol).strip().upper()
        market = str(market).strip().upper()
        return list(
            ValuationAssumptionHistory.objects.filter(symbol=symbol, market=market)
            .order_by('-version')[offset:offset + limit]
        )

    @classmethod
    def rollback_to_version(
        cls,
        symbol: str,
        market: str,
        target_version: int,
        reason: str = ""
    ) -> Any:
        """
        不可變回滾：透過創建新版本來回滾至歷史某個特定版本之假設，確保稽核歷程完整。
        """
        from valuation.models import ValuationAssumptionHistory

        symbol = str(symbol).strip().upper()
        market = str(market).strip().upper()
        target_rec = ValuationAssumptionHistory.objects.filter(
            symbol=symbol, market=market, version=target_version
        ).first()
        if not target_rec:
            raise ValueError(f"找不到 {symbol} ({market}) 的歷史版本 v{target_version}")

        rb_reason = f"回滾至歷史版本 v{target_version}"
        if reason:
            rb_reason += f"（原因：{reason}）"

        new_rec, _ = cls.save_assumptions_scd2(
            symbol=symbol,
            market=market,
            assumptions=target_rec.assumptions,
            valuation_snapshot=target_rec.valuation_snapshot,
            change_reason=rb_reason,
        )
        return new_rec
