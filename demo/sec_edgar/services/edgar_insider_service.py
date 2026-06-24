import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd
from django.db import transaction
from edgar import Company, set_identity
from sec_edgar.models import SecInsiderTrades
from .edgar_cli_bridge import EdgarCLIBridge

logger = logging.getLogger(__name__)

class EdgarInsiderService:
    """
    內部人交易服務層。
    """
    def __init__(self):
        set_identity("sum998888@gmail.com")
        self.cli_bridge = EdgarCLIBridge()

    @transaction.atomic
    def sync_insiders_by_ticker(self, ticker: str, since: str = "12mo", limit: int = 50) -> dict:
        """
        同步特定 Ticker 的內部人交易。優先使用 pp-edgar，若失敗則 Fallback 至 edgartools。
        """
        ticker = ticker.upper()
        
        # 優先嘗試使用 pp-edgar
        if self.cli_bridge.is_available():
            try:
                logger.info(f"Syncing insider trades for {ticker} using pp-edgar (since: {since})")
                data = self.cli_bridge.insider_summary(ticker, since=since)
                
                # 假設 pp-edgar 返回的 JSON 包含一個交易清單
                # 例如: {"transactions": [{"insider": "...", "date": "...", "code": "...", "shares": ..., "price": ...}]}
                # 或者是 Go 結構對應的 JSON
                transactions = data.get("transactions", []) if isinstance(data, dict) else []
                
                if transactions:
                    trades_objs = []
                    for t in transactions:
                        # 轉換資料型態
                        txn_date = datetime.strptime(t.get("date", ""), "%Y-%m-%d").date()
                        price = Decimal(str(t.get("price", 0)))
                        shares = int(t.get("shares", 0))
                        total_val = Decimal(str(t.get("total_value", price * shares)))
                        
                        is_senior = 1 if t.get("is_senior_officer", False) or "Director" in t.get("position", "") or "Officer" in t.get("position", "") else 0
                        
                        trades_objs.append(SecInsiderTrades(
                            cik=t.get("cik", ""),
                            ticker=ticker,
                            insider_name=t.get("insider_name", t.get("insider", "")),
                            insider_title=t.get("insider_title", t.get("position", "")),
                            is_senior_officer=is_senior,
                            transaction_date=txn_date,
                            transaction_code=t.get("transaction_code", t.get("code", "S")),
                            transaction_type=t.get("transaction_type", t.get("description", "")),
                            shares=shares,
                            price_per_share=price,
                            total_value=total_val,
                            shares_owned_after=t.get("shares_owned_after", t.get("remaining_shares", None)),
                            filing_date=datetime.strptime(t.get("filing_date", ""), "%Y-%m-%d").date() if t.get("filing_date") else None,
                            source='pp-edgar'
                        ))
                    
                    if trades_objs:
                        SecInsiderTrades.objects.bulk_create(trades_objs, ignore_conflicts=True)
                        return {"status": "success", "message": f"Synced {len(trades_objs)} trades via pp-edgar", "count": len(trades_objs)}
                        
            except Exception as e:
                logger.error(f"pp-edgar failed for {ticker}, falling back to edgartools: {e}")
        
        # Fallback 方案：使用 edgartools 解析 Form 4
        return self._sync_via_edgartools(ticker, since, limit)

    def _parse_date(self, d):
        if not d:
            return None
        if isinstance(d, date):
            return d
        if isinstance(d, datetime):
            return d.date()
        try:
            return datetime.strptime(str(d).split()[0], "%Y-%m-%d").date()
        except Exception:
            return None

    def _sync_via_edgartools(self, ticker: str, since: str = "12mo", limit: int = 50) -> dict:
        """
        Fallback 方案：使用 edgartools 獲取與解析 Form 4。
        """
        logger.info(f"Syncing insider trades for {ticker} using edgartools fallback (limit: {limit})")
        
        try:
            c = Company(ticker)
            filings = c.get_filings(form="4")
            if not filings:
                return {"status": "success", "message": "No Form 4 filings found", "count": 0}
            
            # 按日期降序排序
            filings = sorted(filings, key=lambda x: x.filing_date, reverse=True)
            
            # 根據 since 進行日期過濾
            # 將 since 轉換成 cutoff 日期
            today = date.today()
            if since == "3mo":
                cutoff = today - timedelta(days=90)
            elif since == "6mo":
                cutoff = today - timedelta(days=180)
            elif since == "12mo":
                cutoff = today - timedelta(days=365)
            else:
                cutoff = today - timedelta(days=365)  # 預設 1 年
                
            target_filings = [f for f in filings if self._parse_date(f.filing_date) >= cutoff]
            target_filings = target_filings[:limit]  # 限制處理數量防過載
            
            logger.info(f"Processing {len(target_filings)} Form 4 filings via edgartools")
            
            trades_to_save = []
            
            for f in target_filings:
                try:
                    obj = f.obj()
                    df = obj.to_dataframe()
                    if df is None or df.empty:
                        continue
                        
                    f_date = self._parse_date(f.filing_date)
                    
                    for _, row in df.iterrows():
                        # 交易日
                        txn_date_str = row.get('Date')
                        if pd.isna(txn_date_str) or not txn_date_str:
                            txn_date = f_date
                        else:
                            txn_date = pd.to_datetime(txn_date_str).date()
                            
                        price_val = row.get('Price', 0)
                        shares_val = int(row.get('Shares', 0))
                        
                        price = Decimal(str(price_val)) if not pd.isna(price_val) else Decimal('0')
                        total_val = Decimal(str(row.get('Value', price * shares_val))) if not pd.isna(row.get('Value')) else price * shares_val
                        
                        pos = str(row.get('Position', '')).strip()
                        is_senior = 1 if 'Director' in pos or 'Officer' in pos or 'CEO' in pos or 'CFO' in pos or 'President' in pos else 0
                        
                        code_val = str(row.get('Code', '')).strip()
                        
                        remaining_val = row.get('Remaining Shares')
                        remaining = int(remaining_val) if not pd.isna(remaining_val) else None
                        
                        trades_to_save.append(SecInsiderTrades(
                            cik=str(f.cik),
                            ticker=ticker,
                            insider_name=str(row.get('Insider', '')),
                            insider_title=pos,
                            is_senior_officer=is_senior,
                            transaction_date=txn_date,
                            transaction_code=code_val,
                            transaction_type=str(row.get('Description', '')),
                            shares=shares_val,
                            price_per_share=price,
                            total_value=total_val,
                            shares_owned_after=remaining,
                            filing_date=f_date,
                            source='edgartools'
                        ))
                except Exception as parse_err:
                    logger.warning(f"Failed to parse Form 4 filing {f.accession_no}: {parse_err}")
            
            if trades_to_save:
                # 批量寫入資料庫
                SecInsiderTrades.objects.bulk_create(trades_to_save, batch_size=150, ignore_conflicts=True)
                return {"status": "success", "message": f"Synced {len(trades_to_save)} trades via edgartools fallback", "count": len(trades_to_save)}
                
            return {"status": "success", "message": "No transactions extracted", "count": 0}
            
        except Exception as e:
            logger.error(f"edgartools fallback failed for {ticker}: {e}")
            return {"status": "error", "message": str(e), "count": 0}
