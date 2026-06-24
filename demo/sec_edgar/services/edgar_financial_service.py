import logging
import re
from datetime import datetime
from decimal import Decimal
import pandas as pd
from django.db import transaction
from edgar import Company, set_identity
from sec_edgar.models import SecFinancialXbrl

logger = logging.getLogger(__name__)

class EdgarFinancialService:
    """
    標準化 XBRL 財務數據提取服務。
    """
    def __init__(self):
        set_identity("sum998888@gmail.com")

    @transaction.atomic
    def sync_financials_by_ticker(self, ticker: str) -> dict:
        """
        同步特定 Ticker 的歷史標準化財務數據。
        """
        ticker = ticker.upper()
        logger.info(f"Syncing financial statements for {ticker} using edgartools")
        
        try:
            c = Company(ticker)
            cik = str(c.cik)
            fins = c.get_financials()
            
            statements = []
            # 依序抓取三大表
            try:
                statements.append(("IS", fins.income_statement()))
            except Exception as e:
                logger.warning(f"No income statement for {ticker}: {e}")
            try:
                statements.append(("BS", fins.balance_sheet()))
            except Exception as e:
                logger.warning(f"No balance sheet for {ticker}: {e}")
            try:
                statements.append(("CF", fins.cashflow_statement() or fins.cash_flow_statement()))
            except Exception as e:
                logger.warning(f"No cashflow statement for {ticker}: {e}")
                
            if not statements:
                return {"status": "success", "message": "No financial statements found", "count": 0}
                
            financial_records = []
            
            # 用於匹配 YYYY-MM-DD (TYPE) 欄位的正則表達式
            period_col_pat = re.compile(r'^(\d{4}-\d{2}-\d{2})\s+\((FY|Q1|Q2|Q3|Q4)\)$')
            
            for statement_type, stmt in statements:
                if stmt is None:
                    continue
                
                df = stmt.to_dataframe()
                if df is None or df.empty:
                    continue
                
                # 尋找所有屬於財務報告期的 columns
                period_cols = []
                for col in df.columns:
                    m = period_col_pat.match(str(col))
                    if m:
                        period_cols.append((col, m.group(1), m.group(2)))
                
                # 遍歷這些 columns
                for col_name, date_str, period_type in period_cols:
                    period_end = datetime.strptime(date_str, "%Y-%m-%d").date()
                    fiscal_year = period_end.year
                    
                    # 決定季度編號
                    if period_type == 'FY':
                        fiscal_quarter = 0
                        form_type = '10-K'
                    else:
                        # Q1 -> 1, Q2 -> 2, etc.
                        fiscal_quarter = int(period_type[1])
                        form_type = '10-Q'
                        
                    # 遍歷每一行提取概念與數值
                    for _, row in df.iterrows():
                        concept = row.get('standard_concept')
                        if pd.isna(concept) or not concept:
                            concept = row.get('concept')  # Fallback to raw concept
                            
                        if pd.isna(concept) or not concept:
                            continue
                            
                        # 移除概念名稱中的前綴（如 us-gaap_）使其更標準化
                        concept_clean = str(concept).split('_')[-1]
                        
                        val = row.get(col_name)
                        if pd.isna(val) or val is None or str(val).strip() == '':
                            continue
                            
                        try:
                            # 轉換為 Decimal 以確保 financial 運算的精度
                            decimal_val = Decimal(str(val))
                            
                            financial_records.append(SecFinancialXbrl(
                                ticker=ticker,
                                cik=cik,
                                period_end=period_end,
                                fiscal_year=fiscal_year,
                                fiscal_quarter=fiscal_quarter,
                                concept=concept_clean,
                                value=decimal_val,
                                unit='USD',
                                form_type=form_type
                            ))
                        except Exception as val_err:
                            # 忽略無法解析為數字的行
                            continue
            
            if financial_records:
                # 批次寫入資料庫
                SecFinancialXbrl.objects.bulk_create(financial_records, batch_size=150, ignore_conflicts=True)
                return {"status": "success", "message": f"Synced {len(financial_records)} financial data points", "count": len(financial_records)}
                
            return {"status": "success", "message": "No financial records extracted", "count": 0}
            
        except Exception as e:
            logger.error(f"Failed to sync financials for {ticker}: {e}")
            return {"status": "error", "message": str(e), "count": 0}
