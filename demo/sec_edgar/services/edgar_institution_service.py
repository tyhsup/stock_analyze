import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
from django.db import connection, transaction
from edgar import set_identity, Company
from sec_edgar.models import SecFilingIndex, Sec13fHoldings

logger = logging.getLogger(__name__)

class EdgarInstitutionService:
    """
    以投資機構為中心的 13F 持股服務層。
    """
    
    # 預設 10 家知名投資機構
    KNOWN_INSTITUTIONS = {
        '0001067983': {'name_en': 'Berkshire Hathaway Inc', 'name_zh': '巴菲特 (Berkshire Hathaway)'},
        '0001336528': {'name_en': 'Bridgewater Associates LP', 'name_zh': '橋水基金 (Bridgewater)'},
        '0001350694': {'name_en': 'Renaissance Technologies LLC', 'name_zh': '文藝復興科技 (Renaissance)'},
        '0001061768': {'name_en': 'Citadel Advisors LLC', 'name_zh': '城堡投資 (Citadel)'},
        '0001037389': {'name_en': 'BlackRock Inc', 'name_zh': '貝萊德 (BlackRock)'},
        '0000921669': {'name_en': 'Soros Fund Management LLC', 'name_zh': '索羅斯 (Soros Fund)'},
        '0001364742': {'name_en': 'Two Sigma Investments LP', 'name_zh': '雙西格瑪 (Two Sigma)'},
        '0001167483': {'name_en': 'D.E. Shaw & Co', 'name_zh': 'D.E. Shaw'},
        '0001056831': {'name_en': 'AQR Capital Management LLC', 'name_zh': 'AQR Capital'},
        '0001649339': {'name_en': 'Tiger Global Management LLC', 'name_zh': '老虎環球 (Tiger Global)'},
    }

    def __init__(self):
        # 確保初始化身分識別
        set_identity("sum998888@gmail.com")

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

    @transaction.atomic
    def sync_institution_13f(self, cik: str, quarters_limit: int = 1) -> dict:
        """
        以機構為中心，同步該機構在 SEC EDGAR 上的 13F 機構持股數據。
        """
        cik = str(cik).strip().zfill(10)
        
        if cik not in self.KNOWN_INSTITUTIONS:
            logger.warning(f"CIK {cik} is not in KNOWN_INSTITUTIONS, but attempting sync anyway.")

        logger.info(f"Syncing 13F holdings for CIK: {cik}")
        
        try:
            company = Company(cik)
            filings = company.get_filings(form="13F-HR")
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return {"status": "error", "message": f"Failed to fetch filings for CIK {cik}: {str(e)}", "count": 0}

        if len(filings) == 0:
            return {"status": "success", "message": f"No 13F filings found for CIK {cik}", "count": 0}

        # 依日期降序排列
        filings = sorted(filings, key=lambda x: x.filing_date, reverse=True)
        
        # 過濾特定季度數量
        today = date.today()
        days_limit = quarters_limit * 100
        cutoff_date = today - timedelta(days=days_limit)
        
        target_filings = [f for f in filings if self._parse_date(f.filing_date) >= cutoff_date]
        
        if not target_filings:
            # Fallback: 若因時間過濾沒半個，強迫留最近 1 個
            target_filings = filings[:1]

        logger.info(f"Filtered to {len(target_filings)} filings to process for CIK {cik}")

        synced_count = 0
        
        for f in target_filings:
            period_dt = None
            try:
                holdings_obj = f.obj()
                if hasattr(holdings_obj, 'report_period'):
                    period_dt = self._parse_date(holdings_obj.report_period)
            except Exception as e:
                logger.warning(f"Could not parse report period for filing {f.accession_no}: {e}")
                
            filing_index, created = SecFilingIndex.objects.get_or_create(
                accession_no=f.accession_no,
                defaults={
                    'cik': cik,
                    'ticker': None,  # 機構級申報不屬於單一個股
                    'company_name': f.company,
                    'form_type': f.form,
                    'filing_date': self._parse_date(f.filing_date),
                    'period_of_report': period_dt,
                    'primary_doc_url': f"https://www.sec.gov/Archives/edgar/data/{f.cik}/{f.accession_no.replace('-', '')}/{f.accession_no}.txt"
                }
            )

            try:
                holdings_obj = f.obj()
                df = holdings_obj.holdings
                
                if df.empty:
                    continue
                
                holdings_to_save = []
                for _, row in df.iterrows():
                    cusip_val = str(row.get('Cusip', '')).strip()
                    if not cusip_val:
                        continue
                        
                    ticker_val = str(row.get('Ticker', '')).strip().upper()
                    if pd.isna(row.get('Ticker')) or not ticker_val or ticker_val == 'NAN' or ticker_val == 'NONE':
                        ticker_val = None
                        
                    put_call_val = str(row.get('PutCall', '')).strip().upper()
                    if pd.isna(row.get('PutCall')) or not put_call_val or put_call_val == 'NAN' or put_call_val == 'NONE':
                        put_call_val = ''
                        
                    shares_val = int(row.get('SharesPrnAmount', 0))
                    value_val = int(row.get('Value', 0))
                    
                    holdings_to_save.append(Sec13fHoldings(
                        filing=filing_index,
                        cik=cik,
                        ticker=ticker_val,
                        cusip=cusip_val,
                        security_name=str(row.get('Issuer', '')),
                        shares=shares_val,
                        value_usd=value_val,
                        put_call=put_call_val,
                        investment_discretion=str(row.get('InvestmentDiscretion', 'SOLE')),
                        period_of_report=period_dt or filing_index.filing_date,
                    ))
                    synced_count += 1
                
                if holdings_to_save:
                    # 批量寫入
                    Sec13fHoldings.objects.bulk_create(holdings_to_save, batch_size=200, ignore_conflicts=True)
                    
            except Exception as parse_err:
                logger.error(f"Error parsing holdings for filing {f.accession_no}: {parse_err}")

        # 計算季度持股變動
        if synced_count > 0:
            self.calculate_quarterly_changes_for_institution(cik)

        return {"status": "success", "message": f"Sync completed for CIK {cik}", "count": synced_count}

    def calculate_quarterly_changes_for_institution(self, cik: str):
        """
        計算特定 CIK 所有持股的季度增減變動。
        """
        cik = str(cik).strip().zfill(10)
        df_cols = ['id', 'cusip', 'shares', 'period_of_report', 'put_call']
        holdings_qs = Sec13fHoldings.objects.filter(cik=cik).values(*df_cols)
        if not holdings_qs.exists():
            return
            
        df = pd.DataFrame(list(holdings_qs))
        df['period_of_report'] = pd.to_datetime(df['period_of_report'])
        
        # 排序
        df = df.sort_values(by=['cusip', 'put_call', 'period_of_report']).copy()
        
        # 計算上一期持股股數
        df['prev_shares'] = df.groupby(['cusip', 'put_call'])['shares'].shift(1)
        df['shares_change'] = df['shares'] - df['prev_shares']
        
        df['change_pct'] = 0.0
        mask_has_prev = (df['prev_shares'].notna()) & (df['prev_shares'] > 0)
        df.loc[mask_has_prev, 'change_pct'] = (df.loc[mask_has_prev, 'shares_change'] / df.loc[mask_has_prev, 'prev_shares']) * 100
        
        df['action_type'] = 'UNCHANGED'
        df.loc[df['prev_shares'].isna(), 'action_type'] = 'NEW'
        df.loc[(df['shares_change'] > 0) & (df['prev_shares'].notna()), 'action_type'] = 'INCREASED'
        df.loc[(df['shares_change'] < 0) & (df['prev_shares'].notna()), 'action_type'] = 'DECREASED'
        df.loc[(df['shares_change'] == 0) & (df['prev_shares'].notna()), 'action_type'] = 'UNCHANGED'
        
        # 清倉偵測
        all_periods = sorted(df['period_of_report'].unique())
        sold_records = []
        
        for (cusip, put_call), group in df.groupby(['cusip', 'put_call']):
            group_periods = set(group['period_of_report'])
            sorted_group = group.sort_values(by='period_of_report')
            for i in range(len(all_periods) - 1):
                curr_p = all_periods[i]
                next_p = all_periods[i+1]
                
                if curr_p in group_periods and next_p not in group_periods:
                    curr_row = sorted_group[sorted_group['period_of_report'] == curr_p].iloc[0]
                    if curr_row['shares'] > 0:
                        filing = SecFilingIndex.objects.filter(
                            cik=cik, 
                            period_of_report=next_p.date(),
                            form_type='13F-HR'
                        ).first()
                        
                        if filing:
                            # 尋找現有的 ticker
                            ticker_val = Sec13fHoldings.objects.filter(cik=cik, cusip=cusip).exclude(ticker__isnull=True).values_list('ticker', flat=True).first()
                            sold_records.append(Sec13fHoldings(
                                filing=filing,
                                cik=cik,
                                ticker=ticker_val,
                                cusip=cusip,
                                security_name=curr_row.get('security_name', ''),
                                shares=0,
                                value_usd=0,
                                put_call=put_call,
                                period_of_report=next_p.date(),
                                prev_shares=curr_row['shares'],
                                shares_change=-curr_row['shares'],
                                change_pct=-100.0,
                                action_type='SOLD_ALL'
                            ))
                            
        if sold_records:
            Sec13fHoldings.objects.bulk_create(sold_records, ignore_conflicts=True)
            
        updates = []
        for _, row in df.iterrows():
            if pd.isna(row['prev_shares']):
                prev_s = None
                s_change = None
                c_pct = None
            else:
                prev_s = int(row['prev_shares'])
                s_change = int(row['shares_change'])
                c_pct = Decimal(str(round(row['change_pct'], 4)))
                
            obj = Sec13fHoldings(
                id=int(row['id']),
                prev_shares=prev_s,
                shares_change=s_change,
                change_pct=c_pct,
                action_type=row['action_type']
            )
            updates.append(obj)
            
        if updates:
            Sec13fHoldings.objects.bulk_update(
                updates, 
                ['prev_shares', 'shares_change', 'change_pct', 'action_type'],
                batch_size=200
            )

    def get_institution_holdings(self, cik: str, period: date = None, limit: int = 200) -> list:
        """
        從資料庫獲取特定 CIK 投資機構在指定報告季度的持有股清單，並計算占比。
        """
        cik = str(cik).strip().zfill(10)
        
        if not period:
            latest_holding = Sec13fHoldings.objects.filter(cik=cik).order_by('-period_of_report').first()
            if not latest_holding:
                return []
            period = latest_holding.period_of_report
            
        holdings_qs = Sec13fHoldings.objects.filter(
            cik=cik,
            period_of_report=period
        ).order_by('-value_usd')[:limit]
        
        # 計算總市值以計算權重百分比
        total_value_usd = sum(h.value_usd for h in holdings_qs)
        
        results = []
        for h in holdings_qs:
            weight = (h.value_usd / total_value_usd * 100) if total_value_usd > 0 else 0.0
            results.append({
                'ticker': h.ticker or '',
                'cusip': h.cusip or '',
                'security_name': h.security_name or '',
                'shares': h.shares,
                'value_usd': h.value_usd * 1000,  # 轉回實際美元
                'put_call': h.put_call or '',
                'weight': round(weight, 4),
                'change_pct': float(h.change_pct) if h.change_pct is not None else None,
                'action_type': h.action_type or '',
                'period_of_report': h.period_of_report.strftime('%Y-%m-%d')
            })
        return results
