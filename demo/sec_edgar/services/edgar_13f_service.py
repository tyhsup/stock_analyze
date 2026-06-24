import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd
from django.db import connection, transaction
from edgar import search_filings, set_identity
from market_data.models import StockUS
from sec_edgar.models import SecFilingIndex, Sec13fHoldings

logger = logging.getLogger(__name__)

class Edgar13FService:
    """
    以股票為中心的 13F 機構持股服務層。
    """
    def __init__(self):
        # 確保初始化身分識別
        set_identity("sum998888@gmail.com")

    def get_cusip_by_ticker(self, ticker: str) -> str:
        """
        透過全文檢索動態映射 Ticker 到 CUSIP。
        """
        ticker = ticker.upper()
        
        # 優先從資料庫的 sec_13f_holdings 找尋已有紀錄的 CUSIP
        existing = Sec13fHoldings.objects.filter(ticker=ticker).first()
        if existing and existing.cusip:
            return existing.cusip

        # 若資料庫沒有，嘗試透過 StockUS name 或 symbol 進行 search
        stock_name = ticker
        try:
            stock = StockUS.objects.filter(symbol=ticker).first()
            if stock and stock.name:
                stock_name = stock.name
        except Exception as e:
            logger.warning(f"Failed to query StockUS name for {ticker}: {e}")

        # 使用公司名稱 + 13F-HR 全文搜尋
        search_query = f"{stock_name} 13F-HR"
        logger.info(f"Mapping CUSIP for {ticker} using query: {search_query}")
        
        try:
            results = search_filings(search_query)
            if results.total > 0:
                filings = results.to_filings()
                # 嘗試解析最近的幾筆 13F，獲取 CUSIP
                for f in filings[:5]:
                    try:
                        holdings = f.obj()
                        df = holdings.holdings
                        # 比對 Ticker
                        matched = df[df['Ticker'].str.upper() == ticker]
                        if not matched.empty:
                            cusip = matched.iloc[0]['Cusip']
                            logger.info(f"Mapped Ticker {ticker} to CUSIP: {cusip}")
                            return cusip
                    except Exception as parse_err:
                        logger.warning(f"Error parsing filing {f.accession_no} for CUSIP: {parse_err}")
        except Exception as e:
            logger.error(f"Error searching filings for CUSIP mapping of {ticker}: {e}")

        # 備用方案：若找不到，拋出錯誤或返回預設，但在 13F 中通常以此為基礎
        raise ValueError(f"Unable to resolve CUSIP for ticker: {ticker}")

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
    def sync_holdings_by_ticker(self, ticker: str, quarters_limit: int = 1) -> dict:
        """
        以股票為中心，同步該股票在 SEC EDGAR 上的機構持股數據。
        """
        ticker = ticker.upper()
        cusip = self.get_cusip_by_ticker(ticker)
        
        # 搜尋包含該 CUSIP 的所有 13F-HR 申報
        search_query = f"{cusip} form:13F-HR"
        logger.info(f"Syncing 13F holdings for {ticker} ({cusip}) via query: {search_query}")
        
        results = search_filings(search_query)
        if results.total == 0:
            return {"status": "success", "message": f"No filings found for CUSIP {cusip}", "count": 0}

        filings = results.to_filings()
        
        # 依日期降序排列
        filings = sorted(filings, key=lambda x: x.filing_date, reverse=True)
        
        # 過濾特定季度數量以防處理過多歷史資料
        today = date.today()
        days_limit = quarters_limit * 100
        cutoff_date = today - timedelta(days=days_limit)
        
        target_filings = [f for f in filings if self._parse_date(f.filing_date) >= cutoff_date]
        
        if not target_filings:
            # Fallback: 若因時間過濾沒半個，強迫留最近 50 個
            target_filings = filings[:50]

        logger.info(f"Filtered to {len(target_filings)} filings to process for {ticker}")

        synced_count = 0
        holdings_to_save = []
        
        for f in target_filings:
            # 1. 確保 SecFilingIndex 存在
            period_dt = None
            try:
                # 解析 period_of_report
                holdings_obj = f.obj()
                if hasattr(holdings_obj, 'report_period'):
                    period_dt = self._parse_date(holdings_obj.report_period)
            except Exception as e:
                logger.warning(f"Could not parse report period for {f.accession_no}: {e}")
                
            filing_index, created = SecFilingIndex.objects.get_or_create(
                accession_no=f.accession_no,
                defaults={
                    'cik': f.cik,
                    'ticker': ticker,
                    'company_name': f.company,
                    'form_type': f.form,
                    'filing_date': self._parse_date(f.filing_date),
                    'period_of_report': period_dt,
                    'primary_doc_url': f"https://www.sec.gov/Archives/edgar/data/{f.cik}/{f.accession_no.replace('-', '')}/{f.accession_no}.txt"
                }
            )

            # 2. 抓取持股明細
            try:
                holdings_obj = f.obj()
                df = holdings_obj.holdings
                
                # 篩選該 CUSIP 的持股
                matched_rows = df[df['Cusip'].str.strip() == cusip.strip()]
                if matched_rows.empty:
                    continue
                
                # 同一個 CUSIP 可能有 PUT/CALL 或 COM，我們分別儲存
                for _, row in matched_rows.iterrows():
                    put_call_val = str(row.get('PutCall', '')).strip()
                    if pd.isna(row.get('PutCall')) or put_call_val == 'nan' or put_call_val == 'None':
                        put_call_val = ''
                    
                    shares_val = int(row.get('SharesPrnAmount', 0))
                    value_val = int(row.get('Value', 0))
                    
                    holdings_to_save.append({
                        'filing': filing_index,
                        'cik': str(f.cik),
                        'ticker': ticker,
                        'cusip': cusip,
                        'security_name': str(row.get('Issuer', '')),
                        'shares': shares_val,
                        'value_usd': value_val,
                        'put_call': put_call_val,
                        'investment_discretion': str(row.get('InvestmentDiscretion', 'SOLE')),
                        'period_of_report': period_dt or filing_index.filing_date,
                    })
                    synced_count += 1
            except Exception as parse_err:
                logger.error(f"Error parsing holdings for filing {f.accession_no}: {parse_err}")

        # 3. 批量 Upsert 寫入資料庫
        # 利用 Django bulk_create 搭配 ignore_conflicts/update_fields
        if holdings_to_save:
            holdings_objs = []
            for h in holdings_to_save:
                # 建立 model 實例
                obj = Sec13fHoldings(
                    filing=h['filing'],
                    cik=h['cik'],
                    ticker=h['ticker'],
                    cusip=h['cusip'],
                    security_name=h['security_name'],
                    shares=h['shares'],
                    value_usd=h['value_usd'],
                    put_call=h['put_call'],
                    investment_discretion=h['investment_discretion'],
                    period_of_report=h['period_of_report']
                )
                holdings_objs.append(obj)
                
            # 批量寫入（符合 rules 中大於 100 批量且使用 parameterized sql 設計）
            # ignore_conflicts=True 會使用 INSERT IGNORE
            Sec13fHoldings.objects.bulk_create(holdings_objs, batch_size=150, ignore_conflicts=True)
            
            # 4. 計算季度持股變動
            self.calculate_quarterly_changes(ticker)

        return {"status": "success", "message": f"Sync completed for {ticker}", "count": synced_count}

    def calculate_quarterly_changes(self, ticker: str):
        """
        計算特定 Ticker 所有機構持股的季度增減變動 (NEW, INCREASED, DECREASED, SOLD_ALL, UNCHANGED)
        """
        # 為了能在 Python 中方便處理，我們可以按 CIK & put_call 分群，並依 period_of_report 排序
        # 由於 Django 處理分群計算較為繁複，我們可以使用 Raw SQL 或是將資料拉出來用 pandas 計算。
        # 考量到 financial analysis 效能，使用 pandas vectorization 優於 python 迴圈 (符合 Rule)。
        
        # 載入所有此 ticker 的 holdings 資料
        df_cols = ['id', 'cik', 'shares', 'period_of_report', 'put_call']
        holdings_qs = Sec13fHoldings.objects.filter(ticker=ticker).values(*df_cols)
        if not holdings_qs.exists():
            return
            
        df = pd.DataFrame(list(holdings_qs))
        df['period_of_report'] = pd.to_datetime(df['period_of_report'])
        
        # 排序：CIK, put_call, period_of_report 遞增
        df = df.sort_values(by=['cik', 'put_call', 'period_of_report']).copy()
        
        # 計算上一期的持股股數 (使用 shift 向量化運算)
        df['prev_shares'] = df.groupby(['cik', 'put_call'])['shares'].shift(1)
        
        # 計算變動股數與百分比
        df['shares_change'] = df['shares'] - df['prev_shares']
        
        # 計算變動百分比
        # 避免除以 0
        df['change_pct'] = 0.0
        mask_has_prev = (df['prev_shares'].notna()) & (df['prev_shares'] > 0)
        df.loc[mask_has_prev, 'change_pct'] = (df.loc[mask_has_prev, 'shares_change'] / df.loc[mask_has_prev, 'prev_shares']) * 100
        
        # 判斷 action_type
        # NEW, INCREASED, DECREASED, UNCHANGED
        df['action_type'] = 'UNCHANGED'
        df.loc[df['prev_shares'].isna(), 'action_type'] = 'NEW'
        df.loc[(df['shares_change'] > 0) & (df['prev_shares'].notna()), 'action_type'] = 'INCREASED'
        df.loc[(df['shares_change'] < 0) & (df['prev_shares'].notna()), 'action_type'] = 'DECREASED'
        df.loc[(df['shares_change'] == 0) & (df['prev_shares'].notna()), 'action_type'] = 'UNCHANGED'
        
        # 處理 SOLD_ALL 邏輯：
        # 如果上一期有持股，但這一期該機構的 13F 中已經沒有該 CUSIP。
        # 為了找出清倉，我們需要找出每對 (cik, put_call) 在報告期序列中的空缺。
        # 我們取得所有現存的報告期列表
        all_periods = sorted(df['period_of_report'].unique())
        
        sold_records = []
        
        for (cik, put_call), group in df.groupby(['cik', 'put_call']):
            group_periods = set(group['period_of_report'])
            # 依時間排序
            sorted_group = group.sort_values(by='period_of_report')
            for i in range(len(all_periods) - 1):
                curr_p = all_periods[i]
                next_p = all_periods[i+1]
                
                # 如果 curr_p 有持股，但 next_p 沒有 (即 CIK 在 next_p 的申報中未列出該持股)
                if curr_p in group_periods and next_p not in group_periods:
                    # 取得 curr_p 的持股資料
                    curr_row = sorted_group[sorted_group['period_of_report'] == curr_p].iloc[0]
                    # 只有當前一期持股大於 0 時，才需要新增清倉紀錄
                    if curr_row['shares'] > 0:
                        # 尋找該 next_p 對應該 CIK 的任何一個 FilingIndex
                        filing = SecFilingIndex.objects.filter(
                            cik=cik, 
                            period_of_report=next_p.date(),
                            form_type='13F-HR'
                        ).first()
                        
                        if filing:
                            sold_records.append(Sec13fHoldings(
                                filing=filing,
                                cik=cik,
                                ticker=ticker,
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
                            
        # 將清倉記錄批次寫入資料庫
        if sold_records:
            Sec13fHoldings.objects.bulk_create(sold_records, ignore_conflicts=True)
            
        # 將計算出來的 values 回填更新原本的資料列 (使用批次更新以提升效能)
        # 為了高效率更新，我們使用 Raw SQL / Cursor 來執行批量更新，或是透過 Django bulk_update。
        # 這裡我們用 Django ORM bulk_update。
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

    def get_top_holders_from_db(self, ticker: str, period: date = None, limit: int = 50) -> list:
        """
        從資料庫獲取特定 Ticker 在指定報告季度的前 N 大機構持有者。
        """
        ticker = ticker.upper()
        
        # 若未指定日期，獲取最新的 period_of_report
        if not period:
            latest_holding = Sec13fHoldings.objects.filter(ticker=ticker).order_by('-period_of_report').first()
            if not latest_holding:
                return []
            period = latest_holding.period_of_report
            
        holdings_qs = Sec13fHoldings.objects.filter(
            ticker=ticker,
            period_of_report=period
        ).select_related('filing').order_by('-shares')[:limit]
        
        results = []
        for h in holdings_qs:
            results.append({
                'company_name': h.filing.company_name,
                'cik': h.cik,
                'shares': h.shares,
                'value_usd': h.value_usd * 1000,  # 轉回實際美元
                'put_call': h.put_call,
                'change_pct': h.change_pct,
                'action_type': h.action_type,
                'period_of_report': h.period_of_report
            })
        return results
