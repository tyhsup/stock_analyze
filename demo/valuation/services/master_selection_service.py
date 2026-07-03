import logging
import os
import pandas as pd
import numpy as np
from decimal import Decimal
from sqlalchemy import text
from stock_Django.mySQL_OP import OP_Fun

logger = logging.getLogger(__name__)

class MasterSelectionService:
    def __init__(self):
        self.op = OP_Fun()

    def run_buffett_selection(self, market: str):
        """
        執行巴菲特選股策略
        market: 'tw' 或 'us'
        回傳選出的前 5 檔股票列表 (dict list)
        """
        market = market.lower()
        table_name = f"financial_raw_{market}"
        
        # 1. 取得所有股票最近 8 季的財務數據以計算 TTM 與去年同期 TTM
        # 為了效能，我們用一個 SQL 查詢撈出所有資料，再用 pandas 做記憶體計算
        query = f"""
            SELECT symbol, year, quarter, statement_type, item_name, amount 
            FROM {table_name}
            WHERE year >= (YEAR(CURDATE()) - 3)
        """
        
        with self.op.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        if df.empty:
            logger.warning(f"No financial raw data found for market: {market}")
            return []

        # 統一單位：台股資料在資料庫中是千元，我們在 memory 中乘以 1000 統一為元，美股為元
        if market == 'tw':
            df['amount'] = df['amount'] * 1000

        # 將 year, quarter 轉化成排序用的數值
        df['period_val'] = df['year'] * 10 + df['quarter']
        
        # 2. 定義項目的 mapping (跟 valuation/services/financial_data.py 一致)
        mapping = {
            'tw': {
                'revenue': ["Total Revenue", "Operating Revenue", "營業收入合計", "營業收入", "營業收入淨額"],
                'gross_profit': ["Gross Profit", "營業毛利", "營業毛利（毛損）"],
                'net_income': ["Net Income", "本期淨利", "本期淨利（淨損）", "歸屬於母公司業主之本期淨利（損）"],
                'equity': ["Common Stock Equity", "歸屬於母公司業主之權益總計", "權益總計", "權益總額"],
                'assets': ["Total Assets", "資產總計", "資產總額"],
                'liabilities': ["Total Liabilities", "負債總計", "負債總額"]
            },
            'us': {
                'revenue': ["Revenues", "TotalRevenue", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingCostReportedOnNetBasis"],
                'gross_profit': ["GrossProfit", "GrossProfitLoss"],
                'net_income': ["NetIncomeLoss", "ProfitLoss"],
                'equity': ["StockholdersEquity"],
                'assets': ["Assets"],
                'liabilities': ["Liabilities"]
            }
        }.get(market, {})

        # 對項目進行映射歸類
        def map_item(name):
            for category, keys in mapping.items():
                if name in keys:
                    return category
            return None

        df['category'] = df['item_name'].apply(map_item)
        df_filtered = df.dropna(subset=['category']).copy()

        # 3. 對每隻股票分開計算指標
        # 找出每隻股票的最大 period_val，代表最新財報季度
        latest_periods = df_filtered.groupby('symbol')['period_val'].max().to_dict()
        
        results = []
        
        # 針對有資料的 symbol 進行計算
        grouped = df_filtered.groupby('symbol')
        for symbol, group in grouped:
            latest_p = latest_periods.get(symbol)
            if not latest_p:
                continue
                
            # 取得最新一季的年與季
            # 為了計算 TTM，我們需要定位最新一季往回推共 4 季的 period_val 清單
            # 以及去年同期的 4 季清單 (往回推第 5 到第 8 季)
            # 先將該 symbol 的所有獨特 period_val 排序
            unique_periods = sorted(group['period_val'].unique())
            if len(unique_periods) < 4:
                # 財報季數不足，無法計算 TTM
                continue
                
            # 定位最新一季在 unique_periods 的 index
            try:
                latest_idx = unique_periods.index(latest_p)
            except ValueError:
                continue
                
            # TTM 區間 (最新 4 季)
            ttm_periods = unique_periods[max(0, latest_idx - 3): latest_idx + 1]
            if len(ttm_periods) < 4:
                continue
                
            # 去年同期 TTM 區間 (往回推第 5 到第 8 季)
            pre_ttm_periods = unique_periods[max(0, latest_idx - 7): max(0, latest_idx - 3)]
            
            # 篩選出對應期間的數據
            group_ttm = group[group['period_val'].isin(ttm_periods)]
            group_pre_ttm = group[group['period_val'].isin(pre_ttm_periods)] if pre_ttm_periods else pd.DataFrame()
            group_latest = group[group['period_val'] == latest_p]

            # 輔助函數：取得 TTM 加總
            def get_ttm_sum(sub_group, category):
                # 預防同一個季度同一個 category 內有多個項目重複加總 (例如 Revenues 和 TotalRevenue)
                # 我們在每個 symbol, period_val, category 內只取 amount 的最大值或第一筆
                cat_df = sub_group[sub_group['category'] == category]
                if cat_df.empty:
                    return 0.0
                # 依據 period_val 進行 pivot/groupby 取第一筆，再加總
                return cat_df.groupby('period_val')['amount'].first().sum()

            # 輔助函數：取得最新單季數值
            def get_latest_val(sub_group, category):
                cat_df = sub_group[sub_group['category'] == category]
                if cat_df.empty:
                    return 0.0
                return cat_df['amount'].iloc[0]

            # 計算 TTM 財務數據
            revenue_ttm = get_ttm_sum(group_ttm, 'revenue')
            gross_profit_ttm = get_ttm_sum(group_ttm, 'gross_profit')
            net_income_ttm = get_ttm_sum(group_ttm, 'net_income')

            # 計算去年同期 TTM 淨利
            net_income_pre_ttm = get_ttm_sum(group_pre_ttm, 'net_income') if not group_pre_ttm.empty else 0.0

            # 計算最新資產負債表數據
            equity_latest = get_latest_val(group_latest, 'equity')
            assets_latest = get_latest_val(group_latest, 'assets')
            liabilities_latest = get_latest_val(group_latest, 'liabilities')

            # 數據合理性檢查 (避免分母為 0)
            if revenue_ttm <= 0 or equity_latest <= 0 or assets_latest <= 0:
                continue

            # 4. 計算巴菲特指標
            # ROE = Net Income TTM / Equity
            roe = net_income_ttm / equity_latest
            
            # Gross Margin = Gross Profit TTM / Revenue TTM
            # 若 Gross Profit 為 0，有些財報可能沒爬到毛利，可用 (Revenue - Cost) 填補，這裡若為 0 則設為 0
            gross_margin = gross_profit_ttm / revenue_ttm if gross_profit_ttm > 0 else 0.0
            
            # Debt Ratio = Liabilities / Assets
            # 若 liabilities 為 0，可能是 1 - (Equity / Assets)
            if liabilities_latest <= 0 and assets_latest > equity_latest:
                liabilities_latest = assets_latest - equity_latest
            debt_ratio = liabilities_latest / assets_latest
            
            # Net Income Growth = (Net Income TTM - Net Income Pre-TTM) / Net Income Pre-TTM
            if net_income_pre_ttm != 0:
                net_income_growth = (net_income_ttm - net_income_pre_ttm) / abs(net_income_pre_ttm)
            else:
                net_income_growth = 0.05 # 預設給一個微幅正數

            # 限制極端值以防干擾評分
            roe = max(min(roe, 1.5), -0.5)
            gross_margin = max(min(gross_margin, 1.0), 0.0)
            debt_ratio = max(min(debt_ratio, 1.0), 0.0)
            net_income_growth = max(min(net_income_growth, 3.0), -1.0)

            # 5. 計算巴菲特評分 (總分 100 分)
            # ROE 權重 40% (目標 >= 15%)
            score_roe = min(roe / 0.15, 1.0) * 100 if roe > 0 else 0
            
            # 毛利率權重 30% (目標 >= 30%)
            score_gm = min(gross_margin / 0.30, 1.0) * 100
            
            # 負債比率權重 20% (目標 <= 50%，越低越高分)
            score_debt = (1.0 - debt_ratio) * 100
            if debt_ratio > 0.5:
                # 負債比大於 50% 扣分
                score_debt = max(score_debt - (debt_ratio - 0.5) * 100, 0)
                
            # 成長率權重 10% (目標 > 0)
            score_growth = 0
            if net_income_growth > 0:
                score_growth = min(net_income_growth / 0.10, 1.0) * 100
            else:
                score_growth = 0

            total_score = (score_roe * 0.4) + (score_gm * 0.3) + (score_debt * 0.2) + (score_growth * 0.1)

            # 6. 判斷是否符合巴菲特的四大嚴格標準
            # 就算沒有完全符合，我們也保留，後面可以用 score 進行綜合排名
            results.append({
                'symbol': symbol,
                'roe': roe,
                'gross_margin': gross_margin,
                'debt_ratio': debt_ratio,
                'net_income_growth': net_income_growth,
                'score': total_score
            })

        if not results:
            return []

        # 轉成 DataFrame 進行排序與對接股票名稱
        res_df = pd.DataFrame(results)
        
        # 撈取股票名稱與最新股價
        stocks_table = f"stocks_{market}"
        price_table = f"stock_cost" if market == 'tw' else "stock_cost_us"
        
        # 取得最新股價
        price_query = f"""
            SELECT p.number as symbol, p.Close as close_price
            FROM {price_table} p
            INNER JOIN (
                SELECT number, MAX(Date) as max_date
                FROM {price_table}
                GROUP BY number
            ) latest ON p.number = latest.number AND p.Date = latest.max_date
        """
        
        # 取得股票名稱
        name_query = f"SELECT symbol, name FROM {stocks_table}"

        with self.op.engine.connect() as conn:
            df_names = pd.read_sql(text(name_query), conn)
            df_prices = pd.read_sql(text(price_query), conn)

        res_df = res_df.merge(df_names, on='symbol', how='left')
        res_df = res_df.merge(df_prices, on='symbol', how='left')
        
        # 處理缺失名稱與價格
        res_df['name'] = res_df['name'].fillna(res_df['symbol'])
        res_df['close_price'] = res_df['close_price'].fillna(0.0)

        # 排序：以 total_score 降序排序，取前 5 檔
        res_df = res_df.sort_values(by='score', ascending=False)
        top_5 = res_df.head(5)

        # 寫入資料庫 master_selection table
        self._save_to_db(top_5, market, 'buffett')

        # 格式化輸出
        output = []
        for i, row in enumerate(top_5.to_dict(orient='records'), 1):
            output.append({
                'rank': i,
                'symbol': row['symbol'],
                'name': row['name'],
                'close_price': float(row['close_price']),
                'roe': float(row['roe']),
                'gross_margin': float(row['gross_margin']),
                'debt_ratio': float(row['debt_ratio']),
                'net_income_growth': float(row['net_income_growth']),
                'score': float(row['score'])
            })
            
        return output

    def _save_to_db(self, df_top, market: str, master_name: str):
        """將篩選出來的前 5 檔股票儲存至資料庫的 master_selection Table"""
        # 刪除既有的該市場與該大師的紀錄
        delete_sql = """
            DELETE FROM master_selection 
            WHERE market = :market AND master_name = :master_name
        """
        
        import datetime
        insert_sql = """
            INSERT INTO master_selection 
            (market, symbol, name, master_name, `rank`, close_price, roe, gross_margin, debt_ratio, net_income_growth, score, updated_at)
            VALUES (:market, :symbol, :name, :master_name, :rank, :close_price, :roe, :gross_margin, :debt_ratio, :net_income_growth, :score, :updated_at)
        """

        with self.op.engine.begin() as conn:
            # 刪除舊資料
            conn.execute(text(delete_sql), {'market': market, 'master_name': master_name})
            
            # 插入新資料
            for idx, row in enumerate(df_top.to_dict(orient='records'), 1):
                conn.execute(text(insert_sql), {
                    'market': market,
                    'symbol': row['symbol'],
                    'name': row['name'],
                    'master_name': master_name,
                    'rank': idx,
                    'close_price': Decimal(str(row['close_price'])) if not pd.isna(row['close_price']) else Decimal('0'),
                    'roe': Decimal(str(row['roe'])) if not pd.isna(row['roe']) else Decimal('0'),
                    'gross_margin': Decimal(str(row['gross_margin'])) if not pd.isna(row['gross_margin']) else Decimal('0'),
                    'debt_ratio': Decimal(str(row['debt_ratio'])) if not pd.isna(row['debt_ratio']) else Decimal('0'),
                    'net_income_growth': Decimal(str(row['net_income_growth'])) if not pd.isna(row['net_income_growth']) else Decimal('0'),
                    'score': Decimal(str(row['score'])) if not pd.isna(row['score']) else Decimal('0'),
                    'updated_at': datetime.datetime.now()
                })
        
        logger.info(f"Successfully saved Buffett selection results for {market} market.")
