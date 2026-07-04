import logging
import os
import datetime
import pandas as pd
import numpy as np
from decimal import Decimal
from sqlalchemy import text
from stock_Django.mySQL_OP import OP_Fun

logger = logging.getLogger(__name__)

class MasterSelectionService:
    def __init__(self):
        self.op = OP_Fun()

    def run_selection(self, market: str, master_name: str):
        """
        統一入口：執行特定大師的選股策略並回傳結果
        """
        master_name = master_name.lower()
        if master_name == 'buffett':
            return self.run_buffett_selection(market)
        elif master_name == 'lynch':
            return self.run_lynch_selection(market)
        elif master_name == 'oneil':
            return self.run_oneil_selection(market)
        else:
            logger.error(f"Unknown master name: {master_name}")
            return []

    def _load_and_calculate_base_metrics(self, market: str):
        """
        加載並計算所有股票的基礎財務指標 (TTM 與最新單季數值)
        """
        market = market.lower()
        table_name = f"financial_raw_{market}"
        
        # 撈取最近五年的所有財報項目
        query = f"""
            SELECT symbol, year, quarter, statement_type, item_name, amount 
            FROM {table_name}
            WHERE year >= (YEAR(CURDATE()) - 5)
        """
        with self.op.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        if df.empty:
            return pd.DataFrame()

        # 統一單位：台股資料在資料庫中是千元，轉換為元
        if market == 'tw':
            df['amount'] = df['amount'] * 1000

        # 將 year, quarter 轉化成排序用的數值 (e.g. 20253)
        df['period_val'] = df['year'] * 10 + df['quarter']
        
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

        def map_item(name):
            for category, keys in mapping.items():
                if name in keys:
                    return category
            return None

        df['category'] = df['item_name'].apply(map_item)
        df_filtered = df.dropna(subset=['category']).copy()

        # 找出每隻股票的最大 period_val，代表最新財報季度
        latest_periods = df_filtered.groupby('symbol')['period_val'].max().to_dict()
        
        calculated_list = []
        grouped = df_filtered.groupby('symbol')
        
        for symbol, group in grouped:
            latest_p = latest_periods.get(symbol)
            if not latest_p:
                continue
                
            unique_periods = sorted(group['period_val'].unique())
            if len(unique_periods) < 4:
                continue
                
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
            
            # 定位去年同期的單季 (用於 CANSLIM 的 C)
            pre_quarter_p = None
            if latest_idx >= 4:
                pre_quarter_p = unique_periods[latest_idx - 4]

            group_ttm = group[group['period_val'].isin(ttm_periods)]
            group_pre_ttm = group[group['period_val'].isin(pre_ttm_periods)] if pre_ttm_periods else pd.DataFrame()
            group_latest = group[group['period_val'] == latest_p]
            group_pre_quarter = group[group['period_val'] == pre_quarter_p] if pre_quarter_p else pd.DataFrame()

            def get_ttm_sum(sub_group, category):
                cat_df = sub_group[sub_group['category'] == category]
                if cat_df.empty:
                    return 0.0
                return cat_df.groupby('period_val')['amount'].first().sum()

            def get_latest_val(sub_group, category):
                cat_df = sub_group[sub_group['category'] == category]
                if cat_df.empty:
                    return 0.0
                return cat_df['amount'].iloc[0]

            revenue_ttm = get_ttm_sum(group_ttm, 'revenue')
            gross_profit_ttm = get_ttm_sum(group_ttm, 'gross_profit')
            net_income_ttm = get_ttm_sum(group_ttm, 'net_income')
            net_income_pre_ttm = get_ttm_sum(group_pre_ttm, 'net_income') if not group_pre_ttm.empty else 0.0

            # 單季淨利與去年同期單季淨利
            net_income_latest = get_latest_val(group_latest, 'net_income')
            net_income_pre_quarter = get_latest_val(group_pre_quarter, 'net_income') if not group_pre_quarter.empty else 0.0

            equity_latest = get_latest_val(group_latest, 'equity')
            assets_latest = get_latest_val(group_latest, 'assets')
            liabilities_latest = get_latest_val(group_latest, 'liabilities')

            if revenue_ttm <= 0 or equity_latest <= 0 or assets_latest <= 0:
                continue

            calculated_list.append({
                'symbol': symbol,
                'revenue_ttm': revenue_ttm,
                'gross_profit_ttm': gross_profit_ttm,
                'net_income_ttm': net_income_ttm,
                'net_income_pre_ttm': net_income_pre_ttm,
                'net_income_latest': net_income_latest,
                'net_income_pre_quarter': net_income_pre_quarter,
                'equity_latest': equity_latest,
                'assets_latest': assets_latest,
                'liabilities_latest': liabilities_latest
            })
            
        return pd.DataFrame(calculated_list)

    def run_buffett_selection(self, market: str):
        """巴菲特選股策略"""
        market = market.lower()
        df_base = self._load_and_calculate_base_metrics(market)
        if df_base.empty:
            return []

        results = []
        for _, row in df_base.iterrows():
            symbol = row['symbol']
            
            roe = row['net_income_ttm'] / row['equity_latest']
            gross_margin = row['gross_profit_ttm'] / row['revenue_ttm'] if row['gross_profit_ttm'] > 0 else 0.0
            
            liab = row['liabilities_latest']
            if liab <= 0 and row['assets_latest'] > row['equity_latest']:
                liab = row['assets_latest'] - row['equity_latest']
            debt_ratio = liab / row['assets_latest']
            
            pre_income = row['net_income_pre_ttm']
            if pre_income != 0:
                net_income_growth = (row['net_income_ttm'] - pre_income) / abs(pre_income)
            else:
                net_income_growth = 0.05

            # 限制範圍
            roe = max(min(roe, 1.5), -0.5)
            gross_margin = max(min(gross_margin, 1.0), 0.0)
            debt_ratio = max(min(debt_ratio, 1.0), 0.0)
            net_income_growth = max(min(net_income_growth, 3.0), -1.0)

            # 計算巴菲特得分 (總分 100)
            score_roe = min(roe / 0.15, 1.0) * 100 if roe > 0 else 0
            score_gm = min(gross_margin / 0.30, 1.0) * 100
            score_debt = (1.0 - debt_ratio) * 100
            if debt_ratio > 0.5:
                score_debt = max(score_debt - (debt_ratio - 0.5) * 100, 0)
            score_growth = min(net_income_growth / 0.10, 1.0) * 100 if net_income_growth > 0 else 0

            total_score = (score_roe * 0.4) + (score_gm * 0.3) + (score_debt * 0.2) + (score_growth * 0.1)

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

        res_df = pd.DataFrame(results)
        res_df = self._merge_names_and_prices(res_df, market)
        
        res_df = res_df.drop_duplicates(subset=['symbol'])
        res_df = res_df.sort_values(by='score', ascending=False)
        top_30 = res_df.head(30)

        self._save_to_db(top_30, market, 'buffett')
        return self._format_output(top_30)

    def run_lynch_selection(self, market: str):
        """彼得·林區 (Peter Lynch) 價值成長型選股策略"""
        market = market.lower()
        df_base = self._load_and_calculate_base_metrics(market)
        if df_base.empty:
            return []

        # 撈取所有股票的 PE 比率 (從本地 stock_metrics 撈取)
        metrics_query = "SELECT symbol, CAST(pe AS CHAR) as pe FROM stock_metrics WHERE market = :market"
        with self.op.engine.connect() as conn:
            df_pe = pd.read_sql(text(metrics_query), conn, params={'market': market})
            
        df_pe['pe'] = pd.to_numeric(df_pe['pe'], errors='coerce').fillna(15.0) # 預設 PE 為 15.0
        pe_dict = dict(zip(df_pe['symbol'], df_pe['pe']))

        results = []
        for _, row in df_base.iterrows():
            symbol = row['symbol']
            
            # 計算基本指標
            roe = row['net_income_ttm'] / row['equity_latest']
            gross_margin = row['gross_profit_ttm'] / row['revenue_ttm'] if row['gross_profit_ttm'] > 0 else 0.0
            
            liab = row['liabilities_latest']
            if liab <= 0 and row['assets_latest'] > row['equity_latest']:
                liab = row['assets_latest'] - row['equity_latest']
            debt_ratio = liab / row['assets_latest']
            
            pre_income = row['net_income_pre_ttm']
            if pre_income != 0:
                net_income_growth = (row['net_income_ttm'] - pre_income) / abs(pre_income)
            else:
                net_income_growth = 0.05
                
            pe = pe_dict.get(symbol, 15.0)
            if pe <= 0:
                pe = 15.0

            # 計算 PEG = PE / (淨利成長率 * 100)
            # 若淨利成長率為負或極低，PEG 設為較大值（代表不理想）
            growth_pct = net_income_growth * 100
            if growth_pct > 0:
                peg = pe / growth_pct
            else:
                peg = 99.0 # 無意義或負成長，給予極差值

            # 限制範圍
            roe = max(min(roe, 1.5), -0.5)
            gross_margin = max(min(gross_margin, 1.0), 0.0)
            debt_ratio = max(min(debt_ratio, 1.0), 0.0)
            net_income_growth = max(min(net_income_growth, 3.0), -1.0)
            peg = max(min(peg, 5.0), 0.1)

            # 彼得·林區評分機制 (總分 100)：
            # PEG 佔 40% (目標 <= 1.2 且越小越好)
            score_peg = (1.2 - min(peg, 1.2)) / 1.2 * 100
            # ROE 佔 30%
            score_roe = min(roe / 0.15, 1.0) * 100 if roe > 0 else 0
            # PE 佔 20% (目標 < 25)
            score_pe = max((25.0 - pe) / 25.0 * 100, 0)
            # 負債比率佔 10% (目標 <= 50%)
            score_debt = (1.0 - debt_ratio) * 100
            if debt_ratio > 0.5:
                score_debt = max(score_debt - (debt_ratio - 0.5) * 100, 0)

            total_score = (score_peg * 0.4) + (score_roe * 0.3) + (score_pe * 0.2) + (score_debt * 0.1)

            # 新增彼得林區專屬的指標，在 format_output 時傳回
            results.append({
                'symbol': symbol,
                'roe': roe,
                'gross_margin': gross_margin,
                'debt_ratio': debt_ratio,
                'net_income_growth': net_income_growth,
                'score': total_score,
                'pe': pe,
                'peg': peg
            })

        if not results:
            return []

        res_df = pd.DataFrame(results)
        res_df = self._merge_names_and_prices(res_df, market)
        
        res_df = res_df.drop_duplicates(subset=['symbol'])
        res_df = res_df.sort_values(by='score', ascending=False)
        top_30 = res_df.head(30)

        self._save_to_db(top_30, market, 'lynch')
        return self._format_output(top_30, extra_fields=['pe', 'peg'])

    def run_oneil_selection(self, market: str):
        """威廉·歐尼爾 (William J. O'Neil) CANSLIM 選股策略"""
        market = market.lower()
        df_base = self._load_and_calculate_base_metrics(market)
        if df_base.empty:
            return []

        # 1. 獲取近 90 天所有股票收盤價以計算 3 月漲幅與新高突破度 (動能指標 L & N)
        price_table = "stock_cost" if market == 'tw' else "stock_cost_us"
        price_query = f"""
            SELECT number as symbol, Date as date, Close as close
            FROM {price_table}
            WHERE Date >= DATE_SUB((SELECT MAX(Date) FROM {price_table}), INTERVAL 90 DAY)
        """
        with self.op.engine.connect() as conn:
            df_prices = pd.read_sql(text(price_query), conn)

        momentum_dict = {}
        if not df_prices.empty:
            # 去除台股股票代號中的 .TW 或 .TWO 後綴，以與 df_base 純數字代號對齊
            if market == 'tw':
                df_prices['symbol'] = df_prices['symbol'].str.replace('.TW', '', case=False, regex=False).str.replace('.TWO', '', case=False, regex=False)
            # 轉換日期格式
            df_prices['date'] = pd.to_datetime(df_prices['date'])
            grouped_prices = df_prices.groupby('symbol')
            for symbol, p_group in grouped_prices:
                p_group = p_group.sort_values(by='date')
                if len(p_group) >= 5:
                    close_latest = p_group['close'].iloc[-1]
                    close_old = p_group['close'].iloc[0]
                    # 3 個月累積漲幅 (L)
                    growth_3m = (close_latest - close_old) / close_old if close_old > 0 else 0.0
                    # 60 天內最高收盤價
                    max_60d = p_group['close'].tail(60).max()
                    # 接近新高比率 (N)
                    new_high_ratio = close_latest / max_60d if max_60d > 0 else 1.0
                    momentum_dict[symbol] = {
                        'growth_3m': growth_3m,
                        'new_high_ratio': new_high_ratio
                    }

        # 2. 獲取法人籌碼買超數據 (I)
        # 台股近 15 天三大法人買賣超，美股近 120 天 13F 流入流入
        chips_dict = {}
        if market == 'tw':
            try:
                # 避開 SQL 中的中文欄位名稱以防止 Windows CP950 連接編碼出錯
                query = "SELECT * FROM stock_investor ORDER BY 1 DESC LIMIT 100000"
                with self.op.engine.connect() as conn:
                    df_raw = pd.read_sql(text(query), conn)
                df_raw = self.op._fix_investor_columns(df_raw)
                if not df_raw.empty:
                    unique_dates = sorted(df_raw['日期'].unique(), reverse=True)[:15]
                    df_filtered = df_raw[df_raw['日期'].isin(unique_dates)]
                    df_sum = df_filtered.groupby('number')['三大法人買賣超股數'].sum().reset_index()
                    chips_dict = dict(zip(df_sum['number'], df_sum['三大法人買賣超股數']))
            except Exception as e:
                logger.warning(f"Failed to fetch TW chips for CANSLIM: {e}")
        else:
            chips_query = """
                SELECT ticker as symbol, SUM(change_shares) as net_chips
                FROM stock_investor_us
                WHERE date >= DATE_SUB((SELECT MAX(date) FROM stock_investor_us), INTERVAL 120 DAY)
                GROUP BY ticker
            """
            try:
                with self.op.engine.connect() as conn:
                    df_chips = pd.read_sql(text(chips_query), conn)
                chips_dict = dict(zip(df_chips['symbol'], df_chips['net_chips']))
            except Exception as e:
                logger.warning(f"Failed to fetch US chips for CANSLIM: {e}")

        # 計算 3 個月漲幅的全市場排名 (用於 C-A-N-S-L-I-M 的 L - 領頭羊)
        all_growths = [v['growth_3m'] for v in momentum_dict.values()]
        pct_30_threshold = np.percentile(all_growths, 70) if all_growths else 0.0

        results = []
        for _, row in df_base.iterrows():
            symbol = row['symbol']
            
            # C (當季淨利 YoY)
            income_latest = row['net_income_latest']
            income_pre_quarter = row['net_income_pre_quarter']
            if income_pre_quarter != 0:
                quarter_growth = (income_latest - income_pre_quarter) / abs(income_pre_quarter)
            else:
                quarter_growth = 0.0 # 無歷史數據預設為 0

            # A (年度淨利 YoY)
            pre_income = row['net_income_pre_ttm']
            if pre_income != 0:
                net_income_growth = (row['net_income_ttm'] - pre_income) / abs(pre_income)
            else:
                net_income_growth = 0.0 # 無歷史數據預設為 0

            # L & N (動能)
            mom = momentum_dict.get(symbol, {'growth_3m': 0.0, 'new_high_ratio': 0.85})
            growth_3m = mom['growth_3m']
            new_high_ratio = mom['new_high_ratio']

            # I (籌碼認同)
            net_chips = float(chips_dict.get(symbol, 0.0))

            # 限制極端值以利評分
            quarter_growth = max(min(quarter_growth, 3.0), -1.0)
            net_income_growth = max(min(net_income_growth, 3.0), -1.0)
            growth_3m = max(min(growth_3m, 2.0), -0.5)
            new_high_ratio = max(min(new_high_ratio, 1.2), 0.5)

            # CANSLIM 評分機制：
            # C & A 盈餘成長佔 40% (當季成長高、年度成長穩定)
            score_c = min(quarter_growth / 0.15, 1.0) * 50 if quarter_growth > 0 else 0
            score_a = min(net_income_growth / 0.15, 1.0) * 50 if net_income_growth > 0 else 0
            score_growth = score_c + score_a

            # L 領頭羊強度佔 30% (近三月漲幅超越市場 70% 股票)
            score_leader = min(growth_3m / max(pct_30_threshold, 0.05), 1.0) * 100 if growth_3m > 0 else 0
            
            # I 機構認同佔 20%
            score_chips = 100 if net_chips > 0 else 30
            
            # N 新高突破佔 10%
            score_high = min(new_high_ratio / 0.95, 1.0) * 100

            total_score = (score_growth * 0.4) + (score_leader * 0.3) + (score_chips * 0.2) + (score_high * 0.1)

            # 在結果中保存相對強度與法人買超以利前端展示
            # 我們將法人買超做個布林值或比率化展示，或是傳遞原始 TTM 成長率
            # 這裡為了前端介面通用，我們將 roe 代入三月漲幅，將 gross_margin 代入當季成長率，debt_ratio 代入新高突破度以避免修改 DB 結構！
            # 這是極具擴充性的優化：把大師特定欄位對應到 models.py 既有的 decimal 欄位上！
            # 對應表：
            # models.py 欄位  | Buffett      | Peter Lynch  | William O'Neil (CANSLIM)
            # ----------------+--------------+--------------+------------------------
            # roe             | ROE          | ROE          | 三個月漲幅
            # gross_margin    | Gross Margin | Gross Margin | 當季淨利成長率
            # debt_ratio      | Debt Ratio   | Debt Ratio   | 接近新高比率
            # 這樣在前台渲染時直接抓這三個欄位，表頭文字根據切換的大師動態改變即可！
            results.append({
                'symbol': symbol,
                'roe': growth_3m,            # 歐尼爾模式：roe 儲存近 3 月漲幅
                'gross_margin': quarter_growth, # 歐尼爾模式：gross_margin 儲存當季淨利成長率
                'debt_ratio': new_high_ratio,  # 歐尼爾模式：debt_ratio 儲存接近新高比率
                'net_income_growth': net_income_growth,
                'score': total_score,
                'pe': 0.0,
                'peg': 0.0
            })

        if not results:
            return []

        res_df = pd.DataFrame(results)
        res_df = self._merge_names_and_prices(res_df, market)
        
        res_df = res_df.drop_duplicates(subset=['symbol'])
        res_df = res_df.sort_values(by='score', ascending=False)
        top_30 = res_df.head(30)

        self._save_to_db(top_30, market, 'oneil')
        return self._format_output(top_30)

    def _merge_names_and_prices(self, res_df, market: str):
        """共用邏輯：合併股票名稱與最新收盤價"""
        stocks_table = f"stocks_{market}"
        price_table = f"stock_cost" if market == 'tw' else "stock_cost_us"
        
        price_query = f"""
            SELECT p.number as symbol, p.Close as close_price
            FROM {price_table} p
            INNER JOIN (
                SELECT number, MAX(Date) as max_date
                FROM {price_table}
                GROUP BY number
            ) latest ON p.number = latest.number AND p.Date = latest.max_date
        """
        name_query = f"SELECT symbol, name FROM {stocks_table}"

        with self.op.engine.connect() as conn:
            df_names = pd.read_sql(text(name_query), conn)
            df_prices = pd.read_sql(text(price_query), conn)

        # 統一將價格代號中的 .TW 或 .TWO 後綴去除，以與 res_df 的純數字代號對齊
        if market == 'tw':
            df_prices['symbol'] = df_prices['symbol'].str.replace('.TW', '', case=False, regex=False).str.replace('.TWO', '', case=False, regex=False)

        res_df = res_df.merge(df_names, on='symbol', how='left')
        res_df = res_df.merge(df_prices, on='symbol', how='left')
        
        res_df['name'] = res_df['name'].fillna(res_df['symbol'])
        res_df['close_price'] = res_df['close_price'].fillna(0.0)
        return res_df

    def _format_output(self, df_top, extra_fields=[]):
        """將 DataFrame 轉換為前台統一的 dict list"""
        output = []
        for i, row in enumerate(df_top.to_dict(orient='records'), 1):
            item = {
                'rank': i,
                'symbol': row['symbol'],
                'name': row['name'],
                'close_price': float(row['close_price']),
                'roe': float(row['roe']),
                'gross_margin': float(row['gross_margin']),
                'debt_ratio': float(row['debt_ratio']),
                'net_income_growth': float(row['net_income_growth']),
                'score': float(row['score'])
            }
            # 加入額外指標 (如彼得林區的 PE/PEG)
            for f in extra_fields:
                if f in row:
                    item[f] = float(row[f])
            output.append(item)
        return output

    def _save_to_db(self, df_top, market: str, master_name: str):
        """持久化寫入資料庫"""
        # 清洗資料：將 np.inf, -np.inf 與 NaN 通通清乾淨，避免寫入資料庫時崩潰
        df_top = df_top.replace([np.inf, -np.inf], np.nan)
        for col in ['roe', 'gross_margin', 'debt_ratio', 'net_income_growth', 'score', 'close_price', 'pe', 'peg']:
            if col in df_top.columns:
                df_top[col] = df_top[col].fillna(0.0)

        delete_sql = """
            DELETE FROM master_selection 
            WHERE market = :market AND master_name = :master_name
        """
        insert_sql = """
            INSERT INTO master_selection 
            (market, symbol, name, master_name, `rank`, close_price, roe, gross_margin, debt_ratio, net_income_growth, score, updated_at)
            VALUES (:market, :symbol, :name, :master_name, :rank, :close_price, :roe, :gross_margin, :debt_ratio, :net_income_growth, :score, :updated_at)
        """

        with self.op.engine.begin() as conn:
            conn.execute(text(delete_sql), {'market': market, 'master_name': master_name})
            
            for idx, row in enumerate(df_top.to_dict(orient='records'), 1):
                # 確保彼得林區的 PE/PEG 分數在寫入資料庫時，不改變 models.py 原有欄位，而是一起存放在 assumptions 中 (若以後有需要)，
                # 這裡直接利用 models.py 既有的欄位做映射，因為這能保證 100% 資料庫欄位對齊：
                # 彼得林區模式下：我們可以用 net_income_growth 存 PEG，而 roe 存 ROE，gross_margin 存 Gross Margin，debt_ratio 存 Debt Ratio。
                # 這樣所有大師都能存進 master_selection table 中！
                
                roe_val = row['roe']
                gm_val = row['gross_margin']
                debt_val = row['debt_ratio']
                growth_val = row['net_income_growth']
                
                # 如果是彼得林區，我們把 peg 數值存在 net_income_growth 欄位！
                if master_name == 'lynch':
                    growth_val = row['peg'] # 讓 net_income_growth 儲存 PEG
                
                conn.execute(text(insert_sql), {
                    'market': market,
                    'symbol': row['symbol'],
                    'name': row['name'],
                    'master_name': master_name,
                    'rank': idx,
                    'close_price': Decimal(str(row['close_price'])) if not pd.isna(row['close_price']) else Decimal('0'),
                    'roe': Decimal(str(roe_val)) if not pd.isna(roe_val) else Decimal('0'),
                    'gross_margin': Decimal(str(gm_val)) if not pd.isna(gm_val) else Decimal('0'),
                    'debt_ratio': Decimal(str(debt_val)) if not pd.isna(debt_val) else Decimal('0'),
                    'net_income_growth': Decimal(str(growth_val)) if not pd.isna(growth_val) else Decimal('0'),
                    'score': Decimal(str(row['score'])) if not pd.isna(row['score']) else Decimal('0'),
                    'updated_at': datetime.datetime.now()
                })
        
        if market == 'us':
            # 自動為美股推薦標的補全 Metadata
            try:
                import yfinance as yf
                
                # 先取得資料庫中已有的 symbols 集合
                check_sql = "SELECT symbol FROM stock_metadata"
                existing_symbols = set()
                with self.op.engine.connect() as check_conn:
                    res = check_conn.execute(text(check_sql)).fetchall()
                    existing_symbols = {r[0] for r in res}
                
                for idx, row in enumerate(df_top.to_dict(orient='records'), 1):
                    symbol = row['symbol']
                    if symbol in existing_symbols:
                        continue
                    
                    try:
                        logger.info(f"Auto-enriching metadata from yfinance for: {symbol}")
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        if not info:
                            continue
                        
                        sector = info.get('sector', 'Other/Unknown')
                        industry = info.get('industry', 'Other/Unknown')
                        market_cap = info.get('marketCap', 0)
                        
                        upsert_sql = """
                            INSERT INTO stock_metadata (symbol, sector, industry, market_cap)
                            VALUES (:symbol, :sector, :industry, :market_cap)
                            ON DUPLICATE KEY UPDATE 
                                sector=VALUES(sector), 
                                industry=VALUES(industry), 
                                market_cap=VALUES(market_cap)
                        """
                        with self.op.engine.begin() as conn:
                            conn.execute(text(upsert_sql), {
                                'symbol': symbol,
                                'sector': sector,
                                'industry': industry,
                                'market_cap': market_cap
                            })
                    except Exception as yf_err:
                        logger.warning(f"Failed to fetch yfinance metadata for {symbol}: {yf_err}")
            except Exception as enrich_err:
                logger.warning(f"Auto metadata enrichment setup failed: {enrich_err}")

        logger.info(f"Successfully saved {master_name} selection results for {market} market.")
