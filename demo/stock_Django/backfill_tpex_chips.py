import sys
import os
import time
import random
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from sqlalchemy import text

# 確保能 import 到 django app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stock_Django import mySQL_OP
except ImportError:
    import mySQL_OP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class TPExBackfiller:
    def __init__(self):
        self.sql = mySQL_OP.OP_Fun()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/detail/day.html'
        }

    def _to_ad_date_str(self, ad_date: date) -> str:
        """將 datetime.date 轉換為民國年格式字串 (例：115/06/11)"""
        tw_year = ad_date.year - 1911
        return f"{tw_year}/{ad_date.strftime('%m/%d')}"

    def fetch_and_save_investor(self, target_date: date) -> bool:
        """抓取並儲存指定日期的三大法人買賣超"""
        date_str = self._to_ad_date_str(target_date)
        url = 'https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade'
        payload = {
            'type': 'Daily',
            'sect': 'AL',
            'date': date_str,
            'id': '',
            'response': 'json'
        }
        
        try:
            r = requests.post(url, data=payload, headers=self.headers, timeout=15)
            if r.status_code != 200:
                logger.error(f"三大法人: HTTP {r.status_code} for {target_date}")
                return False
            
            data = r.json()
            if not data.get('tables') or not data['tables'][0].get('data'):
                logger.warning(f"{target_date} 三大法人: 無數據 (可能是休市日)")
                return False
            
            table = data['tables'][0]
            rows = table['data']
            
            migrated_rows = []
            ad_date_str = target_date.strftime('%Y-%m-%d')
            
            for row in rows:
                if len(row) < 20:
                    continue
                
                # 安全數值轉換
                def to_num(val):
                    try:
                        return float(str(val).replace(',', '').strip())
                    except:
                        return 0.0

                number = str(row[0]).strip()
                name = str(row[1]).strip()
                
                # 外資
                f_buy = to_num(row[2])
                f_sell = to_num(row[3])
                f_net = to_num(row[4])
                
                # 投信
                t_buy = to_num(row[8])
                t_sell = to_num(row[9])
                t_net = to_num(row[10])
                
                # 自營商 (自營 12 + 避險 15)
                d_buy = to_num(row[12]) + to_num(row[15])
                d_sell = to_num(row[13]) + to_num(row[16])
                d_net = to_num(row[11]) # 自營商買賣超合計
                
                # 合計
                total_net = to_num(row[18])
                
                migrated_rows.append((
                    ad_date_str, number, name,
                    f_buy, f_sell, f_net,
                    t_buy, t_sell, t_net,
                    d_buy, d_sell, d_net,
                    total_net
                ))
            
            if migrated_rows:
                dict_list = [
                    {
                        'date': r[0], 'number': r[1], 'name': r[2],
                        'foreign_buy': r[3], 'foreign_sell': r[4], 'foreign_net': r[5],
                        'trust_buy': r[6], 'trust_sell': r[7], 'trust_net': r[8],
                        'dealer_buy': r[9], 'dealer_sell': r[10], 'dealer_net': r[11],
                        'total_net': r[12]
                    } for r in migrated_rows
                ]
                with self.sql.engine.begin() as conn:
                    sql = """
                        INSERT INTO `stock_investor_tw` (
                            date, number, name,
                            foreign_buy, foreign_sell, foreign_net,
                            trust_buy, trust_sell, trust_net,
                            dealer_buy, dealer_sell, dealer_net,
                            total_net
                        ) VALUES (
                            :date, :number, :name,
                            :foreign_buy, :foreign_sell, :foreign_net,
                            :trust_buy, :trust_sell, :trust_net,
                            :dealer_buy, :dealer_sell, :dealer_net,
                            :total_net
                        )
                        ON DUPLICATE KEY UPDATE 
                            name=VALUES(name),
                            foreign_buy=VALUES(foreign_buy), foreign_sell=VALUES(foreign_sell), foreign_net=VALUES(foreign_net),
                            trust_buy=VALUES(trust_buy), trust_sell=VALUES(trust_sell), trust_net=VALUES(trust_net),
                            dealer_buy=VALUES(dealer_buy), dealer_sell=VALUES(dealer_sell), dealer_net=VALUES(dealer_net),
                            total_net=VALUES(total_net)
                    """
                    conn.execute(text(sql), dict_list)
                logger.info(f"✅ 三大法人: 成功寫入 {target_date} 資料 {len(migrated_rows)} 筆。")
                return True
        except Exception as e:
            logger.error(f"❌ 三大法人 {target_date} 發生錯誤: {e}")
        return False

    def fetch_and_save_margin(self, target_date: date) -> bool:
        """抓取並儲存指定日期的融資融券餘額"""
        date_str = self._to_ad_date_str(target_date)
        url = f"https://www.tpex.org.tw/www/zh-tw/margin/balance?date={date_str}&response=json"
        
        try:
            r = requests.get(url, headers=self.headers, timeout=15)
            if r.status_code != 200:
                logger.error(f"融資融券: HTTP {r.status_code} for {target_date}")
                return False
            
            data = r.json()
            if not data.get('tables') or not data['tables'][0].get('data'):
                logger.warning(f"{target_date} 融資融券: 無數據 (可能是休市日)")
                return False
            
            rows = data['tables'][0]['data']
            margin_rows = []
            ad_date_str = target_date.strftime('%Y-%m-%d')
            
            for row in rows:
                if len(row) < 18:
                    continue
                
                def to_num(val):
                    try:
                        return float(str(val).replace(',', '').strip())
                    except:
                        return 0.0

                number = str(row[0]).strip()
                
                # 融資
                margin_purchase = to_num(row[3])
                margin_sales = to_num(row[4])
                margin_balance = to_num(row[6])
                margin_util = to_num(row[8])
                
                # 融券
                short_covering = to_num(row[11])
                short_sale = to_num(row[12])
                short_balance = to_num(row[14])
                short_util = to_num(row[16])
                
                margin_rows.append((
                    ad_date_str, number,
                    margin_purchase, margin_sales, margin_balance,
                    short_sale, short_covering, short_balance,
                    margin_util, short_util
                ))
            
            if margin_rows:
                dict_list = [
                    {
                        'date': r[0], 'number': r[1],
                        'margin_purchase': r[2], 'margin_sales': r[3], 'margin_balance': r[4],
                        'short_sale': r[5], 'short_covering': r[6], 'short_balance': r[7],
                        'margin_utilization_rate': r[8], 'short_utilization_rate': r[9]
                    } for r in margin_rows
                ]
                with self.sql.engine.begin() as conn:
                    sql = """
                        INSERT INTO `stock_margin_balance` (
                            date, number,
                            margin_purchase, margin_sales, margin_balance,
                            short_sale, short_covering, short_balance,
                            margin_utilization_rate, short_utilization_rate
                        ) VALUES (
                            :date, :number,
                            :margin_purchase, :margin_sales, :margin_balance,
                            :short_sale, :short_covering, :short_balance,
                            :margin_utilization_rate, :short_utilization_rate
                        )
                        ON DUPLICATE KEY UPDATE
                            margin_purchase=VALUES(margin_purchase),
                            margin_sales=VALUES(margin_sales),
                            margin_balance=VALUES(margin_balance),
                            short_sale=VALUES(short_sale),
                            short_covering=VALUES(short_covering),
                            short_balance=VALUES(short_balance),
                            margin_utilization_rate=VALUES(margin_utilization_rate),
                            short_utilization_rate=VALUES(short_utilization_rate)
                    """
                    conn.execute(text(sql), dict_list)
                logger.info(f"✅ 融資融券: 成功寫入 {target_date} 資料 {len(margin_rows)} 筆。")
                return True
        except Exception as e:
            logger.error(f"❌ 融資融券 {target_date} 發生錯誤: {e}")
        return False

    def backfill(self, start_date_str: str, end_date_str: str):
        """執行區間回補"""
        start = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        current = start
        while current <= end:
            # 跳過週末
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue
                
            logger.info(f"=== 正在處理 {current} ===")
            
            # 三大法人
            self.fetch_and_save_investor(current)
            time.sleep(random.uniform(1, 2))
            
            # 融資融券
            self.fetch_and_save_margin(current)
            
            # 隨機延遲防封鎖 (依據 stock-cost-guide.md 要求之 2-5s 延遲)
            sleep_time = random.uniform(2, 4)
            logger.info(f"等待 {sleep_time:.2f} 秒以符合 Rate Limit 規範...")
            time.sleep(sleep_time)
            
            current += timedelta(days=1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # 預設回補最近 10 天
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=10)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        logger.info(f"未指定日期範圍，預設回補最近 10 天 ({start_str} 至 {end_str})")
    else:
        start_str = sys.argv[1]
        end_str = sys.argv[2]
        
    backfiller = TPExBackfiller()
    backfiller.backfill(start_str, end_str)
