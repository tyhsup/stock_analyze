import pandas as pd
import requests
import time
import random
import logging
import os
import sys
from datetime import datetime, timedelta

# Fix import path when run directly
try:
    from . import mySQL_OP
except ImportError:
    try:
        from stock_Django import mySQL_OP
    except ImportError:
        import mySQL_OP

from sqlalchemy import text

# --- 日誌設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("stock_investor_tpex.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TPExInvestorManager:
    def __init__(self):
        self.sql = mySQL_OP.OP_Fun()
        self.api_url = 'https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Referer': 'https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/detail/day.html'
        }

    def get_last_date(self):
        """獲取資料庫中最後一筆上櫃資料的日期（註：目前 stock_investor 混用，我們抓取整體的最後日期）"""
        try:
            query = "SELECT 日期 FROM stock_investor ORDER BY 日期 DESC LIMIT 1"
            with self.sql.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                if result:
                    d_str = result[0].replace('/', '-')
                    return datetime.strptime(d_str, '%Y-%m-%d').date()
        except Exception as e:
            logger.error(f"獲取最後日期失敗: {e}")
        return (datetime.today() - timedelta(days=30)).date()

    def fetch_day_data(self, target_date: datetime.date):
        """抓取指定日期的 TPEx 法人數據"""
        date_str = target_date.strftime('%Y/%m/%d')
        payload = {
            'type': 'Daily',
            'sect': 'AL',
            'date': date_str,
            'id': '',
            'response': 'json'
        }
        
        try:
            response = requests.post(self.api_url, data=payload, headers=self.headers, timeout=15)
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code} for {date_str}")
                return None
            
            json_data = response.json()
            if not json_data.get('tables') or not json_data['tables'][0].get('data'):
                logger.warning(f"{date_str} TPEx 無法人資料（可能是休市日）。")
                return None
            
            table = json_data['tables'][0]
            fields = table['fields']
            data = table['data']
            
            df = pd.DataFrame(data, columns=fields)
            
            # --- 格式轉換以符合 stock_investor 表結構 ---
            # stock_investor 預期欄位: 日期, number, 證券名稱, ...
            
            # TPEx 回傳日期為民國，我們統一儲存為 YYYY/MM/DD
            df.insert(0, '日期', date_str)
            df.rename(columns={'代號': 'number', '名稱': '證券名稱'}, inplace=True)
            
            # 清理數值欄位（移除逗號）
            for col in df.columns:
                if any(x in col for x in ['買進', '賣出', '買賣超', '合計']):
                    df[col] = df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"抓取 {date_str} 異常: {e}")
            return None

    def update_tpex_investor(self, days_back=10):
        """更新最近幾日的上櫃法人資料"""
        start_date = self.get_last_date()
        end_date = datetime.today().date()
        
        # 如果是今天，且還沒收盤，可能沒資料，通常 15:00 後才有
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() >= 5: # 跳過週末
                current_date += timedelta(days=1)
                continue
                
            df = self.fetch_day_data(current_date)
            if df is not None and not df.empty:
                # 寫入資料庫
                # 注意：stock_investor 表結構可能需要與 df 欄位完全對齊
                # 這裡調用 mySQL_OP 的批量寫入
                try:
                    self.sql.upload_investor_bulk(df, 'stock_investor')
                    logger.info(f"✅ TPEx {current_date} 成功寫入 {len(df)} 筆。")
                except Exception as e:
                    logger.error(f"寫入資料庫失敗: {e}")
            
            time.sleep(random.uniform(2, 4))
            current_date += timedelta(days=1)

if __name__ == "__main__":
    manager = TPExInvestorManager()
    manager.update_tpex_investor()
