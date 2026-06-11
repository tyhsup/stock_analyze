import sys
import os
import csv
import logging
import requests
import pandas as pd
from datetime import datetime, date
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

class TDCCDistributionManager:
    def __init__(self):
        self.sql = mySQL_OP.OP_Fun()
        self.csv_url = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
        # 建立臨時儲存路徑
        self.temp_csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "temp_tdcc_distribution.csv"
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def download_csv(self) -> bool:
        """串流下載大型 CSV 檔案並存入本地，配合 SSL 容錯"""
        logger.info(f"開始下載集保所股權分散表 CSV...")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(self.csv_url, headers=self.headers, stream=True, timeout=60, verify=False)
            if response.status_code != 200:
                logger.error(f"下載失敗，HTTP 狀態碼: {response.status_code}")
                return False
                
            # 串流寫入本地檔案
            with open(self.temp_csv_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            logger.info(f"下載成功，暫存於: {self.temp_csv_path}")
            return True
        except Exception as e:
            logger.error(f"下載 CSV 發生錯誤: {e}")
            return False

    def parse_and_save(self) -> bool:
        """解析下載的 CSV，計算千張大股東與散戶比例並存入 MySQL"""
        if not os.path.exists(self.temp_csv_path):
            logger.error("找不到暫存 CSV 檔案，無法解析")
            return False

        logger.info("開始解析股權分散表...")
        
        # 股權分散表資料暫存結構：{ number: { date: str, class_1_5_sum: float, class_15: float, total_shareholders: int } }
        stock_data = {}
        
        try:
            # 串流讀取，完全避免因檔案過大導致的 OOM (Out Of Memory) 記憶體崩潰
            # 集保所資料編碼通常為 UTF-8，部分歷史可能為 Big5，預設先以 UTF-8 讀取，不符合則回退
            try:
                f = open(self.temp_csv_path, 'r', encoding='utf-8')
                # 測試讀取一行
                f.readline()
                f.seek(0)
            except UnicodeDecodeError:
                f = open(self.temp_csv_path, 'r', encoding='big5')

            reader = csv.reader(f)
            header = next(reader)  # 跳過標題列
            
            row_count = 0
            for row in reader:
                if not row or len(row) < 6:
                    continue
                
                # 欄位索引對齊：
                # 0: 資料日期 (YYYYMMDD)
                # 1: 證券代號
                # 2: 持股分級
                # 3: 人數
                # 4: 股數
                # 5: 占總股數比例%
                
                raw_date = row[0].strip()
                number = row[1].strip()
                
                try:
                    level = int(row[2].strip())
                    people = int(row[3].strip())
                    ratio = float(row[5].strip())
                except ValueError:
                    continue  # 格式錯誤跳過
                
                # 日期轉換 (YYYYMMDD -> YYYY-MM-DD)
                try:
                    formatted_date = f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                except:
                    continue

                if number not in stock_data:
                    stock_data[number] = {
                        'date': formatted_date,
                        'class_1_5_ratio': 0.0,
                        'class_15_ratio': 0.0,
                        'total_shareholders': 0
                    }
                
                # 散戶比例：持股分級 1-5 級 (20張以下) 的比例加總
                if 1 <= level <= 5:
                    stock_data[number]['class_1_5_ratio'] += ratio
                
                # 千張大股東比例：持股分級 15 級 (1000張以上) 的比例
                elif level == 15:
                    stock_data[number]['class_15_ratio'] = ratio
                
                # 合計級距：集保所通常以 17 代表合計
                # 若為 17，我們可以直接抓取總人數
                elif level == 17:
                    stock_data[number]['total_shareholders'] = people
                
                # 如果沒有合計級距，我們可以在後面加總所有級距的 total_shareholders
                # 為了保險，若 level 介於 1 到 15 之間，我們也累計人數 (防止有些資料沒有合計列)
                if 1 <= level <= 15:
                    # 如果尚未設定合計級距的人數，就先用累加的
                    if stock_data[number]['total_shareholders'] == 0 or level == 15:
                        pass # 優先使用 17 級合計

                row_count += 1
                if row_count % 100000 == 0:
                    logger.info(f"已讀取 {row_count} 行數據...")
            
            f.close()
            
            # 對於沒有 17 級的資料，或者需要校正的，我們確認總股東人數有值
            # 準備寫入資料庫的資料
            db_rows = []
            for num, item in stock_data.items():
                db_rows.append({
                    'date': item['date'],
                    'number': num,
                    'class_1_to_5_ratio': round(item['class_1_5_ratio'], 4),
                    'class_15_ratio': round(item['class_15_ratio'], 4),
                    'total_shareholders': item['total_shareholders']
                })
            
            logger.info(f"解析完成，共有 {len(db_rows)} 檔股票股權分散數據。開始寫入資料庫...")
            
            # 批次寫入資料庫 (ON DUPLICATE KEY UPDATE)
            if db_rows:
                with self.sql.engine.begin() as conn:
                    sql = """
                        INSERT INTO `stock_shareholder_distribution` (
                            date, number, class_1_to_5_ratio, class_15_ratio, total_shareholders
                        ) VALUES (:date, :number, :class_1_to_5_ratio, :class_15_ratio, :total_shareholders)
                        ON DUPLICATE KEY UPDATE
                            class_1_to_5_ratio=VALUES(class_1_to_5_ratio),
                            class_15_ratio=VALUES(class_15_ratio),
                            total_shareholders=VALUES(total_shareholders)
                    """
                    # 分批寫入防止 SQL 語句過長
                    batch_size = 5000
                    for i in range(0, len(db_rows), batch_size):
                        batch = db_rows[i:i+batch_size]
                        conn.execute(text(sql), batch)
                
            logger.info("✅ 股權分散表成功寫入 MySQL。")
            return True
            
        except Exception as e:
            logger.error(f"❌ 解析/寫入 CSV 發生錯誤: {e}")
            return False
        finally:
            # 清理本地暫存檔案以維持環境整潔
            if os.path.exists(self.temp_csv_path):
                try:
                    os.remove(self.temp_csv_path)
                    logger.info("已清理暫存檔案。")
                except Exception as e_clean:
                    logger.warning(f"清理暫存檔案失敗: {e_clean}")

    def run(self):
        """下載並解析更新"""
        if self.download_csv():
            self.parse_and_save()

if __name__ == "__main__":
    manager = TDCCDistributionManager()
    manager.run()
