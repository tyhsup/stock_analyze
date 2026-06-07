import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from bs4 import BeautifulSoup
import time
import random
import logging
import mySQL_OP
from datetime import datetime, timedelta
from sqlalchemy import text
import ssl

# 全域 SSL 猴子補丁
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 日誌設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("stock_investor_final.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 常用 User-Agent 清單 ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36"
]

from typing import Optional
from datetime import date

class StockInvestorManager:
    def __init__(self) -> None:
        self.sql = mySQL_OP.OP_Fun()
        self.session = self._init_session()
        # 改用 http 以避免某些環境下的 SSL 問題與 port 8443 重新導向
        self.api_url = 'http://www.twse.com.tw/rwd/zh/fund/T86'

    def _init_session(self) -> requests.Session:
        """
        初始化具備安全防護與自動重試機制的 HTTP 聯絡 Session。
        
        工作原理：
        透過自訂的 NoVerifyAdapter 徹底避免 Python Requests 常見的 SSL 憑證失效錯誤，
        並隨機指派一個偽裝的瀏覽器 User-Agent Header。
        
        Returns:
            requests.Session: 經配置後的 Requests 連線 Session 物件。
        """
        session = requests.Session()
        
        class NoVerifyAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                kwargs['cert_reqs'] = 'CERT_NONE'
                return super(NoVerifyAdapter, self).init_poolmanager(*args, **kwargs)
            def proxy_manager_for(self, *args, **kwargs):
                kwargs['cert_reqs'] = 'CERT_NONE'
                return super(NoVerifyAdapter, self).proxy_manager_for(*args, **kwargs)

        adapter = NoVerifyAdapter(max_retries=5)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            'User-Agent': random.choice(USER_AGENTS)
        })
        session.verify = False 
        return session

    def get_last_date(self) -> date:
        """
        查詢資料庫取得最後更新之三大法人資料日期。
        
        SQL 查詢邏輯：
        先探查 stock_investor 資料表結構以確保欄位名稱（相容 date 與日期），
        隨後查詢該日期欄位之最大紀錄以作為增量更新之依據。
        若查詢耗時超過 0.5 秒，將記錄警告日誌。
        
        Returns:
            date: 資料庫中最新資料的日期，若無紀錄則預設回傳 8 年前的日期。
        """
        start_time = time.time()
        try:
            with self.sql.engine.connect() as conn:
                # 檢查欄位名稱
                cols_result = conn.execute(text("SHOW COLUMNS FROM stock_investor")).fetchall()
                cols = [c[0] for c in cols_result]
                date_col = 'date' if 'date' in cols else '日期' if '日期' in cols else None
                
                if date_col:
                    query = f"SELECT {date_col} FROM stock_investor ORDER BY {date_col} DESC LIMIT 1"
                    result = conn.execute(text(query)).fetchone()
                    
                    elapsed = time.time() - start_time
                    if elapsed > 0.5:
                        logger.warning(f"SQL execution took too long: {elapsed:.2f} s for query: {query}")
                        
                    if result:
                        d = result[0]
                        if isinstance(d, datetime):
                            return d.date()
                        # 處理字串格式
                        d_str = str(d).replace('/', '-')
                        return datetime.strptime(d_str, '%Y-%m-%d').date()
        except Exception as e:
            logger.warning(f"取得最後日期失敗: {e}")
        # 若無資料，預設抓取 8 年前
        return (datetime.today() - timedelta(days=2920)).date()

    def fetch_data_by_api(self, target_date: date) -> Optional[pd.DataFrame]:
        """
        使用 TWSE RWD API 抓取指定日期的上市三大法人買賣超 JSON 數據。
        
        工作原理：
        透過 HTTP 呼叫證交所三大法人日報表 API，若遇到 SSL 連線憑證錯誤，
        則自動呼叫本機 curl.exe 工具進行安全回退抓取。
        
        Args:
            target_date: 需要抓取的西元日期。
            
        Returns:
            Optional[pd.DataFrame]: 解析後之 Pandas DataFrame，查無資料或異常時回傳 None。
        """
        date_str = target_date.strftime('%Y%m%d')
        params = {
            'date': date_str,
            'selectType': 'ALL',
            'response': 'json'
        }
        
        # 動態更換 User-Agent 以防封鎖
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        
        try:
            # 嘗試標準 requests
            response = self.session.get(self.api_url, params=params, timeout=15, verify=False)
            data = response.json()
        except Exception as e:
            if "SSL" in str(e) or "certificate" in str(e).lower():
                logger.warning(f"偵測到 SSL 錯誤，嘗試使用 curl.exe 作為回退方案 ({target_date})")
                try:
                    import subprocess
                    import json
                    url = f"{self.api_url}?date={date_str}&selectType=ALL&response=json"
                    # 使用 -k 忽略 SSL 錯誤，-L 跟隨重定向，-s 靜默模式
                    result = subprocess.run(['curl.exe', '-L', '-k', '-s', url], capture_output=True, text=True, encoding='utf-8')
                    if result.returncode == 0:
                        data = json.loads(result.stdout)
                    else:
                        logger.error(f"Curl 回退失敗: {result.stderr}")
                        return None
                except Exception as ce:
                    logger.error(f"Curl 執行出錯: {ce}")
                    return None
            else:
                logger.error(f"API 抓取錯誤 ({target_date}): {e}")
                return None
            
        if data.get('stat') != 'OK':
            logger.info(f"{target_date} 證交所查無資料或狀態不正確。")
            return None
            
        fields = data.get('fields', [])
        records = data.get('data', [])
        
        if not records:
            return None
            
        record_len = len(records[0])
        fields_len = len(fields)
        
        # 處理 TWSE API 欄位長度與實際資料不一致或為舊版 16 欄位之情況
        if record_len == 19:
            fields = [
                "證券代號", "證券名稱", 
                "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
                "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
                "投信買進股數", "投信賣出股數", "投信買賣超股數",
                "自營商買賣超股數",
                "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
                "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)",
                "三大法人買賣超股數"
            ]
            df = pd.DataFrame(records, columns=fields)
        elif record_len == 16:
            fields = [
                "證券代號", "證券名稱", 
                "外資買進股數", "外資賣出股數", "外資買賣超股數",
                "投信買進股數", "投信賣出股數", "投信買賣超股數",
                "自營商買賣超股數",
                "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
                "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)",
                "三大法人買賣超股數"
            ]
            df = pd.DataFrame(records, columns=fields)
            # 插入外資自營商空欄位以符合 19 欄位結構
            df.insert(5, "外資自營商買進股數", 0)
            df.insert(6, "外資自營商賣出股數", 0)
            df.insert(7, "外資自營商買賣超股數", 0)
            # 重新命名以對齊資料庫欄位
            df.rename(columns={
                "外資買進股數": "外陸資買進股數(不含外資自營商)",
                "外資賣出股數": "外陸資賣出股數(不含外資自營商)",
                "外資買賣超股數": "外陸資買賣超股數(不含外資自營商)"
            }, inplace=True)
        else:
            logger.warning(f"不支援的欄位數量: {record_len}")
            # 保守策略：以預設欄位名稱截斷或補齊
            if record_len > fields_len:
                fields += [f"未知欄位_{i}" for i in range(fields_len, record_len)]
            else:
                fields = fields[:record_len]
            df = pd.DataFrame(records, columns=fields)
            
        df.insert(0, 'date', target_date.strftime('%Y-%m-%d'))
        return df

    def update_investor_data(self) -> None:
        """
        更新三大法人日報表資料主程序。
        
        工作原理：
        透過遞增日期的循環，逐日下載並進行欄位格式標準化、除逗號並轉為數值等清洗流程，
        最後批次插入資料庫中。
        """
        start_date = self.get_last_date() + timedelta(days=1)
        now_date = datetime.today().date()
        
        if start_date > now_date:
            logger.info("三大法人資料已是最新。")
            return

        current_date = start_date
        while current_date <= now_date:
            # 週末跳過
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            logger.info(f"正在處理日期: {current_date}")
            df = self.fetch_data_by_api(current_date)
            
            if df is not None and not df.empty:
                try:
                    # 1. 欄位名稱標準化
                    df.rename(columns={'證券代號': 'number'}, inplace=True)
                    
                    # 2. 數值清洗 (移除逗號並轉為數值)
                    exclude_cols = ['number', 'date', '證券名稱']
                    num_cols = [c for c in df.columns if c not in exclude_cols]
                    
                    for col in num_cols:
                        df[col] = df[col].astype(str).str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                    # 3. 寫入資料庫
                    self.sql.upload_all(df, 'stock_investor')
                    logger.info(f"✅ {current_date} 成功寫入 {len(df)} 筆資料。")
                    
                    # 避免請求過快，隨機延遲 2 到 4 秒
                    time.sleep(random.uniform(2.0, 4.0))
                except Exception as e:
                    logger.error(f"資料處理/寫入錯誤 ({current_date}): {e}")
            
            current_date += timedelta(days=1)

        logger.info("三大法人更新程序執行完畢。")

if __name__ == "__main__":
    manager = StockInvestorManager()
    manager.update_investor_data()


