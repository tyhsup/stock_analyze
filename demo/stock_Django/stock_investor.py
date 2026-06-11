import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import logging
try:
    from . import mySQL_OP
except ImportError:
    try:
        from stock_Django import mySQL_OP
    except ImportError:
        import mySQL_OP
import re
from datetime import datetime, timedelta, date
from typing import Optional
from sqlalchemy import text

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

class StockInvestorManager:
    def __init__(self) -> None:
        self.sql = mySQL_OP.OP_Fun()
        self.url = 'https://www.twse.com.tw/zh/trading/foreign/t86.html'

    def get_last_date(self) -> date:
        """
        查詢資料庫取得最後更新之三大法人資料日期。
        
        SQL 查詢邏輯：
        自 stock_investor 表格中取得日期欄位最大的單一紀錄，用於增量更新。
        若執行時間超過 0.5 秒，將記錄警告日誌。
        
        Returns:
            date: 資料庫中最新資料的日期，若無紀錄則預設回傳 8 年前的日期。
        """
        start_time = time.time()
        try:
            query = "SELECT date FROM stock_investor ORDER BY date DESC LIMIT 1"
            with self.sql.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                elapsed = time.time() - start_time
                if elapsed > 0.5:
                    logger.warning(f"SQL execution took too long: {elapsed:.2f} s for query: {query}")
                if result:
                    d_val = result[0]
                    if hasattr(d_val, 'strftime'):
                        return d_val
                    elif isinstance(d_val, str):
                        d_str = d_val.replace('/', '-')
                        return datetime.strptime(d_str, '%Y-%m-%d').date()
        except Exception as e:
            logger.error(f"查詢最後更新日期失敗: {e}")
        return (datetime.today() - timedelta(days=2920)).date()

    def get_twse_date(self, driver: webdriver.Edge) -> Optional[str]:
        """
        從 Edge 網頁中解析三大法人買賣超表格的當前日期資訊。
        
        工作原理：
        透過 XPath 定位報告標題中的民國日期字串，並將其轉換成西元格式。
        
        Args:
            driver: 已啟動的 Edge 網頁驅動程式。
            
        Returns:
            Optional[str]: 格式為 YYYY/MM/DD 的日期字串，解析失敗時回傳 None。
        """
        try:
            date_p = driver.find_element(By.XPATH, '//*[@id="reports"]/hgroup/h2/span[1]')
            date_t = re.split('年|月|日', date_p.text)
            year = str(int(date_t[0]) + 1911)
            month = date_t[1].zfill(2)
            day = date_t[2].zfill(2)
            return f"{year}/{month}/{day}"
        except Exception as e:
            logger.debug(f"解析網頁日期失敗: {e}")
            return None

    def update_investor_data(self) -> None:
        """
        啟動 Edge 瀏覽器並爬取缺失日期之上市三大法人數據。
        
        工作原理：
        1. 計算今日與資料庫最後日期之差距，跳過週末執行。
        2. 採用 headless 模式與 User-Agent 輪替啟動網頁。
        3. 控制網頁選單選取日期並展開所有資料行。
        4. 透過 BeautifulSoup 與 Pandas 向量化進行高速解析。
        5. 以 try...finally 架構確保瀏覽器資源關閉釋放。
        """
        start_date = self.get_last_date() + timedelta(days=1)
        now_date = datetime.today().date()
        
        if start_date > now_date:
            logger.info("資料已是最新。")
            return

        # --- 1. 瀏覽器效能參數與安全參數優化 ---
        options = webdriver.EdgeOptions()
        options.add_argument('--headless') 
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--blink-settings=imagesEnabled=false') # 禁圖提升 30% 速度
        options.add_argument(f'user-agent={random.choice(USER_AGENTS)}') # User-Agent 輪替
        
        driver = None
        try:
            driver = webdriver.Edge(options=options)
            wait = WebDriverWait(driver, 20)
            current_date = start_date
            
            while current_date <= now_date:
                if current_date.weekday() >= 5: # 跳過週末
                    current_date += timedelta(days=1)
                    continue

                try:
                    driver.get(self.url)
                    
                    # 填寫條件並查詢
                    wait.until(EC.presence_of_element_located((By.NAME, 'yy')))
                    Select(driver.find_element(By.NAME, 'yy')).select_by_value(str(current_date.year))
                    Select(driver.find_element(By.NAME, 'mm')).select_by_value(str(current_date.month))
                    Select(driver.find_element(By.NAME, 'dd')).select_by_value(str(current_date.day))
                    Select(driver.find_element(By.NAME, 'selectType')).select_by_value('ALL')
                    driver.find_element(By.CLASS_NAME, 'submit').click()
                    
                    # --- 2. 展開全部與彈性監測 ---
                    select_all_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reports"]/hgroup/div/div[1]/select/option[5]')))
                    select_all_btn.click()
                    
                    logger.info(f"正在展開 {current_date} 並偵測渲染...")
                    
                    # 每秒檢查一次表格行數，達成即跳出，不需固定等待
                    table_found = False
                    for i in range(15):
                        soup = BeautifulSoup(driver.page_source, "lxml")
                        table = soup.find('table')
                        if table and len(table.find_all('tr')) > 500:
                            table_found = True
                            break
                        time.sleep(1)

                    if not table_found or "查詢無資料" in soup.text:
                        logger.info(f"{current_date} 證交所無資料，跳過。")
                        current_date += timedelta(days=1)
                        continue

                    # --- 3. 高效解析與表頭對齊 ---
                    header_tr = table.find('thead').find_all('tr')[-1]
                    ths = [th.get_text().strip() for th in header_tr.find_all('th')]
                    ths.insert(0, 'date')
                    
                    # 一次性提取所有資料列
                    rows = [[td.get_text().strip().replace(',', '') for td in tr.find_all('td')] 
                            for tr in table.find('tbody').find_all('tr') if len(tr.find_all('td')) > 1]

                    # --- 4. Pandas 向量化清洗 ---
                    df = pd.DataFrame(rows)
                    actual_date = self.get_twse_date(driver) or current_date.strftime('%Y/%m/%d')
                    df.insert(0, 'date_temp', actual_date)
                    df.columns = ths
                    df.rename(columns={'證券代號': 'number'}, inplace=True)
                    
                    # 向量化轉換數值，取代迴圈
                    df.replace(['', 'None'], 0, inplace=True)
                    num_cols = [c for c in df.columns if any(k in c for k in ['買', '賣', '超', '持股', '金額', '張數'])]
                    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

                    # --- 5. 執行優化後的批量寫入 ---
                    self.sql.upload_investor_bulk(df, 'stock_investor')
                    logger.info(f"✅ {actual_date} 成功寫入 {len(df)} 筆。")

                    # 隨機延遲 2 至 4 秒以避免遭受封鎖
                    time.sleep(random.uniform(2.0, 4.0))
                    current_date += timedelta(days=1)

                except Exception as e:
                    logger.error(f"❌ {current_date} 異常: {e}")
                    time.sleep(5)
                    current_date += timedelta(days=1)
        finally:
            if driver:
                driver.quit()
                logger.info("WebDriver 資源已關閉釋放。")
        logger.info("所有程序執行完畢。")

if __name__ == "__main__":
    manager = StockInvestorManager()
    manager.update_investor_data()