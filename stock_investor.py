import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import logging
import mySQL_OP
import re
from datetime import datetime, timedelta
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

class StockInvestorManager:
    def __init__(self):
        self.sql = mySQL_OP.OP_Fun()
        self.url = 'https://www.twse.com.tw/zh/trading/foreign/t86.html'

    def get_last_date(self):
        try:
            query = "SELECT 日期 FROM stock_investor ORDER BY 日期 DESC LIMIT 1"
            with self.sql.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                if result:
                    d_str = result[0].replace('/', '-')
                    return datetime.strptime(d_str, '%Y-%m-%d').date()
        except:
            pass
        return (datetime.today() - timedelta(days=2920)).date()

    def get_twse_date(self, driver):
        try:
            date_p = driver.find_element(By.XPATH, '//*[@id="reports"]/hgroup/h2/span[1]')
            date_t = re.split('年|月|日', date_p.text)
            year = str(int(date_t[0]) + 1911)
            month = date_t[1].zfill(2)
            day = date_t[2].zfill(2)
            return f"{year}/{month}/{day}"
        except:
            return None

    def update_investor_data(self):
        start_date = self.get_last_date() + timedelta(days=1)
        now_date = datetime.today().date()
        
        if start_date > now_date:
            logger.info("資料已是最新。")
            return

        # --- 1. 瀏覽器效能參數優化 ---
        options = webdriver.EdgeOptions()
        options.add_argument('--headless') 
        options.add_argument('--disable-gpu')
        options.add_argument('--blink-settings=imagesEnabled=false') # 禁圖提升 30% 速度
        
        driver = webdriver.Edge(options=options)
        wait = WebDriverWait(driver, 20)
        
        current_date = start_date
        while current_date <= now_date:
            if current_date.weekday() >= 5:
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
                
                # 每秒檢查一次表格行數，達成即跳出，不需死等
                for i in range(15):
                    soup = BeautifulSoup(driver.page_source, "lxml")
                    table = soup.find('table')
                    if table and len(table.find_all('tr')) > 500:
                        break
                    time.sleep(1)

                if not table or "查詢無資料" in soup.text:
                    logger.info(f"{current_date} 證交所無資料，跳過。")
                    current_date += timedelta(days=1)
                    continue

                # --- 3. 高效解析與表頭對齊 ---
                header_tr = table.find('thead').find_all('tr')[-1]
                ths = [th.get_text().strip() for th in header_tr.find_all('th')]
                ths.insert(0, '日期')
                
                # 一次性提取所有資料列
                rows = [[td.get_text().strip().replace(',', '') for td in tr.find_all('td')] 
                        for tr in table.find('tbody').find_all('tr') if len(tr.find_all('td')) > 1]

                # --- 4. Pandas 向量化清洗 (極速處理) ---
                df = pd.DataFrame(rows)
                actual_date = self.get_twse_date(driver) or current_date.strftime('%Y/%m/%d')
                df.insert(0, '日期_temp', actual_date)
                df.columns = ths
                df.rename(columns={'證券代號': 'number'}, inplace=True)
                
                # 向量化轉換數值，取代迴圈
                df.replace(['', 'None'], 0, inplace=True)
                num_cols = [c for c in df.columns if any(k in c for k in ['買', '賣', '超', '持股', '金額', '張數'])]
                df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

                # --- 5. 執行優化後的批量寫入 ---
                self.sql.upload_investor_bulk(df, 'stock_investor')
                logger.info(f"✅ {actual_date} 成功寫入 {len(df)} 筆。")

                time.sleep(random.uniform(2, 4)) # 友善間隔
                current_date += timedelta(days=1)

            except Exception as e:
                logger.error(f"❌ {current_date} 異常: {e}")
                time.sleep(5)
                current_date += timedelta(days=1)

        driver.quit()
        logger.info("所有程序執行完畢。")

if __name__ == "__main__":
    manager = StockInvestorManager()
    manager.update_investor_data()
