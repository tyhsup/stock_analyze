import pandas as pd
import requests
import time
import random
import logging
import os
import sys
from datetime import datetime, timedelta, date
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36"
]

# 確保能正常 import mySQL_OP
try:
    from . import mySQL_OP
except ImportError:
    try:
        from stock_Django import mySQL_OP
    except ImportError:
        import mySQL_OP

from sqlalchemy import text

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
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Referer': 'https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/detail/day.html'
        }

    def get_last_date(self) -> date:
        """獲取新表 stock_investor_tw 中最後一筆上櫃資料的日期"""
        try:
            query = "SELECT date FROM stock_investor_tw ORDER BY date DESC LIMIT 1"
            with self.sql.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                if result:
                    d_val = result[0]
                    if hasattr(d_val, 'strftime'):
                        return d_val
                    elif isinstance(d_val, str):
                        d_str = d_val.replace('/', '-')
                        return datetime.strptime(d_str, '%Y-%m-%d').date()
        except Exception as e:
            logger.error(f"取得最後更新日期失敗: {e}")
        return (datetime.today() - timedelta(days=10)).date()

    def _to_ad_date_str(self, ad_date: date) -> str:
        """Ad date to ROC date string (例: 115/06/11)"""
        tw_year = ad_date.year - 1911
        return f"{tw_year}/{ad_date.strftime('%m/%d')}"
    def _update_single_day_api(self, target_date: date) -> bool:
        """透過 API 更新單日之上櫃三大法人與信用交易數據，並採用雙軌寫入"""
        date_str = self._to_ad_date_str(target_date)
        ad_date_str = target_date.strftime('%Y-%m-%d')
        
        # 1. 抓取三大法人資料
        investor_url = 'https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade'
        investor_payload = {
            'type': 'Daily',
            'sect': 'AL',
            'date': date_str,
            'id': '',
            'response': 'json'
        }
        
        has_investor = False
        try:
            r = requests.post(investor_url, data=investor_payload, headers=self.headers, timeout=15, verify=False)
            if r.status_code == 200:
                data = r.json()
                if data.get('tables') and data['tables'][0].get('data'):
                    table = data['tables'][0]
                    rows = table['data']
                    fields = table['fields']
                    
                    # 雙軌寫入：寫入新表 stock_investor_tw
                    migrated_rows = []
                    for row in rows:
                        if len(row) < 24:
                            continue
                            
                        def to_num(val):
                            try:
                                return float(str(val).replace(',', '').strip())
                            except:
                                return 0.0

                        number = str(row[0]).strip()
                        name = str(row[1]).strip()
                        
                        f_buy = to_num(row[8])
                        f_sell = to_num(row[9])
                        f_net = to_num(row[10])
                        
                        t_buy = to_num(row[11])
                        t_sell = to_num(row[12])
                        t_net = to_num(row[13])
                        
                        d_buy = to_num(row[20])
                        d_sell = to_num(row[21])
                        d_net = to_num(row[22])
                        
                        total_net = to_num(row[23])
                        
                        migrated_rows.append({
                            'date': ad_date_str, 'number': number, 'name': name,
                            'foreign_buy': f_buy, 'foreign_sell': f_sell, 'foreign_net': f_net,
                            'trust_buy': t_buy, 'trust_sell': t_sell, 'trust_net': t_net,
                            'dealer_buy': d_buy, 'dealer_sell': d_sell, 'dealer_net': d_net,
                            'total_net': total_net
                        })
                    
                    if migrated_rows:
                        with self.sql.engine.begin() as conn:
                            # 寫入新表
                            sql_new = """
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
                            conn.execute(text(sql_new), migrated_rows)
                        
                        # 由於舊表 stock_investor 只有 20 欄（包括 date, number），我們必須將 API 24 欄對應過濾並映射到舊表的欄位名稱
                        filtered_rows = []
                        for row in rows:
                            filtered_row = [
                                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                                row[11], row[12], row[13], row[22], row[14], row[15], row[16], row[17], row[18], row[19], row[23]
                            ]
                            filtered_rows.append(filtered_row)

                        old_columns = [
                            '代號', '名稱',
                            '外陸資買進股數(不含外資自營商)', '外陸資賣出股數(不含外資自營商)', '外陸資買賣超股數(不含外資自營商)',
                            '外資自營商買進股數', '外資自營商賣出股數', '外資自營商買賣超股數',
                            '投信買進股數', '投信賣出股數', '投信買賣超股數',
                            '自營商買賣超股數',
                            '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
                            '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)',
                            '三大法人買賣超股數'
                        ]

                        # 雙軌寫入：寫入舊表 stock_investor
                        df_old = pd.DataFrame(filtered_rows, columns=old_columns)
                        df_old.insert(0, 'date', date_str)
                        df_old.rename(columns={'代號': 'number', '名稱': '證券名稱'}, inplace=True)
                        for col in df_old.columns:
                            if any(x in col for x in ['買進', '賣出', '差額', '合計', '股數']):
                                df_old[col] = df_old[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0)
                        
                        try:
                            self.sql.upload_investor_bulk(df_old, 'stock_investor')
                            logger.info(f"✅ TPEx 三大法人雙軌寫入成功: {ad_date_str}")
                        except Exception as e_old_db:
                            logger.warning(f"⚠️ 寫入舊表 stock_investor 失敗 (但新表已寫入): {e_old_db}")
                        
                        has_investor = True
        except Exception as e:
            logger.error(f"❌ 抓取三大法人資料失敗 ({ad_date_str}): {e}")
            raise e

        # 2. 抓取融資融券資料
        margin_url = f"https://www.tpex.org.tw/www/zh-tw/margin/balance?date={date_str}&response=json"
        has_margin = False
        try:
            r = requests.get(margin_url, headers=self.headers, timeout=15, verify=False)
            if r.status_code == 200:
                data = r.json()
                if data.get('tables') and data['tables'][0].get('data'):
                    rows = data['tables'][0]['data']
                    
                    margin_rows = []
                    for row in rows:
                        if len(row) < 18:
                            continue
                            
                        def to_num(val):
                            try:
                                return float(str(val).replace(',', '').strip())
                            except:
                                return 0.0

                        number = str(row[0]).strip()
                        
                        margin_purchase = to_num(row[3])
                        margin_sales = to_num(row[4])
                        margin_balance = to_num(row[6])
                        margin_util = to_num(row[8])
                        
                        short_covering = to_num(row[11])
                        short_sale = to_num(row[12])
                        short_balance = to_num(row[14])
                        short_util = to_num(row[16])
                        
                        margin_rows.append({
                            'date': ad_date_str, 'number': number,
                            'margin_purchase': margin_purchase, 'margin_sales': margin_sales, 'margin_balance': margin_balance,
                            'short_sale': short_sale, 'short_covering': short_covering, 'short_balance': short_balance,
                            'margin_utilization_rate': margin_util, 'short_utilization_rate': short_util
                        })
                        
                    if margin_rows:
                        with self.sql.engine.begin() as conn:
                            sql_margin = """
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
                            conn.execute(text(sql_margin), margin_rows)
                        logger.info(f"✅ TPEx 融資融券寫入成功: {ad_date_str}")
                        has_margin = True
        except Exception as e:
            logger.error(f"❌ 抓取信用交易資料失敗 ({ad_date_str}): {e}")
            raise e

        return has_investor or has_margin

    def update_single_day(self, target_date: date) -> bool:
        """更新單日之上櫃三大法人與信用交易數據，優先使用 API，失敗時 Fallback 至 Selenium 爬蟲"""
        if target_date.weekday() >= 5:
            return False

        success = False
        try:
            success = self._update_single_day_api(target_date)
        except Exception as e:
            logger.warning(f"⚠️ API 抓取上櫃籌碼失敗，將自動 Fallback 回退至 Selenium 爬蟲 ({target_date.strftime('%Y-%m-%d')}): {e}")

        if not success:
            logger.info(f"🔄 正在啟動 Selenium 備份爬蟲進行 Fallback ({target_date.strftime('%Y-%m-%d')})...")
            try:
                selenium_manager = TPExInvestorSeleniumManager()
                success = selenium_manager.update_single_day_selenium(target_date)
            except Exception as se:
                logger.error(f"❌ Selenium Fallback 爬蟲亦執行失敗 ({target_date.strftime('%Y-%m-%d')}): {se}")

        return success

    def update_tpex_investor(self, days_back=10):
        """定時增量更新任務"""
        start_date = self.get_last_date()
        end_date = datetime.today().date()
        
        current_date = start_date
        while current_date <= end_date:
            self.update_single_day(current_date)
            time.sleep(random.uniform(2, 4))
            current_date += timedelta(days=1)

    def update_all_tpex_investors(self):
        """別名進入點，以相容 services.py 的調用"""
        logger.info("啟動上櫃股票籌碼面定時更新...")
        self.update_tpex_investor(days_back=10)

class TPExInvestorSeleniumManager:
    def __init__(self):
        self.sql = mySQL_OP.OP_Fun()
        self.url_investor = 'https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/detail/day.html'
        self.url_margin = 'https://www.tpex.org.tw/zh-tw/mainboard/trading/margin-trading/transactions.html'

    def _to_roc_date_str(self, ad_date: date) -> str:
        tw_year = ad_date.year - 1911
        return f"{tw_year}/{ad_date.strftime('%m/%d')}"

    def update_single_day_selenium(self, target_date: date) -> bool:
        """透過 Selenium 模擬瀏覽器爬取上櫃三大法人與信用交易，並雙軌寫入資料庫"""
        if target_date.weekday() >= 5:
            return False

        roc_date_str = self._to_roc_date_str(target_date)
        ad_date_str = target_date.strftime('%Y-%m-%d')

        options = webdriver.EdgeOptions()
        options.add_argument('--headless') 
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--blink-settings=imagesEnabled=false')
        options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')

        driver = None
        has_investor = False
        has_margin = False

        try:
            driver = webdriver.Edge(options=options)
            wait = WebDriverWait(driver, 20)

            # --- 1. 爬取三大法人 ---
            try:
                driver.get(self.url_investor)
                date_input = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input.date")))
                date_input.click()
                date_input.send_keys(Keys.CONTROL + "a")
                date_input.send_keys(Keys.BACKSPACE)
                date_input.send_keys(roc_date_str)
                date_input.send_keys(Keys.ENTER)

                table_found = False
                for _ in range(10):
                    soup = BeautifulSoup(driver.page_source, "lxml")
                    table = soup.find('table')
                    if table and len(table.find_all('tr')) > 10:
                        table_found = True
                        break
                    time.sleep(1)

                if table_found and "查詢無資料" not in soup.text and "查無資料" not in soup.text:
                    tbody = table.find('tbody')
                    
                    rows = []
                    for tr in tbody.find_all('tr'):
                        tds = [td.get_text().strip().replace(',', '') for td in tr.find_all('td')]
                        if len(tds) >= 24:
                            rows.append(tds)

                    if rows:
                        # 雙軌寫入新表 stock_investor_tw
                        migrated_rows = []
                        for row in rows:
                            def to_num(val):
                                try:
                                    return float(str(val).replace(',', '').strip())
                                except:
                                    return 0.0

                            number = str(row[0]).strip()
                            name = str(row[1]).strip()

                            f_buy = to_num(row[8])
                            f_sell = to_num(row[9])
                            f_net = to_num(row[10])

                            t_buy = to_num(row[11])
                            t_sell = to_num(row[12])
                            t_net = to_num(row[13])

                            d_buy = to_num(row[20])
                            d_sell = to_num(row[21])
                            d_net = to_num(row[22])

                            total_net = to_num(row[23])

                            migrated_rows.append({
                                'date': ad_date_str, 'number': number, 'name': name,
                                'foreign_buy': f_buy, 'foreign_sell': f_sell, 'foreign_net': f_net,
                                'trust_buy': t_buy, 'trust_sell': t_sell, 'trust_net': t_net,
                                'dealer_buy': d_buy, 'dealer_sell': d_sell, 'dealer_net': d_net,
                                'total_net': total_net
                            })

                        if migrated_rows:
                            with self.sql.engine.begin() as conn:
                                sql_new = """
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
                                conn.execute(text(sql_new), migrated_rows)

                            # 由於舊表 stock_investor 只有 20 欄（包括 date, number），我們必須將 Selenium 24 欄對應過濾並映射到舊表的欄位名稱
                            filtered_rows = []
                            for row in rows:
                                filtered_row = [
                                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                                    row[11], row[12], row[13], row[22], row[14], row[15], row[16], row[17], row[18], row[19], row[23]
                                ]
                                filtered_rows.append(filtered_row)

                            old_columns = [
                                '代號', '名稱',
                                '外陸資買進股數(不含外資自營商)', '外陸資賣出股數(不含外資自營商)', '外陸資買賣超股數(不含外資自營商)',
                                '外資自營商買進股數', '外資自營商賣出股數', '外資自營商買賣超股數',
                                '投信買進股數', '投信賣出股數', '投信買賣超股數',
                                '自營商買賣超股數',
                                '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
                                '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)',
                                '三大法人買賣超股數'
                            ]

                            # 雙軌寫入舊表 stock_investor
                            df_old = pd.DataFrame(filtered_rows, columns=old_columns)
                            df_old.insert(0, 'date', roc_date_str)
                            df_old.rename(columns={'代號': 'number', '名稱': '證券名稱'}, inplace=True)
                            for col in df_old.columns:
                                if any(x in col for x in ['買進', '賣出', '差額', '合計', '股數']):
                                    df_old[col] = df_old[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0)

                            try:
                                self.sql.upload_investor_bulk(df_old, 'stock_investor')
                                logger.info(f"✅ Selenium TPEx 三大法人雙軌寫入成功: {ad_date_str}")
                            except Exception as e_old_db:
                                logger.warning(f"⚠️ Selenium 寫入舊表 stock_investor 失敗 (但新表已寫入): {e_old_db}")

                            has_investor = True
            except Exception as e:
                logger.error(f"❌ Selenium 抓取三大法人資料失敗 ({ad_date_str}): {e}")

            # --- 2. 爬取融資融券 ---
            try:
                driver.get(self.url_margin)
                date_input = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input.date")))
                date_input.click()
                date_input.send_keys(Keys.CONTROL + "a")
                date_input.send_keys(Keys.BACKSPACE)
                date_input.send_keys(roc_date_str)
                date_input.send_keys(Keys.ENTER)

                table_found = False
                for _ in range(10):
                    soup = BeautifulSoup(driver.page_source, "lxml")
                    table = soup.find('table')
                    if table and len(table.find_all('tr')) > 10:
                        table_found = True
                        break
                    time.sleep(1)

                if table_found and "查詢無資料" not in soup.text and "查無資料" not in soup.text:
                    tbody = table.find('tbody')
                    rows = []
                    for tr in tbody.find_all('tr'):
                        tds = [td.get_text().strip().replace(',', '') for td in tr.find_all('td')]
                        if len(tds) >= 17:
                            rows.append(tds)

                    if rows:
                        margin_rows = []
                        for row in rows:
                            def to_num(val):
                                try:
                                    return float(str(val).replace(',', '').strip())
                                except:
                                    return 0.0

                            number = str(row[0]).strip()
                            margin_purchase = to_num(row[3])
                            margin_sales = to_num(row[4])
                            margin_balance = to_num(row[6])
                            margin_util = to_num(row[8])

                            short_covering = to_num(row[11])
                            short_sale = to_num(row[12])
                            short_balance = to_num(row[14])
                            short_util = to_num(row[16])

                            margin_rows.append({
                                'date': ad_date_str, 'number': number,
                                'margin_purchase': margin_purchase, 'margin_sales': margin_sales, 'margin_balance': margin_balance,
                                'short_sale': short_sale, 'short_covering': short_covering, 'short_balance': short_balance,
                                'margin_utilization_rate': margin_util, 'short_utilization_rate': short_util
                            })

                        if margin_rows:
                            with self.sql.engine.begin() as conn:
                                sql_margin = """
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
                                conn.execute(text(sql_margin), margin_rows)
                            logger.info(f"✅ Selenium TPEx 融資融券寫入成功: {ad_date_str}")
                            has_margin = True
            except Exception as e:
                logger.error(f"❌ Selenium 抓取信用交易資料失敗 ({ad_date_str}): {e}")

        finally:
            if driver:
                driver.quit()

        return has_investor or has_margin

if __name__ == "__main__":
    manager = TPExInvestorManager()
    manager.update_all_tpex_investors()
