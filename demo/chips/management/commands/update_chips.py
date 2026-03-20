from django.core.management.base import BaseCommand
from chips.models import ChipData
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import re

class Command(BaseCommand):
    help = 'Update Taiwan Stock Chips Data (Institutional Investors)'

    def get_last_date(self):
        try:
            last_entry = ChipData.objects.order_by('-date').first()
            if last_entry:
                return last_entry.date
        except Exception:
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

    def handle(self, *args, **options):
        self.stdout.write("Starting Chips Data Update...")
        
        start_date = self.get_last_date() + timedelta(days=1)
        now_date = datetime.today().date()
        
        if start_date > now_date:
            self.stdout.write("Data is already up to date.")
            return

        # 1. Browser Performance Optimization
        edge_options = webdriver.EdgeOptions()
        edge_options.add_argument('--headless') 
        edge_options.add_argument('--disable-gpu')
        edge_options.add_argument('--blink-settings=imagesEnabled=false')
        
        try:
            driver = webdriver.Edge(options=edge_options)
        except Exception as e:
            self.stderr.write(f"Failed to initialize WebDriver: {e}")
            return
            
        wait = WebDriverWait(driver, 20)
        url = 'https://www.twse.com.tw/zh/trading/foreign/t86.html'
        
        current_date = start_date
        
        while current_date <= now_date:
            if current_date.weekday() >= 5: # Skip weekends
                current_date += timedelta(days=1)
                continue

            try:
                driver.get(url)
                
                # Fill form
                wait.until(EC.presence_of_element_located((By.NAME, 'yy')))
                Select(driver.find_element(By.NAME, 'yy')).select_by_value(str(current_date.year))
                Select(driver.find_element(By.NAME, 'mm')).select_by_value(str(current_date.month))
                Select(driver.find_element(By.NAME, 'dd')).select_by_value(str(current_date.day))
                Select(driver.find_element(By.NAME, 'selectType')).select_by_value('ALL')
                driver.find_element(By.CLASS_NAME, 'submit').click()
                
                # Expand all
                try:
                    select_all_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="reports"]/hgroup/div/div[1]/select/option[5]')))
                    select_all_btn.click()
                except:
                    self.stdout.write(f"No data or select button for {current_date}")
                    current_date += timedelta(days=1)
                    continue

                self.stdout.write(f"Rendering {current_date}...")
                
                # Wait for table
                table_found = False
                for i in range(15):
                    soup = BeautifulSoup(driver.page_source, "lxml")
                    table = soup.find('table')
                    if table and len(table.find_all('tr')) > 500:
                        table_found = True
                        break
                    time.sleep(1)

                if not table_found or "查詢無資料" in soup.text:
                    self.stdout.write(f"{current_date} No Data.")
                    current_date += timedelta(days=1)
                    continue

                # Parse Table
                header_tr = table.find('thead').find_all('tr')[-1]
                ths = [th.get_text().strip() for th in header_tr.find_all('th')]
                ths.insert(0, '日期')
                
                rows = [[td.get_text().strip().replace(',', '') for td in tr.find_all('td')] 
                        for tr in table.find('tbody').find_all('tr') if len(tr.find_all('td')) > 1]

                df = pd.DataFrame(rows)
                actual_date_str = self.get_twse_date(driver) or current_date.strftime('%Y/%m/%d')
                df.insert(0, '日期_temp', actual_date_str)
                df.columns = ths
                df.rename(columns={'證券代號': 'number'}, inplace=True)
                
                # Data Cleaning
                df.replace(['', 'None'], 0, inplace=True)
                num_cols = [c for c in df.columns if any(k in c for k in ['買', '賣', '超', '持股', '金額', '張數'])]
                for c in num_cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                
                # Save to DB (Using defined Model fields)
                # Map Chinese columns to Model fields
                # foreign_buy = '外陸資買進股數(不含外資自營商)'
                # ... check chips/models.py for mapping
                
                bulk_list = []
                for _, row in df.iterrows():
                    # Parse date str to object
                    d_obj = datetime.strptime(actual_date_str, '%Y/%m/%d').date()
                    
                    obj = ChipData(
                        date=d_obj,
                        number=row['number'],
                        foreign_buy=row.get('外陸資買進股數(不含外資自營商)', 0),
                        foreign_sell=row.get('外陸資賣出股數(不含外資自營商)', 0),
                        foreign_net=row.get('外陸資買賣超股數(不含外資自營商)', 0),
                        trust_buy=row.get('投信買進股數', 0),
                        trust_sell=row.get('投信賣出股數', 0),
                        trust_net=row.get('投信買賣超股數', 0),
                        dealer_net=row.get('自營商買賣超股數', 0),
                        total_net=row.get('三大法人買賣超股數', 0),
                    )
                    bulk_list.append(obj)
                
                # Bulk Create with ignore_conflicts (Django 4.1+) 
                # Since we are using managed=False, we assume the table supports it. 
                # If ignore_conflicts is not supported by backend for this specific case, use manual loop.
                ChipData.objects.bulk_create(bulk_list, ignore_conflicts=True)
                
                self.stdout.write(self.style.SUCCESS(f"Saved {len(bulk_list)} records for {actual_date_str}"))

                time.sleep(random.uniform(2, 4))
                current_date += timedelta(days=1)

            except Exception as e:
                self.stderr.write(f"Error on {current_date}: {e}")
                time.sleep(5)
                current_date += timedelta(days=1)
        
        driver.quit()
        self.stdout.write(self.style.SUCCESS("Chips update completed."))
