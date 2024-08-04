import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
import time
import gc
import mySQL_OP
import re

class stock_investor():
    
    def get_day(driver):
        date_p = driver.find_element(By.XPATH, '//*[@id="reports"]/hgroup/h2/span[1]')
        date_t = re.split('\年|\月|\日', date_p.text)
        day = date_t[2]
        return day
    
    def get_month(driver):
        date_p = driver.find_element(By.XPATH, '//*[@id="reports"]/hgroup/h2/span[1]')
        date_t = re.split('\年|\月|\日', date_p.text)
        month = date_t[1]
        return month
    
    def get_year(driver):
        date_p = driver.find_element(By.XPATH, '//*[@id="reports"]/hgroup/h2/span[1]')
        date_t = re.split('\年|\月|\日', date_p.text)
        year = date_t[0]
        year = str(int(year)+1911)
        return year
    
    def get_date(driver):
        date_p = driver.find_element(By.XPATH, '//*[@id="reports"]/hgroup/h2/span[1]')
        date_t = re.split('\年|\月|\日', date_p.text)
        year = date_t[0]
        month = date_t[1]
        day = date_t[2]
        year = str(int(year)+1911)
        date = year + '/' + month + '/' + day
        date_list = []
        for i in range(list_r):
            date_list.append(date)
        return date_list
        
    
    def get_html_table_h(table):
        ths = []
        for th in table.find('thead').find_all('th'):
            th = th.get_text()
            ths.append(th)
        ths.insert(0, '日期')
        return ths
    
    def get_html_table_d(table):
        data_g = []
        for row in table.find('tbody').find_all('tr'):
            data = [td.get_text() for td in row.find_all("td")]
            data_g.append(data)
        data_g = pd.DataFrame(data_g)
        return data_g
        
    def get_data_cycle():
        url='https://www.twse.com.tw/zh/trading/foreign/t86.html'
        options = webdriver.EdgeOptions()
        #options.add_argument('--headless=new')						#無頭模式
        #options.add_argument('--disable-gpu')						#禁用GPU加速
        options.add_experimental_option('detach', True)				#不自動關閉瀏覽器
        options.add_argument('--memory-model-cache-size-mb=512')	#設定瀏覽器記憶體大小
        driver=webdriver.Edge(options=options)
        driver.get(url)
        driver.implicitly_wait(5)
        select_e_y = Select(driver.find_element(By.NAME, 'yy'))
        select_e_m = Select(driver.find_element(By.NAME, 'mm'))
        select_e_d = Select(driver.find_element(By.NAME, 'dd'))
        y_cycle = len(select_e_y.options)
        m_cycle = len(select_e_m.options)
        d_cycle = len(select_e_d.options)
        tds = pd.DataFrame()
        for i in range(y_cycle):
            for j in range(m_cycle):
                for k in range(d_cycle):
                    try:
                        select_e_y.select_by_index(i)
                        select_e_m.select_by_index(j)
                        select_e_d.select_by_index(k)
                        select_element = Select(driver.find_element(By.NAME, 'selectType'))
                        select_element.select_by_value('ALL')
                        select_click = driver.find_element(By.CLASS_NAME, 'submit')
                        select_click .click()
                        time.sleep(2)
                        select_element = driver.find_element(By.XPATH, '//*[@id="reports"]/hgroup/div/div[1]/select/option[5]')
                        select_element.click()
                        time.sleep(2)
                        soup = BeautifulSoup(driver.page_source,"lxml")
                        data_table = soup.find('table')
                        data_g = stock_investor.get_html_table_d(table = data_table)
                        date_list = stock_investor.get_date(driver = driver, list_r = data_g.shape[0])
                        data_g.insert(0, column = 'data', value = date_list)
                        data_g = data_g.set_axis(list(range(data_g.shape[1])),axis="columns")
                        tds = pd.concat([tds,data_g], ignore_index = True)
                        ths = stock_investor.get_html_table_h(table = data_table)
                        time.sleep(2)
                        driver.refresh()
                        time.sleep(5)
                        driver.delete_all_cookies()
                        driver.execute_script("window.localStorage.clear();")
                        driver.execute_script("window.sessionStorage.clear();")
                        gc.collect()
                        select_e_y = Select(driver.find_element(By.NAME, 'yy'))
                        select_e_m = Select(driver.find_element(By.NAME, 'mm'))
                        select_e_d = Select(driver.find_element(By.NAME, 'dd'))
                    except Exception as e:
                        print(e)
                        continue
            data = tds.set_axis(ths,axis = 'columns')
            data.drop_duplicates()
            data.dropna()
            mySQL_OP.OP_Fun().upload_all(data, 'stock_tw_analyse', 'stock_investor_data' + '-' + str(stock_investor.get_year(driver)))
            ths = []
            tds = pd.DataFrame()
        driver.quit()

