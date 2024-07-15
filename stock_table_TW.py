#本爬蟲程式目前以台灣證券交易所網站所公布的格式進行整理, 若使用其他網頁內容須進行適度修改
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import mysql.connector
import pymysql
from sqlalchemy import create_engine
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

class stock_table_TW:
    #def __init__(self):
    
    def stock_number_get(self):
        url='http://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        driver=webdriver.Edge()
        driver.get(url)
        driver.implicitly_wait(2)
        soup = BeautifulSoup(driver.page_source,"lxml")
        tr=soup.find_all('tr')
        tds = []
        for row in tr:
            data=[td.get_text() for td in row.find_all("td")]
            if len(data) == 7:
                tds.append(data)

        driver.quit()
        tds = pd.DataFrame(tds)
        tds_0 = tds.iloc[1:,0]
        tds_1 = tds.iloc[1:,1:7]
        tds_0_number = []
        tds_0_name =[]
        for i in range (tds_0.shape[0]):
            data_split = str(tds_0.iloc[i]).split('\u3000')
            data_number = data_split[0]
            data_name = data_split[1]
            tds_0_number.append(data_number)
            tds_0_name.append(data_name)
        tds_all = {'有價證卷代號' : tds_0_number,
                   '有價證卷名稱' : tds_0_name,
                   '國際證卷辨識號碼(ISIN Code)' : list(tds_1.iloc[:,0]),
                   '上市日' : list(tds_1.iloc[:,1]),
                   '市場別' : list(tds_1.iloc[:,2]),
                   '產業別' : list(tds_1.iloc[:,3]),
                   'CFI Code' : list(tds_1.iloc[:,4]),
                   '備註' : list(tds_1.iloc[:,5])
                   }
        final_data =pd.DataFrame(tds_all)
        return final_data
    
    def upload_mySQL(self,data):
        connection = create_engine('mysql+pymysql://root:terryHsup9211!@localhost:3306/stock_tw_analyse') 
        data.to_sql(name = 'stock_table_tw', con = connection, schema = 'stock_tw_analyse', if_exists = 'append')
