import requests, random
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.wait import WebDriverWait
import time,datetime
from datetime import datetime


class economy_news_webbug:
    def __init__(self):
        self.url = 'https://news.cnyes.com/news/cat/tw_stock'
        self.user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
                           "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",]
        self.stock_list = []#之後修改程式使用list + 迴圈收集不同股票的新聞
        self.stock_number = '2330'
        self.scroll_times = 10
        self.headless_mode = False
        self.gpu_acc = False
        self.news_count = 10
        
    def scroll_controller(self, driver):
        SCROLL_PAUSE_TIME = 2
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def create_driver(self):
        options = webdriver.EdgeOptions()
        options.add_argument(f"user-agent = {self.user_agents}")
        #無頭模式
        if self.headless_mode == True:
            options.add_argument('--headless = new')
        #禁用GPU加速
        if self.gpu_acc == True:
            options.add_argument('--disable-gpu')
        options.add_experimental_option('detach', True)
        return webdriver.Edge(options=options)
    
    def input_stock_number(self, driver, stock_number):
        search_input = driver.find_element(By.TAG_NAME, 'input')
        search_input.send_keys(stock_number)
        time.sleep(1)
        search_input.send_keys(Keys.ENTER)
        time.sleep(3)
        
    def get_news_link(self, driver):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.find_all("a", href=True)
        news_links = set()
        for article in articles:
            href = article['href']
            if '/news/id/' in href:
                full_url = href if href.startswith("http") else f"https://www.cnyes.com{href}"
                news_links.add(full_url)
        news_links = list(news_links)
        news_links_count = news_links[:self.news_count]
        return news_links_count
    
    def get_news(self, stock_number):
        try:
            driver = self.create_driver()
            driver.get(self.url)
            driver.implicitly_wait(5)
            time.sleep(2)
            #操作scroll 元素持續往下滑帶出更多新聞
            #self.scroll_controller(driver)
            #輸入股票代碼或名稱進行搜尋
            self.input_stock_number(driver, stock_number)
            search_input = driver.find_element(By.CSS_SELECTOR, '[data-key="news"]').click()
            time.sleep(2)
            #提取10筆新聞連結
            news_links = self.get_news_link(driver)
            #提取新聞內文
            news_data = []
            for link in news_links:
                try:
                    driver.get(link)
                    time.sleep(2)
                    article_soup = BeautifulSoup(driver.page_source, "html.parser")
                    title = article_soup.find("h1").get_text(strip=True)
                    date_str = article_soup.find("time").get("datetime")
                    publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    content_div = article_soup.find('main',id='article-container')
                    paragraphs = content_div.find_all("p")
                    content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                    news_data.append({
                        "標題": title,
                        "發布時間": publish_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "內文": content,
                        "連結": link
                    })
                except Exception as e:
                    print(f"錯誤擷取新聞：{e}, 連結：{link}")
                    continue
            driver.quit()
            if news_data:
                df = pd.DataFrame(news_data)
                df.to_excel(f'{self.stock_number}_news_{datetime.now().strftime("%Y%m%d")}.xlsx',index=False)
        except Exception as e:
            print(f"運行發生錯誤:{e}")

webbug = economy_news_webbug()
webbug.get_news(webbug.stock_number)
        
        
        
