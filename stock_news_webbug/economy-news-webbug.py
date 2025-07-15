import requests, random, os, pickle, csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from zhconv import convert
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential, save_model, load_model
from tensorflow.keras.layers import SimpleRNN, Embedding, BatchNormalization, Dense, Activation, Input, Dropout, LSTM, InputLayer
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm.notebook import tqdm
import tkinter as tk
from tkinter import messagebox
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments,pipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset

class economy_news_webbug:
    def __init__(self):
        self.url = 'https://news.cnyes.com/news/cat/tw_stock'
        self.user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
                           "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",]
        self.stock_list = []#之後修改程式使用list + 迴圈收集不同股票的新聞
        self.stock_number = '2330'
        self.headless_mode = False
        self.gpu_acc = False
        self.news_count = 500
        self.file_path = f'E:/Infinity/webbug/{self.stock_number}_news.xlsx'
        #self.ckiptagger_data = data_utils.download_data_gdown("./")
        self.ws = WS("./data")
        self.pos = POS("./data")
        self.ner = NER("./data")
        self.model_path = 'final_model_stock_news_BERT.h5'
        self.weights_path = 'ositive_or_negative_nofunctional_stock_news_BERT.h5'
        self.token_path ='words_to_vector_stock_news.pickle'
        self.bert_tokenizer = BertTokenizer.from_pretrained('final_tokenizer_stock_news_BERT_1w')
        self.bert_model = BertForSequenceClassification.from_pretrained('final_model_stock_news_BERT_1w')
        
    def scroll_controller(self, driver):
        SCROLL_PAUSE_TIME = 2
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        
    def scroll_controller_last_position(self, driver):
        SCROLL_PAUSE_TIME = 2
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            time.sleep(SCROLL_PAUSE_TIME)
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
    
    def input_stock_number(self, driver):
        search_input = driver.find_element(By.TAG_NAME, 'input')
        search_input.send_keys(self.stock_number)
        time.sleep(1)
        search_input.send_keys(Keys.ENTER)
        time.sleep(3)
        
    def get_news_link(self, driver):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        news_links = set()
        articles = soup.find_all("a", href=True)
        for article in articles:
            href = article['href']
            if '/news/id/' in href:
                full_url = href if href.startswith("http") else f"https://www.cnyes.com{href}"
                news_links.add(full_url)
        news_links = list(news_links)
        return news_links
    
    def get_news(self):
        try:
            driver = self.create_driver()
            driver.get(self.url)
            driver.maximize_window()
            driver.implicitly_wait(5)
            time.sleep(2)
            #輸入股票代碼或名稱與新聞數量進行搜尋
            self.stock_number = input_number.get()
            self.news_count = int(news_count.get())
            self.input_stock_number(driver)
            search_input = driver.find_element(By.CSS_SELECTOR, '[data-key="news"]').click()
            time.sleep(2)
            while True:
                #提取新聞連結
                last_height = driver.execute_script("return document.body.scrollHeight")
                news_links = self.get_news_link(driver)
                if len(news_links) >= self.news_count:
                    news_links = news_links[:self.news_count]
                    break
                else:
                    #操作scroll 元素持續往下滑帶出更多新聞
                    self.scroll_controller(driver)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        print(f'已拉到最底層, 新聞總數 : {len(news_links)}')
                        break
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
                    pass
            driver.quit()
            if news_data:
                df = pd.DataFrame(news_data)
                #df.to_excel(f'{self.stock_number}_news_{datetime.now().strftime("%Y%m%d")}.xlsx',index=False)
            return df
        except Exception as e:
            print(f"運行發生錯誤:{e}")
            
    def _read_excel(self):
        return pd.read_excel(self.file_path)
    
    def _write_excel(self, df):
        df.to_excel(self.file_path, index = False)
        
    def add_news(self, url):
        df = self._read_excel()
        if url in df['連結'].values:
            print('該新聞已存在')
            return
        article = url
        news_data = []
        if article :
            driver = self.create_driver()
            driver.get(article)
            time.sleep(2)
            article_soup = BeautifulSoup(driver.page_source,'html.parser')
            title = article_soup.find('h1').get_text(strip = True)
            date_str = article_soup.find('time').get('datetime')
            publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            content_div = article_soup.find('main',id='article-container')
            paragraphs = content_div.find_all('p')
            content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
            news_data.append({
                "標題": title,
                "發布時間": publish_date.strftime("%Y-%m-%d %H:%M:%S"),
                "內文": content,
                "連結": article
                })
            df = pd.concat([df,pd.DataFrame(news_data)])#, ignore_index = True)
            driver.quit()
            self._write_excel(df)
            print('新增news 成功')
        else:
            print('無法擷取新聞內容')
            
    def delete_news(self,df, identifier, by='title'):
        #df = self._read_excel()
        if by == 'title':
            df = df[df['標題'] != identifier]
        elif by == 'link':
            df = df[df['連結'] != identifier]
        elif by == 'time':
            df = df[df['時間'] != identifier]
        else:
            print("請指定正確的刪除依據：'title' 或 'link' 或 'time'")
            return
        #self._write_excel(df)
        print("刪除新聞成功。")
        return df
    
    def search_news(self,keyword,limit = 5):
        df = self._read_excel()
        filtered = df[df['標題'].str.contains(keyword,case=False,na=False)]
        print(filtered.head(limit))
        
    def _update_database(self):
        data = self.get_news()
        titles = list(data.loc[:,'標題'])
        df = self._read_excel()
        for title in titles:
            if title in df['標題'].values:
                print('該新聞已存在')
                data = self.delete_news(data, identifier = title, by = 'title')
        df = pd.concat([df,data])
        self._write_excel(df)
        print('update news database成功')
         
    def print_word_pos_sentence(self, word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            print(f"{word}({pos})", end="\u3000")
        print()
        return
    
    def print_word_pos_array(self, news_number):
        sentence_list = self._read_excel()['內文']
        word_sentence_list = self.ws(sentence_list)
        pos_sentence_list = self.pos(word_sentence_list)
        entity_sentence_list = self.ner(word_sentence_list, pos_sentence_list)
        entity_sentence_dataframe = self.word_pos_DataFrame(entity_sentence_list[news_number])
        return entity_sentence_dataframe
    
    def word_pos_DataFrame(self, data):
        dataFrame = pd.DataFrame(data).drop([0,1],axis = 1).rename(columns = {2:'詞性', 3:'文字'})
        return dataFrame
    
    def print_one_hot_code(self, data):
        one_hot_code_frame = pd.get_dummies(data['詞性'], dtype = int)
        return one_hot_code_frame
    
    def _read_stop_words(self, data_txt):
        stopwords = []
        file = open(data_txt, newline='' ,encoding="utf-8").readlines()
        for lines in file:
            target_sentence = convert(lines.strip(),'zh-tw')
            stopwords.append(target_sentence)
        return stopwords
    
    def _pre_data_label(self, data_csv):
        dataset = []
        with open(data_csv, newline='' ,encoding="utf-8") as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)
            #print(rows)
            counter = 0
            for row in rows:
                counter += 1
                if(counter == 1201):
                    break
                #print(row)
                dataset.append(row)
        data = []
        label = []
        length = []
        for index in tqdm(range(1, len(dataset))):
            label.append(dataset[index][0])
            data.append(dataset[index][1])
            length.append(len(dataset[index][1]))
        return data,label,length
    
    def _data_pre_predictions(self,data_data, label_data, tokenizer, model):
        label = np.array(label_data)
        trainX, testX, trainY, testY = train_test_split(data_data, label_data, test_size=0.2, random_state=8787)
        testX = tokenizer.texts_to_sequences(testX)
        trainX = tokenizer.texts_to_sequences(trainX)
        trainX = tf.keras.utils.pad_sequences(trainX, maxlen=50)
        testX = tf.keras.utils.pad_sequences(testX, maxlen=50)
        predictions = model.predict(testX)
    
    def _predictions(self, news_number, tokenizer, model, stopwords):
        news = self._read_excel()
        input_news = news.loc[:,'內文'][news_number]
        ws = WS('E:/Infinity/webbug/data')
        word_sentence_list = ws(input_news)
        reg = []
        for word in word_sentence_list[0]:
            if(word not in stopwords):
                reg.append(word)
        input_news  = " ".join(reg)
        input_news  = tokenizer.texts_to_sequences([input_news])
        input_news  = tf.keras.utils.pad_sequences(input_news, maxlen=50)
        result = model.predict(input_news)
        return result
    
    def _predictions_BERT(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = self.bert_tokenizer
        model = self.bert_model.to(device)
        model.eval()
        inputs = tokenizer(
            text,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = 50
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits)[0].item()
            print(score)
        return 1 if score >= 0.5 else 0
    
    def GB_analyze_LSTM(self, news_number):
        os.chdir('E:/Infinity/webbug/')
        ## read stop words
        stopwords = webbug._read_stop_words(data_txt = 'cn_stop_words.txt')
        # loading先訓練好的one-hot encodeing
        with open(self.token_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        #load_model
        model = load_model(self.model_path)
        model.load_weights(self.weights_path)
        model.summary()
        #讀入預訓練資料
        data, label, length = webbug._pre_data_label(data_csv = '測試資料_繁體.csv')
        #model 預訓練
        webbug._data_pre_predictions(data_data = data, label_data = label, tokenizer = tokenizer, model = model)
        #讀入新聞做prediction
        result = webbug._predictions(news_number, tokenizer = tokenizer, model = model, stopwords = stopwords)
        return result
        '''if(result >= 0.5):
            print('正面' , result)
        else:
            print('負面' , result)'''
    
    def GB_analyze_BERT(self, news_number):
        os.chdir('E:/Infinity/webbug/')
        # 載入新聞
        news_df = self._read_excel()
        content = news_df.loc[:,'內文'][news_number]
        # 去除停用詞（可選）
        stopwords = self._read_stop_words('cn_stop_words.txt')
        ws = WS('E:/Infinity/webbug/data')
        word_sentence_list = ws([content])
        filtered_words = [w for w in word_sentence_list[0] if w not in stopwords]
        clean_text = " ".join(filtered_words)
        # 預測
        score = self._predictions_BERT(clean_text)
        #print(f'新聞第 {news_number} 篇 預測分數: {score:.3f}')
        return score
    
    def _result_plot(self):
        pos = 0
        neg = 0
        for i in range(10):
            result = webbug.GB_analyze_BERT(news_number = i)
            if (result >= 0.5):
                pos += 1
            else:
                neg += 1
        attribute = ['positive','Negative']
        news_analytz_result = [pos,neg]
        data = {'attribute': attribute, 'news_analytz_result' : news_analytz_result}
        df = pd.DataFrame(data)
        explode = [0.2,0]
        fig ,ax = plt.subplots(figsize=(12,8))
        ax.pie(df.loc[:,'news_analytz_result'],
              labels=df.loc[:,'attribute'],
              autopct='%.1f%%', # 比例格式
              explode=explode,  # 凸顯
              shadow = True),   # 陰影
        ax.set_title("news analytz Positive & Negative",fontsize=20)
        ax.legend(attribute, loc=3, fontsize='small')
        print("分析結果（正面/負面）:", pos, "/", neg)
        #photo = plt.savefig("my_plot.png")
        return fig
        
    def _run_analyze(self):
        self.stock_number = input_number.get()
        self.news_count = int(news_count.get())
        fig = self._result_plot()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
          

webbug = economy_news_webbug()
#webbug.get_news('2330')
#new_news = 'https://news.cnyes.com/news/id/6002622'
#webbug.delete_news(new_news, by ='link')
#webbug.search_news('台積電', limit = 5)
#webbug._result_plot()
#------------------------------------------------------------
window = tk.Tk()
window.title('news_analytz')
window.geometry('760x800')
window.resizable(False, False)
input_number = tk.Entry()
input_number.pack(side = 'top')
label = tk.Label(window, text = 'stock_number')
label.place(x=220, y=0)
news_count = tk.Entry()
news_count.pack(side = 'top')
labe2 = tk.Label(window, text = 'news_count')
labe2.place(x=220, y=15)
execute = tk.Button(text = '開始', command = webbug._run_analyze)
execute.pack(side = 'top')
execute = tk.Button(text = 'update database', command = webbug._update_database)
execute.pack(side = 'top')
window.mainloop()


'''entity_sentence_dataframe = webbug.print_word_pos_array(news_number = 2)
one_hot_code_data = webbug.print_one_hot_code(entity_sentence_dataframe)
print(entity_sentence_dataframe)
print(one_hot_code_data)'''

'''for i, sentence in enumerate(sentence_list):
    print()
    print(f"'{sentence}'")
    webbug.print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
    for entity in sorted(entity_sentence_list[i]):
        print(entity)'''
