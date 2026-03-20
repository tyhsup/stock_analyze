import os,csv
from tqdm import tqdm
from zhconv import convert
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset


#--------------------------------------------------------------------------
os.chdir('E:/Infinity/webbug/NLP 情緒正負向分類/NLP 情緒正負向分類')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = []
with open('deal_ch_auto.csv', newline='' ,encoding="utf-8") as csvfile:

  # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    #print(rows)

    counter = 0

    for row in rows:
      counter += 1
      if(counter == 11):
        break
      #print(row)
      dataset.append(row)
#print(dataset[2][0])

data = []
label = []
length = []

for index in tqdm(range(1, len(dataset))):
    label.append(dataset[index][0])
    data.append(dataset[index][1])
    length.append(len(dataset[index][1]))
#print(max(length))


## read stop words
stopwords = []
file = open('E:/Infinity/webbug/cn_stop_words.txt', newline='' ,encoding="utf-8").readlines()
for lines in file:
    target_sentence = convert(lines.strip(),'zh-tw')
    stopwords.append(target_sentence)
#print(stopwords[33])

#print(data[:10])  ##print出前10筆來看看(處理前)

#data_utils.download_data_gdown("./") # gdrive-ckip
#已下載先註解

### 切割分詞 去除 stopwords  data資料夾就是ckiptagger的模型
ws = WS('E:/Infinity/webbug/data')

for index in tqdm(range(len(data))):
    word_sentence_list = ws([data[index]])
    reg = []
    for element in word_sentence_list[0]:
        if(element not in stopwords):
            reg.append(element)
    data[index] = ' '.join(reg)
#print(data[:10])  ##print出前10筆來看看(處理後)


label = np.array(label)

# 標籤二值化
lb = LabelBinarizer()
label = lb.fit_transform(label)

# 將資料分為 80% 訓練, 20% 測試
trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, random_state=8787)
# 將訓練資料的單字轉成向量(one hot encoding) 以及補到50個字
#print(testX[22])


# 載入 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# BERT 格式預處理
def encode_bert(texts, tokenizer):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        #max_length=50,
        return_tensors='pt'
    )

#PyTorch Dataset 類別
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        encodings = encode_bert(texts, tokenizer)
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

train_dataset = NewsDataset(trainX, trainY)
test_dataset = NewsDataset(testX,testY)


# 載入model
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=1)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


def train():
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).squeeze(-1)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
train()

model.save_pretrained('./final_model_stock_news_BERT_10')
tokenizer.save_pretrained('./final_tokenizer_stock_news_BERT_10')

