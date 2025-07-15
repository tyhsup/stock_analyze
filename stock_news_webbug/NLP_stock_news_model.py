import os,csv
from tqdm import tqdm
from zhconv import convert
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN, Embedding, BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, LSTM
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

os.chdir('E:/Infinity/webbug/NLP 情緒正負向分類/NLP 情緒正負向分類')

dataset = []
with open('deal_ch_auto.csv', newline='' ,encoding="utf-8") as csvfile:

  # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    #print(rows)

    counter = 0

    for row in rows:
      counter += 1
      if(counter == 70002):
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

'''### 【Code Block講解】
### Training Model環節
- Step1: 把所有資料分出**訓練集**與**驗證集**-> 80% training dataset  20 testing dataset
- Step2: 文字做**編碼encoding(one hot encoding)**，這算是前處理的一環，進入到高級embedding之前需要先把資料變成one hot的格式(Tokenizer)
- Step3: **統一所有資料長度**以便丟進去神經網路model
- Step4: 保存Tokenizer的編碼權重 Type: .pickle
  - 我們之後自己的輸入也需要編碼，因此需要當前的編碼格式來去幫未來的資料作編碼。'''

label = np.array(label)

# 標籤二值化
lb = LabelBinarizer()
label = lb.fit_transform(label)

# 將資料分為 80% 訓練, 20% 測試
trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, random_state=8787)
# 將訓練資料的單字轉成向量(one hot encoding) 以及補到50個字
#print(testX[22])
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(testX)
testX = tokenizer.texts_to_sequences(testX)


tokenizer.fit_on_texts(trainX)
trainX = tokenizer.texts_to_sequences(trainX)


### 統一所有資料長度為50個向量值
trainX = tf.keras.utils.pad_sequences(trainX, maxlen=50)
testX = tf.keras.utils.pad_sequences(testX, maxlen=50)

#print(testX[22])

# saving tokenizer for 之後要把文字做編碼 之後可以直接不用fit  直接做texts to sequences
with open('words_to_vector_stock_news.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#print(len(testX[22]))


########建立model

seqnc_lngth =  50  
embddng_dim = 64
vocab_size = 35000
model = Sequential()
model.add(InputLayer(input_shape=(seqnc_lngth,)))
model.add(Embedding(vocab_size, embddng_dim, input_length=seqnc_lngth))
model.add(Dropout(0.1))
model.add(LSTM(50))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# # model that takes input and encodes it into the latent space
# rnn = Model(inpt_vec, output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Fitting the RNN to the data


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                              min_delta=1e-4, mode='min', verbose=1)

stop_alg = EarlyStopping(monitor='val_loss', patience=7,
                         restore_best_weights=True, verbose=1)


hist = model.fit(trainX, trainY, batch_size=100, epochs=1000,
                   callbacks=[stop_alg, reduce_lr], shuffle=False,
                   validation_data=(testX, testY))

model.save_weights("positive_or_negative_nofunctional_stock_news_V3.h5")
model.save("final_model_stock_news_V3.h5")

### 【Code Block講解】
### 測試model 精準度
'''- 混淆矩陣(Confusion Matrix)
  - 會統計出所有的預期與結果數量，假如最終預測結果有N，則混淆矩陣為N*N矩陣
  - 目前這邊就是2*2矩陣'''

predictions = model.predict(testX)
#pd.crosstab(testY, predictions, rownames=['實際值'], colnames=['預測值'])
for index in range(len(predictions)):
    if(predictions[index] >= 0.5):
        predictions[index] = 1
    else:
        predictions[index] = 0
#實際顯示測試結果 混淆矩陣目的在於表示出模型的判別精準度，神經網路預測的結果與自己給入的答案有百分之多少是完全符合的。

cm = confusion_matrix(testY, predictions)

print('混淆矩陣：')
print(cm)
print(' balanced_accuracy_score為：', (cm[0,0] + cm[1,1]) / len(predictions))
