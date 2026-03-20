import pandas as pd
import mySQL_OP
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import fontManager
import mplfinance as mpf
import time
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers,models
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.models import load_model
import talib
import datetime
import gc
from stock_cost_AI import *

#抓籌碼資料最後一天的數據計算交易主力
#挑選前後各5名取得名單及資料
#名單再10天數據繪圖看主力是否有連續性
#連續性天數多少排序

def last_investor_H_T():
    SQL_OP = mySQL_OP.OP_Fun()
    investor_all = SQL_OP.sel_table_data(table_name = 'stock_investor')
    investor_last_time = investor_all.sort_values(by = '日期', ascending = True).iloc[-1].at['日期']
    mask_last_date = (investor_all['日期']==investor_last_time)
    investor_lastday_data = investor_all[mask_last_date].sort_values(by = '三大法人買賣超股數', ascending = False)
    get_data_list = ['日期','number','外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數(自行買賣)',
                     '自營商買賣超股數(避險)']
    investor_head_5 = investor_lastday_data.head(5)[get_data_list]
    investor_head_5.set_index('日期', inplace = True)
    investor_tail_5 = investor_lastday_data.tail(5)[get_data_list]
    investor_tail_5.set_index('日期', inplace = True)
    return investor_head_5,investor_tail_5

def get_investor(stock,days):
    SQL_OP = mySQL_OP.OP_Fun()
    investor_data = SQL_OP.get_cost_data(table_name = 'stock_investor', stock_number = str(stock))
    investor_sort = investor_data.sort_values(by = '日期', ascending = True)
    get_data_list = ['日期','number','外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數(自行買賣)',
                     '自營商買賣超股數(避險)']
    investor_tail_5 = investor_sort.tail(days)[get_data_list]
    return investor_tail_5

def investor_plt(data,axes = 'None'):
    cols = data.columns
    for col in cols:
        if col =='日期':
            data['日期'] = pd.DatetimeIndex(data['日期']).to_period('D')
        elif col =='number':
            pass
        else :
            #批量去除千位符
            data[col] = data[col].str.replace(',','').astype(int)
    number = data['number'].to_list()
    data = data.drop(['number'],axis=1)
    data.set_index('日期', inplace = True)
    #繪製堆疊柱狀圖
    if axes == 'None':
        data.plot(kind = 'bar',stacked = True, figsize = (12,6),fontsize = 15, title = str(number[-1]),rot = 75)
    else :
        data.plot(kind = 'bar',stacked = True, figsize = (12,10),fontsize = 8, ax = axes,
                  title = str(number[-1]), rot = 75, legend = False)


#load_data_H,load_data_T = last_investor_H_T()
#H_number = load_data_H['number']
#T_number = load_data_T['number']
#matplotlib.rc('font', family='Noto Sans SC')
#fig, axes = plt.subplots(nrows = 2, ncols = 5)
#for i in range(len(H_number)):
#    data = get_investor(H_number.iloc[i],5)
#    investor_plt(data, axes = axes[0,i])
#    if i == 0 :
#        lines, labels = axes[0,i].get_legend_handles_labels()
#for i in range(len(T_number)):
#    data = get_investor(T_number.iloc[i],5)
#    investor_plt(data, axes = axes[1,i])
#fig.suptitle('investor TOP 5 Increase/decrease')
#fig.legend(lines, labels, loc = 'upper right', fontsize = 8)
#fig.subplots_adjust(wspace = 0.5, hspace = 0.5)

fig = stock_cost_AI.investor_TOP_plt(5,5)
plt.show()


