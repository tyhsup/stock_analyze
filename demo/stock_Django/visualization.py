import pandas as pd
import datetime,gc,io,urllib,base64,time
import numpy as np
from io import BytesIO
from PIL import Image
import mySQL_OP
import matplotlib
import matplotlib.dates as mdates
from matplotlib.font_manager import fontManager
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import mplfinance as mpf
import mpld3
import talib
import yfinance as yf

class visualization():
    
    def transfer_numeric(data):
        cols = data.columns
        for col in cols:
            if col =='日期':
                data['日期'] = pd.DatetimeIndex(data['日期']).to_period('D')
            elif col =='number':
                pass
            elif col =='證券名稱':
                pass
            else :
                #批量去除千位符
                data[col] = data[col].str.replace(',','')
                data[col] = data[col].replace(to_replace = '', value = 0)
                data[col] = data[col].replace(to_replace = 'None', value = 0)
                data[col] = data[col].fillna(0)
                #字串轉整數
                data[col] = pd.to_numeric(data[col], errors = 'coerce' ,downcast = 'signed')
        return data
    
    def date_create(start_date, days):
        bdate_range = pd.bdate_range(start = str(start_date), periods = days, freq = 'B', name = 'Date')
        date = pd.DataFrame(bdate_range)
        return date
    
    def get_investor_days(data,stock,days):
        mask = (data['number']==str(stock))
        data2 = data[mask]
        data_sort = data2.sort_values(by = '日期', ascending = True)
        investor_tail = data_sort.tail(days)
        #investor_tail.set_index('日期', inplace = True)
        return investor_tail

    def last_investor_H_T(data, amount):
        investor_last_time = data.sort_values(by = '日期', ascending = True).iloc[-1].at['日期']
        mask_last_date = (data['日期']==investor_last_time)
        investor_lastday_data = data[mask_last_date]
        investor_lastday_data.sort_values(by = '三大法人買賣超股數', ascending = False, inplace = True)
        investor_head= investor_lastday_data.head(int(amount))
        investor_head.set_index('日期', inplace = True)
        investor_tail= investor_lastday_data.tail(int(amount))
        investor_tail.set_index('日期', inplace = True)
        return investor_head,investor_tail

    def investor_plt(data, axes = 'NA'):
        matplotlib.rc('font', family='Noto Sans SC')
        cols = data.columns
        number = data['number'].to_list()
        data = data.drop(['number','證券名稱'],axis=1)
        data['日期']= pd.datetime(data['日期'])
        mask = ['外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數','自營商買賣超股數(自行買賣)','自營商買賣超股數(避險)']
        data = data[mask]
        if axes == 'NA':
            fig,ax = plt.subplots(figsize = (12,6))
        else :
            ax = axes
            fig = ax.figure
        #繪製堆疊柱狀圖
        data.plot(kind = 'bar',stacked = True,x = data['日期'],fontsize = 15, title = str(number[-1]),rot = 75, ax = ax)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        fig.autofmt_xdate()
        plt.minorticks_off()
        mpld3.show()
        #return fig

    def investor_TOP_plt(data, amount, days):
        load_data_H,load_data_T = visualization.last_investor_H_T(data, amount)
        H_number = load_data_H['number']
        T_number = load_data_T['number']
        matplotlib.rc('font', family='Noto Sans SC')
        fig, axes = plt.subplots(nrows = 2, ncols = amount, sharex = True, sharey = True)
        for i in range(len(H_number)):
            H_data = visualization.get_investor_days(data, H_number.iloc[i],days)
            visualization.investor_plt(H_data, axes = axes[0,i])
            if i == 0 :
                handles, labels = axes[0,i].get_legend_handles_labels()
        for i in range(len(T_number)):
            T_data = visualization.get_investor_days(data, T_number.iloc[i],days)
            visualization.investor_plt(T_data, axes = axes[1,i])
        fig.suptitle('investor ' + 'TOP' + str(amount) + ' Increase/decrease')
        fig.legend(handles, labels, loc = 'outside upper right', fontsize = 7)
        fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
        return fig
    
    def stock_plot(data, pred_days = 'Na'):
        data['Date'] = pd.to_datetime(data['Date'])
        fig = make_subplots(rows = 2,cols =1, shared_xaxes = True,
                            vertical_spacing = 0.03,
                            row_width = [0.2, 0.7])
        fig.add_trace(go.Candlestick(x=data['Date'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     increasing_line_color = 'green',
                                     decreasing_line_color = 'red',
                                     showlegend = False),
                                     row = 1, col = 1)
        fig.update_layout(title='K 線圖', xaxis_title='日期', yaxis_title='價格')
        fig.show()

SQL = mySQL_OP.OP_Fun()
'''data = SQL.get_cost_data(table_name = 'stock_investor')
data2 = visualization.get_investor_days(data,'2330',5)
data_trans = visualization.transfer_numeric(data2)
visualization.investor_plt(data_trans)'''
data = SQL.get_cost_data(table_name = 'stock_cost',stock_number = '2330')
visualization.stock_plot(data)


