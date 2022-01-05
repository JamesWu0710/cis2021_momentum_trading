#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[ ]:


#读取数据
com = pd.read('Commodity.csv')
option = pd.read('option.csv')
fx = pd.read('fx.csv')


# In[ ]:


#商品期货
def ema_12(x):
    x['ema_12'] = x['close'].rolling(12).mean()
    return x
def ema_26(x):
    x['ema_26'] = x['close'].rolling(26).mean()
    return x
def dif(x):
    x['dif'] = x['ema_12']-x['ema_26']
    return x
def macd(x):
    x['dea'] = x['dif'].rolling(9).mean()
    x['macd'] = x['dif']-x['dea']
    return x
def ma5(x):
    x['ma5'] = x['close']-x['close'].rolling(5).mean()
    return x
com = com.groupby('ticker').apply(lambda x: ema_12(x))
com = com.groupby('ticker').apply(lambda x: ema_26(x))
com = com.groupby('ticker').apply(lambda x: dif(x))
com = com.groupby('ticker').apply(lambda x: macd(x))
com = com.groupby('ticker').apply(lambda x: ma5(x))
com['macd_score'] = np.sign(com['macd'])
com['ma5_score'] = np.sign(com['ma5'])


# In[ ]:


#外汇
fx = fx.groupby('ticker').apply(lambda x: ema_12(x))
fx = fx.groupby('ticker').apply(lambda x: ema_26(x))
fx = fx.groupby('ticker').apply(lambda x: dif(x))
fx = fx.groupby('ticker').apply(lambda x: macd(x))
fx = fx.groupby('ticker').apply(lambda x: ma5(x))
fx['macd_score'] = np.sign(fx['macd'])
fx['ma5_score'] = np.sign(fx['ma5'])


# In[ ]:


def backtest_com(start_date,end_date,universe,money,length):
    data = universe[(universe['date'] > start_date) & (universe['date'] < end_date)]
    date = start_date.previous()
    value = [money]
    holding_tick = {}
    holding_vol = {}
    while date <= end_date:
        i = 0
        now = data[data['date']==date]
        pool = now[(now['macd_score']==1) & (now['macd_score']==1)]
        holding_tick[date] = list(pool.ticker.drop_duplicates())
        holding = {}
        whole = 0
        for ticker in holding[date]:
            holding[ticker] = money[i]/(len(holding[date])*pool[pool['ticker']==ticker]['close'])
            now = data[data['date']==date.next()]
            whole = whole + (holding['ticker'] * now[now['ticker']==ticker]['close'])
        holding_vol[date] = holding
        value.append(whole)
        date = date + length
        i = i+1
    #可能会有其他指标的计算，比如年化收益、sharpe ratio等等
    return holding_tick,holding_vol,value
        

