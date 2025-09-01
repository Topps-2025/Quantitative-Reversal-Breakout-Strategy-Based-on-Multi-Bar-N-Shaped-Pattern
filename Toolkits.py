# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:18:31 2023
功能：工具包
@author: liukaizhi
"""

import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from datetime import datetime,timedelta,date

from scipy import stats
import math
import warnings

from Tools.LoadSQL import *
warnings.filterwarnings('ignore')
import random
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from statsmodels.tools.validation import array_like, PandasWrapper
import itertools
import os



def hprescott(X, side=2, smooth=1600, freq=''):
    """
    功能：实现HP滤波
    """

    '''
    Hodrick-Prescott filter with the option to use either the standard two-sided 
    or one-sided implementation. The two-sided implementation leads to equivalent
    results as when using the statsmodel.tsa hpfilter function

    Parameters
    ----------
    X : array-like
        The time series to filter (1-d), need to add multivariate functionality.

    side : int
           The implementation requested. The function will default to the standard
           two-sided implementation.

    smooth : float 
            The Hodrick-Prescott smoothing parameter. A value of 1600 is
            suggested for quarterly data. Ravn and Uhlig suggest using a value
            of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly
            data. The function will default to using the quarterly parameter (1600).

    freq : str
           Optional parameter to specify the frequency of the data. Will override
           the smoothing parameter and implement using the suggested value from
           Ravn and Uhlig. Accepts annual (a), quarterly (q), or monthly (m)
           frequencies.

    Returns
    -------

    cycle : ndarray
            The estimated cycle in the data given side implementation and the 
            smoothing parameter.

    trend : ndarray
            The estimated trend in the data given side implementation and the 
            smoothing parameter.

    References
    ----------
    Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An
        Empirical Investigation." `Carnegie Mellon University discussion
        paper no. 451`.

    Meyer-Gohde, A. 2010. "Matlab code for one-sided HP-filters."
        `Quantitative Macroeconomics & Real Business Cycles, QM&RBC Codes 181`.

    Ravn, M.O and H. Uhlig. 2002. "Notes On Adjusted the Hodrick-Prescott
        Filter for the Frequency of Observations." `The Review of Economics and
        Statistics`, 84(2), 371-80.

    Examples
    --------
    from statsmodels.api import datasets, tsa
    import pandas as pd
    dta = datasets.macrodata.load_pandas().data
    index = pd.DatetimeIndex(start='1959Q1', end='2009Q4', freq='Q')
    dta.set_index(index, inplace=True)

    #Run original tsa.filters two-sided hp filter
    cycle_tsa, trend_ts = tsa.filters.hpfilter(dta.realgdp, 1600)
    #Run two-sided implementation
    cycle2, trend2 = hprescott(dta.realgdp, 2, 1600)
    #Run one-sided implementation
    cycle1, trend1 = hprescott(dta.realgdp, 1, 1600)
    '''

    # Determine smooth if a specific frequency is given
    if freq == 'q':
        smooth = 1600  # quarterly
    elif freq == 'a':
        smooth = 6.25  # annually
    elif freq == 'm':
        smooth = 129600  # monthly
    elif freq != '':
        print('''Invalid frequency parameter inputted. Defaulting to defined smooth
        parameter value or 1600 if no value was provided.''')

    pw = PandasWrapper(X)
    X = array_like(X, 'X', ndim=1)
    T = len(X)

    # Preallocate trend array
    trend = np.zeros(len(X))

    # Rearrange the first order conditions of minimization problem to yield matrix
    # First and last two rows are mirrored
    # Middle rows follow same pattern shifting position by 1 each row

    a1 = np.array([1 + smooth, -2 * smooth, smooth])
    a2 = np.array([-2 * smooth, 1 + 5 * smooth, -4 * smooth, smooth])
    a3 = np.array([smooth, -4 * smooth, 1 + 6 * smooth, -4 * smooth, smooth])

    Abeg = np.concatenate(([np.append([a1], [0])], [a2]))
    Aend = np.concatenate(([a2[3::-1]], [np.append([0], [a1[2::-1]])]))

    Atot = np.zeros((T, T))
    Atot[:2, :4] = Abeg
    Atot[-2:, -4:] = Aend

    for i in range(2, T - 2):
        Atot[i, i - 2:i + 3] = a3

    if (side == 1):
        t = 2
        trend[:t] = X[:t]

        # Third observation minimization problem is as follows
        r3 = np.array([-2 * smooth, 1 + 4 * smooth, -2 * smooth])

        Atmp = np.concatenate(([a1, r3], [a1[2::-1]]))
        Xtmp = X[:t + 1]

        # Solve the system A*Z = X
        trend[t] = cho_solve(cho_factor(Atmp), Xtmp)[t]

        t += 1

        # Pattern begins with fourth observation
        # Create base A matrix with unique first and last two rows
        # Build recursively larger through time period
        Atmp = np.concatenate(([np.append([a1], [0])], [a2], [a2[3::-1]], [np.append([0], a1[2::-1])]))
        Xtmp = X[:t + 1]

        trend[t] = cho_solve(cho_factor(Atmp), Xtmp)[t]

        while (t < T - 1):
            t += 1

            Atmp = np.concatenate((Atot[:t - 1, :t + 1], np.zeros((2, t + 1))))
            Atmp[t - 1:t + 1, t - 3:t + 1] = Aend

            Xtmp = X[:t + 1]
            trend[t] = cho_solve(cho_factor(Atmp), Xtmp)[t]

    elif (side == 2):
        trend = cho_solve(cho_factor(Atot), X)
    else:
        raise ValueError('Side Parameter should be 1 or 2')

    cyclical = X - trend

    return pw.wrap(cyclical, append='cyclical'), pw.wrap(trend, append='trend')


def hptrend(X, side=2, smooth=1600, freq=''):
    cyclical, trend = hprescott(X, side, smooth, freq)
    return trend

#hp滤波
def hp(eco_data,side=1, smooth=100):
    if type(eco_data) == pd.core.series.Series:
        hp_data = hptrend(eco_data,side, smooth)
    else:
        hp_data = eco_data.apply(hptrend,axis=0,args=(side, smooth))
    return hp_data


def getExponentialDecayWeight(halflife, period):
    '''
    函数名称：getExponentialDecayWeight
    函数功能：获取指数衰减权重，用于多个因子值计算逻辑
    输入参数：halflife：半衰期
            period：窗口期
    输出参数：weightSeries权重
    '''
    alpha = np.power(0.5, 1 / halflife)
    weightSeries = np.arange(period)
    weightSeries = np.power(alpha, weightSeries)
    weightSeries = weightSeries / np.sum(weightSeries)
    weightSeries = weightSeries[::-1]
    return weightSeries


def gettickerlist(start_date, end_date):
    '''
    函数名称：gettickerlist
    函数功能：从BASE数据获取全部股票代码
    输入参数：
    输出参数：
    '''
    base = getAShareBase(start_date, end_date)
    tickerlist = []
    for i in base['S_INFO_WINDCODE'].unique():
        tickerlist.append(i)
    return tickerlist


def gettimelist(start_date, end_date):
    '''
    函数名称：gettimelist
    函数功能：获取日频交易日
    输入参数：
    输出参数：
    '''
    Calender = getAShareCalender()
    Calender.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)
    Calender = Calender[(Calender['TRADE_DT'] >= start_date)&(Calender['TRADE_DT'] <= end_date)]
    datelist = Calender['TRADE_DT'].values.tolist()
    return datelist


# 波动率倒数
def re_vol(series):
    return 1/series.dropna().std() 


def vol_weight_roll(df,window=120):
    rec_vol = df.rolling(window=window,min_periods=window).apply(re_vol) #波动率倒数
    rec_vol.fillna(method='bfill',inplace=True)
    weight = rec_vol.div(rec_vol.sum(axis=1),axis=0)
    data = (df.mul(weight)).sum(axis=1)
    return data

def GetAindexEODPrices_Extra(start_date, end_date,index_code):
    '''
    函数名称：GetAindexEODPrices_Extra
    函数功能：读取国证成长价值指数序列
    输入参数：start_date：开始日期
            end_date：截止日期
    输出参数：国证成长价值的AINDEXEODPRICES表
    '''
    if index_code=='399370.SZ':
        indexprice = pd.read_excel("E:\work\Data_Input_Output\Smart Beta轮动\Input\国证成长_EODPRICE.xlsx")
        #存疑，为什么直接改变列就没法正常导出
        indexprice['TRADE_DT']=[datetime.datetime.strptime(i, '%Y-%m-%d').strftime('%Y%m%d') for i in indexprice['TRADE_DT']]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(int)
        indexprice = indexprice[
            (indexprice['TRADE_DT'] >= int(start_date)) & (indexprice['TRADE_DT'] <= int(end_date))]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(str)
    else:
        indexprice = pd.read_excel("E:\work\Data_Input_Output\Smart Beta轮动\Input\国证价值_EODPRICE.xlsx",index_col=1)
        #存疑，为什么直接改变列就没法正常导出
        indexprice['TRADE_DT']=[datetime.datetime.strptime(i, '%Y-%m-%d').strftime('%Y%m%d') for i in indexprice['TRADE_DT']]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(int)
        indexprice = indexprice[
            (indexprice['TRADE_DT'] >= int(start_date)) & (indexprice['TRADE_DT'] <= int(end_date))]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(str)

    return indexprice

def GetSBIEODPrices(start_date, end_date,index_code):#待修改
    '''
    函数名称：GetAindexEODPrices_Extra
    函数功能：读取国证成长价值指数序列
    输入参数：start_date：开始日期
            end_date：截止日期
    输出参数：国证成长价值的AINDEXEODPRICES表
    '''
    if index_code=='399370.SZ':
        indexprice = pd.read_excel("E:\work\Data_Input_Output\Smart Beta轮动\Input\国证成长_EODPRICE.xlsx")
        #存疑，为什么直接改变列就没法正常导出
        indexprice['TRADE_DT']=[datetime.datetime.strptime(i, '%Y-%m-%d').strftime('%Y%m%d') for i in indexprice['TRADE_DT']]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(int)
        indexprice = indexprice[
            (indexprice['TRADE_DT'] >= int(start_date)) & (indexprice['TRADE_DT'] <= int(end_date))]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(str)
    else:
        indexprice = pd.read_excel("E:\work\Data_Input_Output\Smart Beta轮动\Input\国证价值_EODPRICE.xlsx",index_col=1)
        #存疑，为什么直接改变列就没法正常导出
        indexprice['TRADE_DT']=[datetime.datetime.strptime(i, '%Y-%m-%d').strftime('%Y%m%d') for i in indexprice['TRADE_DT']]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(int)
        indexprice = indexprice[
            (indexprice['TRADE_DT'] >= int(start_date)) & (indexprice['TRADE_DT'] <= int(end_date))]
        indexprice['TRADE_DT']=indexprice['TRADE_DT'].astype(str)

    return indexprice 

def Get_Index_Return(start_date='', end_date='',asset_list=[],frequence='D'):
    '''
    函数名称：Get_Index_Return
    函数功能：读取指数的收益率序列
    输入参数：start_date,end_date：起止日期
            frequence：频率
    输出参数：asset_return：资产收益率
    '''
    # if asset_list==['国证成长','国证价值'] or asset_list==['国证价值','国证成长']:
    #     if frequence=='D':
    #         asset_return_daily = pd.read_csv("E:\work\Data_Input_Output\Smart Beta轮动\Input\资产收益率-国证成长价值.csv", index_col=0, encoding='gbk')[asset_list]
    #     if frequence=='M':
    #         asset_return_daily = pd.read_csv("E:\work\Data_Input_Output\Smart Beta轮动\Input\资产收益率-国证成长价值-月度.csv", index_col=0, encoding='gbk')[asset_list]
    #     asset_return_daily.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in asset_return_daily.index]
    #     asset_return_daily.loc[:, asset_list] = asset_return_daily[asset_list]
    #     asset_return_daily.dropna(inplace=True)
    #     asset_return_daily = asset_return_daily[
    #         (asset_return_daily.index >= datetime.datetime.strptime(start_date, '%Y%m%d').date()) & (asset_return_daily.index <= datetime.datetime.strptime(end_date, '%Y%m%d').date())]
    #     return asset_return_daily
    
    if set(asset_list)<set(['上证180','180成长','180价值','180等权','180基本','180动态','180稳定','180波动','180高贝','180低贝','180红利','黄金','货币市场基金','南华工业品','南华农业品','国债']):
        if frequence=='D':
            asset_return_daily = pd.read_csv(r"E:\work\Data_Input_Output\Smart Beta轮动\Input\rtn_180SB.csv", index_col=0, encoding='gbk')[asset_list]/100
        if frequence=='M':
            asset_return_daily = pd.read_csv(r'E:\work\Data_Input_Output\Smart Beta轮动\Input\rtn_180SB_month.csv', index_col=0, encoding='gbk')[asset_list]/100
        asset_return_daily.index=asset_return_daily.index.astype(str)

        asset_return_daily.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in asset_return_daily.index]
        asset_return_daily.loc[:, asset_list] = asset_return_daily[asset_list]
        asset_return_daily.dropna(inplace=True)
        asset_return_daily = asset_return_daily[
            (asset_return_daily.index >= datetime.datetime.strptime(start_date, '%Y%m%d').date()) & (asset_return_daily.index <= datetime.datetime.strptime(end_date, '%Y%m%d').date())]
        return asset_return_daily
    
    elif set(asset_list)<set(['沪深300','300波动','300红利','300高贝','300低贝','300动态','300稳定','300成长','300价值','300分层','300等权']):
        if frequence=='D':
            asset_return_daily = pd.read_csv(r'E:\work\Data_Input_Output\Smart Beta轮动\Input\rtn_300SB.csv', index_col=0, encoding='gbk')[asset_list]/100
        if frequence=='M':
            asset_return_daily = pd.read_csv(r"E:\work\Data_Input_Output\Smart Beta轮动\Input\rtn_300SB_month.csv", index_col=0, encoding='gbk')[asset_list]/100
        asset_return_daily.index=asset_return_daily.index.astype(str)
        asset_return_daily.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in asset_return_daily.index]
        asset_return_daily.loc[:, asset_list] = asset_return_daily[asset_list]
        asset_return_daily.dropna(inplace=True)
        asset_return_daily = asset_return_daily[
            (asset_return_daily.index >= datetime.datetime.strptime(start_date, '%Y%m%d').date()) & (asset_return_daily.index <= datetime.datetime.strptime(end_date, '%Y%m%d').date())]
        return asset_return_daily
    
    elif set(asset_list)<set(['黄金']):
        if frequence=='M':
            asset_return_daily = pd.read_csv(r"E:\work\Data_Input_Output\Smart Beta轮动\Input\大宗商品.csv", index_col=0, encoding='gbk')[asset_list]/100
        asset_return_daily.index=asset_return_daily.index.astype(str)
        asset_return_daily.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in asset_return_daily.index]
        asset_return_daily.loc[:, asset_list] = asset_return_daily[asset_list]
        asset_return_daily.dropna(inplace=True)
        asset_return_daily = asset_return_daily[
            (asset_return_daily.index >= datetime.datetime.strptime(start_date, '%Y%m%d').date()) & (asset_return_daily.index <= datetime.datetime.strptime(end_date, '%Y%m%d').date())]
        return asset_return_daily
    
    elif asset_list==['无风险利率']:
        asset_return_daily = pd.read_csv(r"E:\work\Data_Input_Output\Smart Beta轮动\Input\资产收益率_无风险利率_万得货币市场基金指数.csv", index_col=0, encoding='gbk')[asset_list]
        asset_return_daily.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in asset_return_daily.index]
        asset_return_daily.loc[:, asset_list] = asset_return_daily[asset_list]
        asset_return_daily.dropna(inplace=True)
        asset_return_daily = asset_return_daily[
            (asset_return_daily.index >= datetime.datetime.strptime(start_date, '%Y%m%d').date()) & (asset_return_daily.index <= datetime.datetime.strptime(end_date, '%Y%m%d').date())]
        return asset_return_daily
    
    else:
        start_date_tmp = (datetime.datetime.strptime(start_date, '%Y%m%d') - timedelta(weeks=20)).date()
        base_pool = {'沪深300':'000300.SH','中证1000':'000852.SH','中证2000':'932000.CSI','中证全指':'000985.CSI','无风险利率':'885009.WI','沪深300全收益':'H00300.CSI','中证2000全收益':'932000CNY010.CSI','国证成长':'399370.SZ','国证价值':'399371.SZ'}
        datelist = Get_DateList(start_date=str(start_date_tmp).replace('-',''), end_date=end_date, frequence=frequence,type='str')
        Result = pd.DataFrame(index=datelist,columns=asset_list)
        for index_code in asset_list:
            AindexEODPrices = getAindexEODPrices(str(start_date_tmp).replace('-',''), end_date,base_pool[index_code])
            AindexEODPrices.set_index(keys=['TRADE_DT'],inplace=True)
            AindexEODPrices = AindexEODPrices.loc[AindexEODPrices.index.isin(datelist),:][['S_DQ_CLOSE']]
            AindexEODPrices['RETURN'] = AindexEODPrices['S_DQ_CLOSE'] / AindexEODPrices['S_DQ_CLOSE'].shift(1) - 1
            Result.loc[AindexEODPrices.index,index_code] = AindexEODPrices['RETURN']
        #按输入的起止日期截取
        Result = Result[(Result.index>=start_date)&(Result.index<=end_date)]
        Result.index = [pd.to_datetime(i).to_pydatetime().date() for i in Result.index]
        return Result


def Get_DateList(start_date, end_date,frequence,type):
    '''
    函数名称：getdatelist
    函数功能：获取不同频率的交易日，例：Get_DateList(start_date=start_date, end_date=end_date, frequence='D', type='datetime.date')
    输入参数：
    输出参数：
    '''
    Calender = getAShareCalender()
    Calender.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)
    Calender = Calender[(Calender['TRADE_DT'] >= start_date)&(Calender['TRADE_DT'] <= end_date)]
    if frequence=='D':
        datelist = Calender['TRADE_DT'].values.tolist()
    elif frequence=='W':
        Calender['weekend'] = [datetime.datetime.strftime(datetime.datetime.strptime(i, "%Y%m%d") + timedelta(7 - datetime.datetime.strptime(i, "%Y%m%d").weekday() - 1), "%Y%m%d") for i in Calender['TRADE_DT']]
        Calender = Calender.groupby('weekend').last().reset_index()
        datelist = Calender['TRADE_DT'].values.tolist()
    elif frequence=='M':
        Calender['month'] = [str(i)[:6] for i in Calender['TRADE_DT']]
        Calender = Calender.groupby('month').last().reset_index()
        datelist = Calender['TRADE_DT'].values.tolist()

    elif frequence=='3M':
        Calender['month'] = [str(i)[:6] for i in Calender['TRADE_DT']]
        Calender = Calender.groupby('month').last().reset_index()

        Calender=Calender.loc[(Calender['month'].str.endswith('3'))|(Calender['month'].str.endswith('6'))|
                              (Calender['month'].str.endswith('9'))|(Calender['month'].str.endswith('12')),:]
        datelist = Calender['TRADE_DT'].values.tolist()
    elif frequence=='HY':
        Calender['month'] = [str(i)[:6] for i in Calender['TRADE_DT']]
        Calender = Calender.groupby('month').last().reset_index()
        Calender=Calender.loc[(Calender['month'].str.endswith('6'))|(Calender['month'].str.endswith('12')),:]
        datelist = Calender['TRADE_DT'].values.tolist()
    if type=='datetime.date':
        datelist = [pd.to_datetime(i).to_pydatetime().date() for i in datelist]
    elif type=='str':
        datelist = datelist
    return datelist

def Get_Rank(group):
    '''
    函数名称：Get_Rank
    函数功能：排序，groupby内部函数
    输入参数：group：待排序值
    输出参数：group：添加了一列序号
    '''
    group['RANK'] = group[['Size']].rank(ascending=False, method='min', axis=0, na_option='keep')
    return group

def Get_Group(group, bins, GroupLabels):
    '''
    函数名称：Get_Group
    函数功能：分组，groupby内部函数
    输入参数：group：待排序值；bins分组边界；GroupLabels：各组标签
    输出参数：group：添加了一列组标签
    '''
    group['GROUP'] = pd.cut(group.RANK, bins.loc[group.TRADE_DT.values[0], 'RANK'], labels=GroupLabels)
    return group


def Cal_Stock_Return(start_date,end_date,frequence):
    if frequence=='W':
        timelist = Get_DateList(start_date=last_week_end(start_date), end_date=end_date, frequence=frequence,type='str')
        EOD = getAShareEODPrices(last_week_end(start_date),end_date)[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']]
    elif frequence=='M':
        timelist = Get_DateList(start_date=last_month_end(start_date), end_date=end_date, frequence=frequence,type='str')
        EOD = getAShareEODPrices(last_month_end(start_date), end_date)[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']]
    elif frequence=='D':
        timelist = Get_DateList(start_date=last_date(start_date), end_date=end_date, frequence=frequence, type='str')
        EOD = getAShareEODPrices(last_date(start_date), end_date)[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']]
    EOD=EOD[EOD['TRADE_DT'].isin(timelist)]
    EOD.reset_index(drop=True,inplace=True)
    def func(group):
        group['RETURN'] = group['S_DQ_ADJCLOSE'] / group['S_DQ_ADJCLOSE'].shift(1) - 1
        return group
    RTN = EOD.groupby('S_INFO_WINDCODE').apply(func).reset_index(drop=True)
    RTN = RTN[['S_INFO_WINDCODE','TRADE_DT','RETURN']]
    RTN = RTN[(RTN['TRADE_DT']>=start_date)&(RTN['TRADE_DT']<=end_date)]
    return RTN


def CalDailyIndexWeight(IndexWeight,datelist,Stock_Rtn):
    '''
    根据指数月频成分股权重，推算出日频成分股权重
    '''
    Calendar = pd.DataFrame(datelist,columns=['TRADE_DT'])
    Calendar['MONTH'] = [i[:6] for i in Calendar['TRADE_DT']]
    monthlist = Calendar.groupby('MONTH').last()['TRADE_DT'].tolist()
    IndexWeight=IndexWeight[IndexWeight['TRADE_DT'].isin(monthlist)]

    DailyIndexWeight=pd.DataFrame(columns=['INDEX_CODE','TRADE_DT','S_INFO_WINDCODE','WEIGHT'])
    period_list = sorted(IndexWeight['TRADE_DT'].unique())
    #为确保获取到所需完整时间段，在period_list末尾附加截止时间
    period_list.append(next_date(datelist[-1]))
    for period, next_period in itertools.islice(zip(period_list, period_list[1:]), 0, None, 1):
        #将月频权重pivot成矩阵方便计算
        Weight_period = IndexWeight[IndexWeight['TRADE_DT']==period]
        Weight_period = pd.pivot(Weight_period,index='TRADE_DT',columns='S_INFO_WINDCODE',values='WEIGHT')
        date_list_period = [i for i in datelist if i>=period and i<next_period]
        stock_list_period = Weight_period.columns.tolist()
        #将月频权重为起点，推算为日频权重
        Weight_period_daily = pd.DataFrame(index=date_list_period,columns=stock_list_period)
        Weight_period_daily.loc[period] = Weight_period.loc[period]
        #读取所需的股票收益率数据，同样pivot成矩阵
        RTN_tmp = Stock_Rtn[(Stock_Rtn['S_INFO_WINDCODE'].isin(stock_list_period))&(Stock_Rtn['TRADE_DT'].isin(date_list_period))]
        RTN_tmp = pd.pivot(RTN_tmp,index='TRADE_DT',columns='S_INFO_WINDCODE',values='RETURN')
        #通过收益率累乘得到每天权重
        Weight_period_daily.iloc[1:] = Weight_period_daily.iloc[0]*np.cumprod(RTN_tmp.iloc[1:] + 1)
        #归一化
        Weight_period_daily = Weight_period_daily.div(Weight_period_daily.sum(axis=1), axis=0)
        #转为面板格式进行保存
        Weight_period_daily = Weight_period_daily.stack().reset_index()
        Weight_period_daily.columns = ['TRADE_DT', 'S_INFO_WINDCODE', 'WEIGHT']
        Weight_period_daily['INDEX_CODE'] = IndexWeight['INDEX_CODE'].values[0]
        DailyIndexWeight = pd.concat([DailyIndexWeight,Weight_period_daily],ignore_index=True)

    return DailyIndexWeight

def GetIndexWeight(start_date, end_date,asset_index,input_path):
    '''
    读取指数成分股权重,并转化为日频
    '''
    # 多读一些原始权重，在转为日频后再进行截断
    start_date_lastmonth = last_month_end(start_date)
    if asset_index == '932000.CSI':
        # 读取中证2000的成分股数据
        index_stock_weight1 = pd.read_csv(input_path+'zscfgday2000.csv')
        index_stock_weight1 = index_stock_weight1[index_stock_weight1['index'] == asset_index]
        index_stock_weight1['trade_dt'] = [i.replace('-', '') for i in index_stock_weight1['trade_dt']]
        index_stock_weight1 = index_stock_weight1[index_stock_weight1['trade_dt'] < '20230811']
        index_stock_weight1.rename(columns={'i_weight': 'WEIGHT', 'index': 'INDEX_CODE', 's_info_windcode': 'S_INFO_WINDCODE', 'trade_dt': 'TRADE_DT'}, inplace=True)

        index_stock_weight2 = getAShareIndexWeight('20230811', end_date, index_code=asset_index)
        index_stock_weight2.rename(columns={'S_INFO_WINDCODE': 'INDEX_CODE', 'S_CON_WINDCODE': 'S_INFO_WINDCODE', 'I_WEIGHT': 'WEIGHT'}, inplace=True)
        index_stock_weight = pd.concat([index_stock_weight1, index_stock_weight2], axis=0)
        index_stock_weight = index_stock_weight[(index_stock_weight['TRADE_DT'] >= start_date_lastmonth)&(index_stock_weight['TRADE_DT'] <= end_date)]
        index_stock_weight['WEIGHT'] = index_stock_weight['WEIGHT'] / 100
        
    elif asset_index == '000985.CSI':
        index_stock_weight1 = pd.read_csv(input_path+'中证全指成分股.csv')
        index_stock_weight1=index_stock_weight1.loc[:,['S_INFO_WINDCODE','S_CON_WINDCODE','TRADE_DT','I_WEIGHT']]
        index_stock_weight1.rename(columns={'S_INFO_WINDCODE': 'INDEX_CODE', 'S_CON_WINDCODE': 'S_INFO_WINDCODE', 'I_WEIGHT': 'WEIGHT'}, inplace=True)
        index_stock_weight1['TRADE_DT']=index_stock_weight1['TRADE_DT'].astype(int)
        index_stock_weight = index_stock_weight1[(index_stock_weight1['TRADE_DT'] >= int(start_date_lastmonth))&(index_stock_weight1['TRADE_DT'] <= int(end_date))]
        index_stock_weight['TRADE_DT']=index_stock_weight['TRADE_DT'].astype(str)
        index_stock_weight['WEIGHT'] = index_stock_weight['WEIGHT'] / 100

    elif asset_index == '399370.SZ':
        # index_stock_weight1 = pd.read_csv(input_path+'gzczcfg.csv')
        index_stock_weight1 = pd.read_csv(input_path+'国证成长成分股_月频.csv')
        index_stock_weight1=index_stock_weight1.iloc[:,1:]
        index_stock_weight1['TRADE_DT']=[datetime.datetime.strptime(i, '%Y-%m-%d').strftime('%Y%m%d') for i in index_stock_weight1['TRADE_DT']]
        index_stock_weight1['TRADE_DT']=index_stock_weight1['TRADE_DT'].astype(int)
        index_stock_weight = index_stock_weight1[(index_stock_weight1['TRADE_DT'] >= int(start_date_lastmonth))&(index_stock_weight1['TRADE_DT'] <= int(end_date))]
        index_stock_weight['TRADE_DT']=index_stock_weight['TRADE_DT'].astype(str)

    elif asset_index == '399371.SZ':
        index_stock_weight1 = pd.read_csv(input_path+'国证价值成分股_月频.csv')
        index_stock_weight1=index_stock_weight1.iloc[:,1:]
        index_stock_weight1['TRADE_DT']=[datetime.datetime.strptime(i, '%Y-%m-%d').strftime('%Y%m%d') for i in index_stock_weight1['TRADE_DT']]
        index_stock_weight1['TRADE_DT']=index_stock_weight1['TRADE_DT'].astype(int)
        index_stock_weight = index_stock_weight1[(index_stock_weight1['TRADE_DT'] >= int(start_date_lastmonth))&(index_stock_weight1['TRADE_DT'] <= int(end_date))]
        index_stock_weight['TRADE_DT']=index_stock_weight['TRADE_DT'].astype(str)

    elif asset_index == '000135.SH':

        index_stock_weight1 = pd.read_csv(input_path+'180高贝成分股.csv')
        index_stock_weight1=index_stock_weight1.iloc[:,1:]
        index_stock_weight1['TRADE_DT']=index_stock_weight1['TRADE_DT'].astype(str)
        index_stock_weight1['TRADE_DT']=[datetime.datetime.strptime(i, '%Y%m%d').strftime('%Y%m%d') for i in index_stock_weight1['TRADE_DT']]
        index_stock_weight1['TRADE_DT']=index_stock_weight1['TRADE_DT'].astype(int)
        index_stock_weight = index_stock_weight1[(index_stock_weight1['TRADE_DT'] >= int(start_date_lastmonth))&(index_stock_weight1['TRADE_DT'] <= int(end_date))]
        index_stock_weight['TRADE_DT']=index_stock_weight['TRADE_DT'].astype(str)

    elif asset_index == '000129.SH':

        index_stock_weight1 = pd.read_csv(input_path+'180波动成分股.csv')
        index_stock_weight1=index_stock_weight1.iloc[:,1:]
        index_stock_weight1['TRADE_DT']=index_stock_weight1['TRADE_DT'].astype(str)
        index_stock_weight1['TRADE_DT']=[datetime.datetime.strptime(i, '%Y%m%d').strftime('%Y%m%d') for i in index_stock_weight1['TRADE_DT']]
        index_stock_weight1['TRADE_DT']=index_stock_weight1['TRADE_DT'].astype(int)
        index_stock_weight = index_stock_weight1[(index_stock_weight1['TRADE_DT'] >= int(start_date_lastmonth))&(index_stock_weight1['TRADE_DT'] <= int(end_date))]
        index_stock_weight['TRADE_DT']=index_stock_weight['TRADE_DT'].astype(str)

    else:
        index_stock_weight = getAShareIndexWeight(start_date_lastmonth, end_date, index_code=asset_index)
        index_stock_weight.rename(columns={'S_INFO_WINDCODE': 'INDEX_CODE', 'S_CON_WINDCODE': 'S_INFO_WINDCODE', 'I_WEIGHT': 'WEIGHT'}, inplace=True)
        index_stock_weight['WEIGHT'] = index_stock_weight['WEIGHT'] / 100
    
    print(index_stock_weight)
    

    datelist = Get_DateList(start_date=start_date_lastmonth, end_date=end_date,frequence='D',type='str')
    # 获取股票收益率
    Stock_Rtn = Cal_Stock_Return(start_date=start_date_lastmonth,end_date=end_date,frequence='D')
    # 根据股票收益变化进行日度再平衡
    index_stock_weight = CalDailyIndexWeight(index_stock_weight, datelist, Stock_Rtn)
    # 截取最终时间区间
    datelist_final = Get_DateList(start_date=start_date, end_date=end_date,frequence='D',type='str')
    index_stock_weight = index_stock_weight[index_stock_weight['TRADE_DT'].isin(datelist_final)]
    index_stock_weight = index_stock_weight[['INDEX_CODE', 'TRADE_DT', 'S_INFO_WINDCODE', 'WEIGHT']].reset_index(drop=True)
    return index_stock_weight


def Cal_Stratify(data, GroupNum):
    '''
    函数名称：Cal_Stratify
    函数功能：获取因子值分组
    输入参数：data：因子值；GroupNum：组数
    输出参数：Result：因子分组
    '''
    # 按照ep分成10组，分别计算收益率
    data_Rank = data.groupby('TRADE_DT').apply(Get_Rank).reset_index(drop=True)
    length = data_Rank.groupby('TRADE_DT')[['RANK']].count()  # 计算每月共有多少只股票
    # bins记录分组的边界，是分组的依据（左闭右开）
    bins = pd.DataFrame(length['RANK'].apply(lambda _: [math.ceil(_ / GroupNum * i) for i in range(GroupNum + 1)]))
    GroupLabels = [str(i) for i in range(1, GroupNum + 1)]  # 为每组标号
    data_Rank = data_Rank.groupby('TRADE_DT').apply(Get_Group, bins, GroupLabels).reset_index(drop=True)
    Result = data_Rank[['TRADE_DT', 'S_INFO_WINDCODE', 'GROUP']]
    return Result


def last_month_end(date):
    '''
    函数名称：getmonthlist
    函数功能：获取上个月末的交易日（若输入为月末，则原样输出）
    输入参数：
    输出参数：
    '''
    Calender = getAShareCalender()
    Calender.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)

    Calender['month'] = [str(i)[:6] for i in Calender['TRADE_DT']]
    Calender = Calender.groupby('month').last().reset_index()
    Calender.drop(columns='month',inplace=True)
    monthlist = Calender['TRADE_DT'].values.tolist()
    if date in monthlist:
        return date
    else:
        result = [i for i in monthlist if i[:6]<date[:6]][-1]
        return result

def last_week_end(date):
    '''
    函数名称：getmonthlist
    函数功能：获取上周最后一个交易日
    输入参数：
    输出参数：
    '''
    Calender = getAShareCalender()
    Calender.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)
    Calender['weekend'] = [datetime.datetime.strftime(datetime.datetime.strptime(i, "%Y%m%d") + timedelta(7 - datetime.datetime.strptime(i, "%Y%m%d").weekday() - 1), "%Y%m%d") for i in Calender['TRADE_DT']]
    Calender = Calender.groupby('weekend').last().reset_index()
    datelist = Calender['TRADE_DT'].values.tolist()
    result = [i for i in datelist if i > date][-1]
    return result

def last_date(date):
    '''
    函数名称：last_date
    函数功能：获取上个交易日
    输入参数：
    输出参数：
    '''
    Calender = getAShareCalender()
    Calender.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)
    datelist = Calender['TRADE_DT'].values.tolist()
    result = [i for i in datelist if i < date][-1]

    return result


def next_date(date):
    '''
    函数名称：next_date
    函数功能：获取下个交易日
    输入参数：
    输出参数：
    '''
    Calender = getAShareCalender()
    Calender.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)
    datelist = Calender['TRADE_DT'].values.tolist()
    result = datelist[datelist.index(date)+1]
    return result



def IndustriesTimeSeries(start_date, end_date,IndustriesClass,AShareCalendar):
    '''
    生成各行业成分股的时序数据
    '''
    base = getAShareBase(start_date, end_date)
    IndustriesClass=IndustriesClass.sort_values(by=['S_INFO_WINDCODE','ENTRY_DT']).reset_index(drop=True)
    # 第一步将行业变更信息表中的ENTRY_DT转为下一个交易日（这么处理为了提高运行效率）
    # 获取日历日和交易日对应关系
    dfDate = pd.DataFrame(pd.date_range('20000101', datetime.datetime.strftime(date.today(), '%Y%m%d'), freq='D', name='ENTRY_DT'))
    dfDate['ENTRY_DT'] = dfDate['ENTRY_DT'].apply(lambda x: x.strftime('%Y%m%d'))
    AShareCalendar.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)
    dfDate = pd.merge(dfDate,AShareCalendar, left_on='ENTRY_DT', right_on='TRADE_DT', how='left')
    dfDate = dfDate.fillna(method='bfill')
    # ENTRY_DT_ALTER代表ENTRY_DT下一个交易日
    IndustriesClass = pd.merge(IndustriesClass,dfDate, on='ENTRY_DT', how='left')
    IndustriesClass.rename(columns={'TRADE_DT': 'ENTRY_DT_ALTER'}, inplace=True)
    # 接下来将行业变更信息转为交易日上的时序数据
    Result = pd.DataFrame(columns=['S_INFO_WINDCODE','TRADE_DT','INDUSTRIESNAME'])
    # 根据ENTRY_DT_ALTER设置行业值
    for stock in IndustriesClass['S_INFO_WINDCODE'].unique():
            IndustriesClassStock = IndustriesClass[IndustriesClass['S_INFO_WINDCODE']==stock]
            for entry_dt_alter in IndustriesClassStock['ENTRY_DT_ALTER'].values:
                IndustriesClassStockPeriod = IndustriesClassStock[IndustriesClassStock['ENTRY_DT_ALTER']==entry_dt_alter]
                Result.loc[len(Result)] = [stock,entry_dt_alter,IndustriesClassStockPeriod['INDUSTRIESNAME'].values[0]]

    #与base合并，得到完整时序行业信息
    #为了保留在base日期之前的行业所属信息，比方说2003年的entry_dt，先用outer进行merge，ffill填补nan后再根据base合并
    Result = pd.merge(Result, base, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='outer')
    Result.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], inplace=True)
    # 向前填充nan
    Result = Result.groupby('S_INFO_WINDCODE').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
    Result = pd.merge(Result, base, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='right')
    return Result





def FloatCap_Weight_df(BASE,EOD_DI):
    '''
    函数名称：FreeFloatCap_Weight_matrix
    函数功能：获取流通市值，计算权重，作为因子标准化权重
    输入参数：EOD_DI：ASHAREEODDERIVATIVEINDICATOR数据表
    输出参数：weight：流通市值权重
    '''
    # 对自由流通市值在横截面上归一化得到权重
    EOD_DI = EOD_DI[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_MV']]
    data=pd.merge(BASE,EOD_DI,on=['S_INFO_WINDCODE', 'TRADE_DT'],how='left')
    data['WEIGHT'] = np.nan
    def func(group):
        group['WEIGHT'] = group['S_DQ_MV'].dropna() / np.nansum(group['S_DQ_MV'])  # nansum把非零值视为0并求和
        return group
    data=data.groupby('TRADE_DT').apply(func).reset_index(drop=True)
    data = data[['S_INFO_WINDCODE', 'TRADE_DT','WEIGHT']]
    # 按照TRADE_DT、S_INFO_WINDCODE排序
    data.sort_values(by=['TRADE_DT','S_INFO_WINDCODE'],inplace=True)
    return data


def process_factor(factor, weight_df, I, mad,factor_name):
    '''
    函数名称：process_factor
    函数功能：因子预处理函数，包括去极值(可选)、标准化、填补缺失值三步
    输入参数：factor：因子值
            weight_matrix：标准化权重
            I：股票所属行业
            mad：是否去极值
    输出参数：factor_fillna：填补完后的因子值
    '''
    factor = pd.merge(weight_df,factor,how='left')
    if mad:
        # 去极值、标准化
        factor_stand = standardize_mad_factor(factor,factor_name)
    else:
        # 不去极值、只标准化
        factor_stand = standardize_factor(factor,factor_name)
    # 填补缺失值（行业中值）
    factor_fillna = fillna_factor(factor_stand, I,factor_name)
    # 按照TRADE_DT、S_INFO_WINDCODE排序
    factor_fillna.sort_values(by=['TRADE_DT','S_INFO_WINDCODE'],inplace=True)
    return factor_fillna


def standardize_factor(factor,factor_name):
    '''
    函数名称：standardize_mad_factor
    函数功能：对因子值进行标准化(有些因子不适用去极值，只需要标准化)
    输入参数：factor：因子值
            weight_matrix：标准化权重
    输出参数：stard_factor_matrix：标准化后的因子值
    '''
    # 定义标准化后的因子值
    factor[factor_name+'_stand'] = np.nan
    def func(group):
        #标准化
        fa = group[factor_name]
        weight = group['WEIGHT']
        if np.nanstd(fa.dropna()) <= 1e-8:#std=0则说明只有一个有效数据，不进行标准化，写成<=1e-8是因为有计算误差，不会完全等于0
            group[factor_name+'_stand'] = group[factor_name]
            return group
        stand_fa = (fa.dropna() - np.nansum(weight.dropna() * fa.dropna())) / np.nanstd(fa.dropna())
        group[factor_name+'_stand'] = stand_fa
        return group
    factor = factor.groupby('TRADE_DT').apply(func).reset_index(drop=True)
    factor = factor[['S_INFO_WINDCODE','TRADE_DT',factor_name+'_stand']]
    factor.rename(columns={factor_name+'_stand':factor_name},inplace=True)
    return factor

def mad(fa, n=5):
    '''
    函数名称：mad
    函数功能：中位数去极值函数,factor:以股票code为index，因子值为value的Series,std为几倍的标准差,输出Series"
    输入参数：factor：因子值
            n：阈值倍数
    输出参数：
    '''
    r = fa.dropna().copy()
    median = r.quantile(0.5)
    new_median = abs(r - median).quantile(0.5)
    up = median + n * new_median
    down = median - n * new_median
    return np.clip(r, down, up)


def standardize_mad_factor(factor,factor_name):
    '''
    函数名称：standardize_mad_factor
    函数功能：对因子值进行去极值和标准化
    输入参数：factor：因子值
            weight_matrix：标准化权重
    输出参数：stard_factor_matrix：去极值和标准化后的因子值
    '''
    # 定义标准化后的因子值
    factor[factor_name+'_mad'] = np.nan
    factor[factor_name+'_mad_stand'] = np.nan
    #去极值，标准化
    def func(group):
        #去极值
        fa = group[factor_name]
        # 如果某一列都是NAN,跳过
        if len(fa.dropna()) == 0:
            return group
        mad_fa = mad(fa, n=5)
        group[factor_name+'_mad'] = mad_fa
        #标准化
        weight = group['WEIGHT']
        if np.nanstd(mad_fa.dropna()) <= 1e-8:#std=0则说明只有一个有效数据，不进行标准化，写成<=1e-8是因为有计算误差，不会完全等于0
            group[factor_name+'_mad_stand'] = group[factor_name+'_mad']
            return group
        stand_fa = (group[factor_name+'_mad'].dropna() - np.nansum(weight.dropna() * group[factor_name+'_mad'].dropna())) / np.nanstd(group[factor_name+'_mad'].dropna())
        group[factor_name+'_mad_stand'] = stand_fa
        return group
    factor=factor.groupby('TRADE_DT').apply(func).reset_index(drop=True)
    factor = factor[['S_INFO_WINDCODE','TRADE_DT',factor_name+'_mad_stand']]
    factor.rename(columns={factor_name+'_mad_stand':factor_name},inplace=True)
    return factor


def fillna_factor(factor, I,factor_name):
    '''
    函数名称：fillna_factor
    函数功能：行业中值填补因子缺失值
    输入参数：factor：因子值
            I：股票所属行业
    输出参数：factor_fillna：填补完后的因子值
    '''
    factor_df = pd.merge(factor, I, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')

    def fill(df):
        industry_names = ['交通运输', '传媒', '农林牧渔', '医药', '商贸零售', '国防军工',
                          '基础化工', '家电', '建材', '建筑', '房地产', '有色金属', '机械', '汽车', '消费者服务', '煤炭',
                          '电力及公用事业', '电力设备及新能源', '电子', '石油石化', '纺织服装', '综合', '综合金融', '计算机',
                          '轻工制造', '通信', '钢铁', '银行', '非银行金融', '食品饮料']
        for ind_name in industry_names:
            df.loc[df['INDUSTRIESNAME'] == ind_name, factor_name] = df.loc[df['INDUSTRIESNAME'] == ind_name, factor_name].fillna(df.loc[df['INDUSTRIESNAME'] == ind_name, factor_name].median(axis=0))
        return df

    factor_fillna_df = factor_df.groupby('TRADE_DT').apply(fill).reset_index(drop=True)
    return factor_fillna_df[['S_INFO_WINDCODE', 'TRADE_DT',factor_name]]


def Cal_Quantile(factor,rolling_year):
    '''
    函数名称：Cal_Quantile
    函数功能：计算滚动分位数
    输入参数：factor：因子值
            rolling_year：滚动年数
    输出参数：factor_quantile：滚动分位数
    '''
    # 计算滚动分位数
    factor_quantile = pd.DataFrame(columns=factor.columns,index=factor.index)
    for i in range(len(factor.index)):
        date = factor.index[i]
        date_pre = date - timedelta(days=rolling_year*365)
        # 如果能取到足够的长度
        if date_pre >= factor.index[0]:
            tmp = factor.loc[date_pre:date]
            for factor_name in factor.columns:
                #计算分位点
                factor_quantile.loc[date,factor_name] = stats.percentileofscore(tmp[factor_name], tmp[factor_name].values[-1])/100
    factor_quantile.dropna(inplace=True)
    return factor_quantile

def last_monthend(date):
    '''
    函数名称：getmonthlist
    函数功能：获取上个月末的交易日（若输入为月末，则原样输出）
    输入参数：
    输出参数：
    '''
    Calender =getAShareCalender()
    Calender.rename(columns={'TRADE_DAYS':'TRADE_DT'},inplace=True)

    Calender['month'] = [str(i)[:6] for i in Calender['TRADE_DT']]
    Calender = Calender.groupby('month').last().reset_index()
    Calender.drop(columns='month',inplace=True)
    monthlist = Calender['TRADE_DT'].values.tolist()
    if date in monthlist:
        return date
    else:
        result = [i for i in monthlist if i[:6]<date[:6]][-1]
        return result
def CreateFolder(base_path, main_folder, subfolders):
    """
    在基础路径下创建主文件夹及其子文件夹结构
    
    参数:
    base_path -- 基础目录路径 (例如: r"D:\Code\自研策略\打板策略\Tools\Historydata\Factor")
    main_folder -- 要创建的主文件夹名称 (例如: "factor1")
    subfolders -- 主文件夹内部的子文件夹列表 (例如: ["feature", "文件夹1", "文件夹2"])
    
    返回:
    bool -- 是否成功创建所有目录
    str -- 状态消息
    """
    try:
        # 构造主文件夹完整路径
        main_path = os.path.join(base_path, main_folder)
        
        # 创建主文件夹（如果不存在）
        os.makedirs(main_path, exist_ok=True)
        
        # 创建所有子文件夹
        for folder in subfolders:
            dir_path = os.path.join(main_path, folder)
            os.makedirs(dir_path, exist_ok=True)
        
        return True, f"目录结构创建成功: {main_path}"

    except PermissionError:
        return False, "错误：没有创建目录的权限，请检查磁盘权限"
    except OSError as e:
        return False, f"系统错误：{str(e)}"
    except Exception as e:
        return False, f"未知错误：{str(e)}"

if __name__ == "__main__":
    asset_col =  ['沪深300','中证500','恒生指数','标普500']
    eco_col = ['工业增加值','PMI','固定资产投资','社消同比','进出口','CPI','PPI']
    start_date = '2005-01-01'
    end_date='2010-01-31'