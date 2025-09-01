
import sys
import os

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import tushare as ts
from Tools.Tushare_apidata import *
warnings.filterwarnings('ignore')
ts.set_token('1163ddac783c82274f41e334870950f57ca40a558f4f29518ea249df')
pro = ts.pro_api('1163ddac783c82274f41e334870950f57ca40a558f4f29518ea249df')



class Test:
    def __init__(self,start_date='20200101',end_date='20210131',frequence='D'):
        self.start_date = start_date
        self.end_date = end_date
        self.frequence = frequence

    def ForwardTest(self,date,fac_name,label_name,fac,label,pool_name):
        test_data=pd.merge(fac,label,on='ts_code')
        test_data.to_feather(rf'Factor/{pool_name}/{fac_name}/{label_name}/{fac_name}_{label_name}_{date}.fea')
        ic=test_data['fac'].corr(test_data['label'])
        return ic
    
    def FactorTest(self,fac_name,label_name,start_date,end_date,pool_name):
        df=pd.read_csv(rf'Result/{fac_name}/{pool_name}_{fac_name}_{label_name}.csv',index_col=0)
        df=df.loc[(df['date']>start_date)&(df['date']<end_date),:]
        ic=(df['ic']).mean()
        ir=(df['ic']).std()
        icir=ic/ir
        result=pd.DataFrame([[fac_name,label_name,ic,ir,icir]],columns=['fac_name','label_name','ic','ir','icir'])
        return result


    
    def CalWinPct(self,date,Stocklist,label,benchmark,loss_benchmark):
        # label=label.sort_values(by='label',ascending=False).reset_index(drop=True)
        # print(label)
        data=label.loc[(label['ts_code'].isin(Stocklist))&(label['label']>benchmark),:].reset_index(drop=True)
        windata=data['ts_code'].to_list()
        Stocknum=len(Stocklist)
        Win=len(windata)
        WinPct=Win/Stocknum

        # 止损标准
        label.loc[label['label'] < loss_benchmark,'label'] = loss_benchmark

        AvgRtn=label.loc[label['ts_code'].isin(Stocklist),'label'].mean()
        # # 相对胜率和收益的计算
        # data2=label.loc[(label['label']>benchmark),:].reset_index(drop=True)
        # windataall=data2['ts_code'].to_list()
        # WinAll=len(windataall)
        # ReWinPct=Win/WinAll
        # AvgRtn2=label.loc[:,'label'].mean()
        # Rertn=AvgRtn/AvgRtn2
        # Result=pd.DataFrame([[date,WinPct,AvgRtn,ReWinPct,Rertn]],columns=['date','winpct','avgrtn','rewinpct','rertn'])
        Result=pd.DataFrame([[date,WinPct,AvgRtn]],columns=['date','winpct','avgrtn'])
        return Result