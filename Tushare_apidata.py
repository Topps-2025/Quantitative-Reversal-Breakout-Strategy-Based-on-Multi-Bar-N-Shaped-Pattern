# -*- coding: utf-8 -*-
"""
功能：读取tushare接口数据
@author: Topps
"""

import sys
import os

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import tushare as ts

warnings.filterwarnings('ignore')
ts.set_token('1163ddac783c82274f41e334870950f57ca40a558f4f29518ea249df')
pro = ts.pro_api('1163ddac783c82274f41e334870950f57ca40a558f4f29518ea249df')


class StockData:
    def __init__(self,start_date='20200101',end_date='20210131',frequence='D'):
        self.start_date = start_date
        self.end_date = end_date
        self.frequence = frequence

    
    def AllStock(self,exchange):
        '''查询当前所有正常上市的企业'''
        data = pro.stock_basic(exchange=exchange, list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        data.to_feather('AllStock.fea')
        return data


    def AshareCalendarAll(self,exchange,start_date,end_date):
        '''
        查询交易日历
        '''
        df = pro.trade_cal(exchange=exchange,start_date=start_date, end_date=end_date, is_open='1')
        df.to_feather('AShareCalendar.fea')
        return df
    
    def AshareCalendar(self,exchange,start_date,end_date):
        '''
        查询交易日历
        '''
        df = pro.trade_cal(exchange=exchange,start_date=start_date, end_date=end_date, is_open='1')
        return df
    
    def Nowtick(self,ts_code,src):
        if src=='dc':
            df = ts.realtime_quote(ts_code=ts_code,src='dc')
        else:
            df = ts.realtime_quote(ts_code=ts_code)
        return df
    
    def NowTrans(self,ts_code,src,date):
        if src=='dc':
            df = ts.realtime_tick(ts_code=ts_code,src='dc')
        else:
            df = ts.realtime_tick(ts_code=ts_code).reset_index(drop=True)
        print(df)
        df.to_feather(rf'{date}_Trans.fea')
        return df
    
    def DailyKlineAll(self,ts_code,start_date,end_date):
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        df=df.sort_values(by=['ts_code','trade_date']).reset_index(drop=True)
        df.to_feather('DailyKline.fea')
        return df
    
    def DailyKline(self,ts_code,start_date,end_date):
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df=df.sort_values(by=['ts_code','trade_date']).reset_index(drop=True)
        return df
    
    def Datapro(self,ts_code,adj,freq,start_date,end_date):
        df=ts.pro_bar(ts_code=ts_code,adj=adj,freq=freq,start_date=start_date, end_date=end_date)
        return df
    

    def DTlist(self,trade_date):
        '''返回龙虎榜每日交易明细'''
        data=pro.top_list(trade_date=trade_date)
        return data

    def LimitUD(self,trade_date,limit_type):
        data = pro.limit_list_d(trade_date=trade_date, limit_type=limit_type, fields='ts_code,trade_date,industry,name,close,pct_chg,amount,limit_amount,fd_amount,open_times,up_stat,limit_times,last_time')
        return data
    

    def MoneyFlow(self,ts_code,start_date,end_date):
        data=pro.moneyflow(ts_code=ts_code,start_date=start_date, end_date=end_date)
        return data
    

    def Concept_data(self,trade_date):
        data=pro.kpl_concept(trade_date=trade_date)
        data=data.rename(columns={'ts_code':'concept_code'})
        return data
    
    def Concept_cons(self,trade_date):
        data=pro.kpl_concept_cons(trade_date=trade_date)
        data=data.rename(columns={'ts_code':'concept_code','con_code':'ts_code'})
        return data
    
    def DailyChip(self,ts_code,start_date,end_date):
        data=pro.cyq_perf(ts_code=ts_code,start_date=start_date, end_date=end_date)
        return data
    
    def StockDaily(self,ts_code,trade_date):
        df = pro.stk_premarket(trade_date=trade_date)
        df=df.loc[df['ts_code'].isin(ts_code)]
        df['MV']=(df['pre_close']*df['total_share'])/10000
        df['Float_MV']=df['pre_close']*df['float_share']/10000
        return df
    
    def StockDailyBase(self,ts_code,start_date,end_date):
        df = pro.daily_basic(ts_code=ts_code,start_date=start_date,end_date=end_date, fields='ts_code,close,total_mv,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,circ_mv')
        return df   

    def Margin(self,ts_code,start_date,end_date):
        df = pro.margin_detail(start_date=start_date,end_date=end_date)
        return df
    
    def TechFactor_Tushare(self,trade_date):
        df=pro.stk_factor_pro(trade_date=trade_date)
        return df
    # 计算用函数
    def Callabel(self,label_name,base_poolstr,date,start_date,end_date,pool_name):
        labelall=StockData.DailyKline(self,ts_code=base_poolstr,start_date=start_date,end_date=end_date)
        if label_name=='T12open':
            label=labelall.loc[:,['ts_code','trade_date','open']]
            # 去除算不出label1的部分       
            mask=label['ts_code'].value_counts()
            mask=mask[mask==2].index
            label=label.loc[label['ts_code'].isin(mask),:]
            label['label']=(label['open']/label['open'].shift(1))-1
            label['trade_date']=label['trade_date'].astype(int)
            labeltest=label.loc[label['trade_date']==end_date,['ts_code','label']].drop_duplicates().reset_index(drop=True)
            
        elif label_name=='T1opentoT2close':
            label=labelall.loc[:,['ts_code','trade_date','open','close']]
            # 去除算不出label2的部分
            mask=label['ts_code'].value_counts()
            mask=mask[mask==2].index
            label=label.loc[label['ts_code'].isin(mask),:]
            label['label']=(label['close']/label['open'].shift(1))-1
            label['trade_date']=label['trade_date'].astype(int)
            labeltest=label.loc[label['trade_date']==end_date,['ts_code','label']].drop_duplicates().reset_index(drop=True)

        elif label_name=='T1closeT2open':
            label=labelall.loc[:,['ts_code','trade_date','open','close']]
            # 去除算不出label2的部分
            mask=label['ts_code'].value_counts()
            mask=mask[mask==2].index
            label=label.loc[label['ts_code'].isin(mask),:]
            label['label']=(label['open']/label['close'].shift(1))-1
            label['trade_date']=label['trade_date'].astype(int)
            labeltest=label.loc[label['trade_date']==end_date,['ts_code','label']].drop_duplicates().reset_index(drop=True)

        elif label_name=='T12close':
            label=labelall.loc[:,['ts_code','trade_date','close']]
            # 去除算不出label1的部分       
            mask=label['ts_code'].value_counts()
            mask=mask[mask==2].index
            label=label.loc[label['ts_code'].isin(mask),:]
            label['label']=(label['close']/label['close'].shift(1))-1
            label['trade_date']=label['trade_date'].astype(int)
            labeltest=label.loc[label['trade_date']==end_date,['ts_code','label']].drop_duplicates().reset_index(drop=True)

        elif label_name=='T15close':
            labelall['trade_date']=labelall['trade_date'].astype(int)
            label=labelall.loc[(labelall['trade_date']==int(start_date))|(labelall['trade_date']==int(end_date)),['ts_code','trade_date','close']]
            # 去除算不出label的部分       
            mask=label['ts_code'].value_counts()
            mask=mask[mask==2].index
            label=label.loc[label['ts_code'].isin(mask),:]
            label['label']=(label['close']/label['close'].shift(1))-1
            label['trade_date']=label['trade_date'].astype(int)
            labeltest=label.loc[label['trade_date']==end_date,['ts_code','label']].drop_duplicates().reset_index(drop=True)
        folder_path=rf'Label/{pool_name}/{label_name}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        labeltest.to_feather(rf'Label/{pool_name}/{label_name}/{label_name}_{date}.fea')
        return labeltest
    
    def Stockpool(self,pool_name,date,AllDateList):
        index=AllDateList.index(date)
        pre_date=AllDateList[index-1]
        pre_date2=AllDateList[index-2]
        if pool_name=='pool1':
            pool1=StockData.LimitUD(self,trade_date=date,limit_type='U')
            pool1=pool1.loc[(pool1['up_stat']=='1/1')|(pool1['up_stat']=='2/2')]['ts_code'].to_list()
            pool2=StockData.LimitUD(self,trade_date=pre_date,limit_type='U')
            pool2=pool2.loc[(pool2['up_stat']=='1/1')|(pool2['up_stat']=='2/2')]['ts_code'].to_list()
            finalpool=list(set(pool1)|set(pool2))
            base_pool=pd.DataFrame(finalpool,columns=['ts_code'])
            base_poolstr=','.join(finalpool)

        elif pool_name=='pool2':
            pool1=StockData.LimitUD(self,trade_date=date,limit_type='U')
            pool1=pool1.loc[(pool1['up_stat']=='1/1')|(pool1['up_stat']=='2/2')]['ts_code'].to_list()
            pool2=StockData.LimitUD(self,trade_date=pre_date,limit_type='U')
            pool2=pool2.loc[(pool2['up_stat']=='1/1')|(pool2['up_stat']=='2/2')]['ts_code'].to_list()
            pool=list(set(pool1)|set(pool2))
            base_pool=pd.DataFrame(pool,columns=['ts_code'])
            base_poolstr=','.join(pool)

            '''其他基本面条件'''

            '''Condition1:市值最好在30亿以上,250亿以下'''
            MV=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            MV=MV.loc[((MV['total_mv']/10000)>30)&((MV['total_mv']/10000)<200),'ts_code'].to_list()

            '''Condition2:股价尽量在10-60'''
            Price=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            Price=Price.loc[(Price['close']>=10)&(Price['close']<=60),'ts_code'].to_list()


            '''Condition3:两日换手率应当适中在10%-30%'''
            Turnover=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=pre_date2,end_date=date)
            AvgTurnover=Turnover.groupby('ts_code')['turnover_rate'].mean().reset_index(drop=False)
            AvgTurnover=AvgTurnover.loc[(AvgTurnover['turnover_rate']>=5)&(AvgTurnover['turnover_rate']<=30),'ts_code'].to_list()


            finalpool=list(set(pool)&set(MV)&set(Price)&set(AvgTurnover))

        elif pool_name=='pool3':
            pool1=StockData.LimitUD(self,trade_date=date,limit_type='U')
            pool1=pool1.loc[(pool1['up_stat']=='1/1')|(pool1['up_stat']=='2/2')]['ts_code'].to_list()
            pool2=StockData.LimitUD(self,trade_date=pre_date,limit_type='U')
            pool2=pool2.loc[(pool2['up_stat']=='1/1')|(pool2['up_stat']=='2/2')]['ts_code'].to_list()
            pool=list(set(pool1)|set(pool2))
            base_pool=pd.DataFrame(pool,columns=['ts_code'])
            base_poolstr=','.join(pool)

            '''其他基本面条件'''
            '''Condition1:市值最好在30亿以上,250亿以下'''
            MV=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            MV=MV.loc[((MV['total_mv']/10000)>30)&((MV['total_mv']/10000)<200),'ts_code'].to_list()

            '''Condition2:股价尽量在10-60'''
            Price=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            Price=Price.loc[(Price['close']>=10)&(Price['close']<=60),'ts_code'].to_list()


            '''Condition3:两日换手率应当适中在5%-30%'''
            Turnover=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=pre_date2,end_date=date)
            AvgTurnover=Turnover.groupby('ts_code')['turnover_rate'].mean().reset_index(drop=False)
            AvgTurnover=AvgTurnover.loc[(AvgTurnover['turnover_rate']>=5)&(AvgTurnover['turnover_rate']<=30),'ts_code'].to_list()

            '''Condition4:未有当日炸板和跌停板情况'''
            Boom=StockData.LimitUD(self,trade_date=date,limit_type='Z')['ts_code'].to_list()
            Down=StockData.LimitUD(self,trade_date=date,limit_type='D')['ts_code'].to_list()
            BD=list(set(Boom)|set(Down))

            '''Condition5:量比需要在0.8-8范围内'''
            Volume_ratio=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=pre_date2,end_date=date)
            AvgVolume_ratio=Volume_ratio.groupby('ts_code')['volume_ratio'].mean().reset_index(drop=False)
            AvgVolume_ratio=AvgVolume_ratio.loc[(AvgVolume_ratio['volume_ratio']>=0.8)&(AvgVolume_ratio['volume_ratio']<=8),'ts_code'].to_list()

            finalpool=list((set(pool)&set(MV)&set(Price)&set(AvgTurnover)&set(AvgVolume_ratio))-set(BD))

        elif pool_name=='pool4':
            pool1=StockData.LimitUD(self,trade_date=date,limit_type='U')
            pool1=pool1.loc[(pool1['up_stat']=='1/1')|(pool1['up_stat']=='2/2')]['ts_code'].to_list()
            pool=list(set(pool1))
            base_pool=pd.DataFrame(pool,columns=['ts_code'])
            base_poolstr=','.join(pool)

            '''其他基本面条件'''
            '''Condition1:市值最好在30亿以上,200亿以下'''
            MV=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            MV=MV.loc[((MV['total_mv']/10000)>30)&((MV['total_mv']/10000)<200),'ts_code'].to_list()

            '''Condition2:股价尽量在10-60'''
            Price=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            Price=Price.loc[(Price['close']>=10)&(Price['close']<=60),'ts_code'].to_list()


            '''Condition3:两日换手率应当适中在5%-30%'''
            Turnover=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=pre_date2,end_date=date)
            AvgTurnover=Turnover.groupby('ts_code')['turnover_rate'].mean().reset_index(drop=False)
            AvgTurnover=AvgTurnover.loc[(AvgTurnover['turnover_rate']>=5)&(AvgTurnover['turnover_rate']<=30),'ts_code'].to_list()

            '''Condition4:未有当日炸板和跌停板情况'''
            Boom=StockData.LimitUD(self,trade_date=date,limit_type='Z')['ts_code'].to_list()
            Down=StockData.LimitUD(self,trade_date=date,limit_type='D')['ts_code'].to_list()
            BD=list(set(Boom)|set(Down))

            '''Condition5:量比需要在0.8-8范围内'''
            Volume_ratio=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=pre_date2,end_date=date)
            AvgVolume_ratio=Volume_ratio.groupby('ts_code')['volume_ratio'].mean().reset_index(drop=False)
            AvgVolume_ratio=AvgVolume_ratio.loc[(AvgVolume_ratio['volume_ratio']>=0.8)&(AvgVolume_ratio['volume_ratio']<=8),'ts_code'].to_list()

            finalpool=list((set(pool)&set(MV)&set(Price)&set(AvgTurnover)&set(AvgVolume_ratio))-set(BD))


        elif pool_name=='pool5':
            pool1=StockData.LimitUD(self,trade_date=date,limit_type='U')
            pool1=pool1.loc[(pool1['up_stat']=='1/1')|(pool1['up_stat']=='2/2')]['ts_code'].to_list()
            pool2=StockData.LimitUD(self,trade_date=pre_date,limit_type='U')
            pool2=pool2.loc[(pool2['up_stat']=='1/1')|(pool2['up_stat']=='2/2')]['ts_code'].to_list()
            pool=list(set(pool1)|set(pool2))
            base_pool=pd.DataFrame(pool,columns=['ts_code'])
            base_poolstr=','.join(pool)

            '''其他基本面条件'''
            '''Condition1:市值最好在50亿以上'''
            MV=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            MV=MV.loc[((MV['total_mv']/10000)>50),'ts_code'].to_list()

            '''Condition2:股价尽量在10-60'''
            Price=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=date,end_date=date)
            Price=Price.loc[(Price['close']>=10)&(Price['close']<=60),'ts_code'].to_list()


            '''Condition3:两日换手率应当适中在5%-30%'''
            Turnover=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=pre_date2,end_date=date)
            AvgTurnover=Turnover.groupby('ts_code')['turnover_rate'].mean().reset_index(drop=False)
            AvgTurnover=AvgTurnover.loc[(AvgTurnover['turnover_rate']>=5)&(AvgTurnover['turnover_rate']<=30),'ts_code'].to_list()

            '''Condition4:未有当日炸板和跌停板情况'''
            Boom=StockData.LimitUD(self,trade_date=date,limit_type='Z')['ts_code'].to_list()
            Down=StockData.LimitUD(self,trade_date=date,limit_type='D')['ts_code'].to_list()
            BD=list(set(Boom)|set(Down))

            '''Condition5:量比需要在0.8-8范围内'''
            Volume_ratio=StockData.StockDailyBase(self,ts_code=base_poolstr,start_date=pre_date2,end_date=date)
            AvgVolume_ratio=Volume_ratio.groupby('ts_code')['volume_ratio'].mean().reset_index(drop=False)
            AvgVolume_ratio=AvgVolume_ratio.loc[(AvgVolume_ratio['volume_ratio']>=0.8)&(AvgVolume_ratio['volume_ratio']<=8),'ts_code'].to_list()

            finalpool=list((set(pool)&set(MV)&set(Price)&set(AvgTurnover)&set(AvgVolume_ratio))-set(BD))

        save=pd.DataFrame(finalpool,columns=['ts_code'])
        pool_path=rf'Stockpool/{pool_name}'
        if not os.path.exists(pool_path):
            os.makedirs(pool_path)
        save.to_csv(rf'Stockpool/{pool_name}/{date}.csv')
        return finalpool