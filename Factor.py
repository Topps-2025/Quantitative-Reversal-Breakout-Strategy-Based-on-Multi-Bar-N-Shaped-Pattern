from __future__ import print_function, absolute_import
from Tools.Tushare_apidata import *
from Tools.Test import *
from Tools.Toolkits import *
import tushare as ts
from gm.api import *
print(ts.__version__)
import time

os.chdir(r'D:\Code\自研策略\打板策略\Tools\Historydata')


if __name__ == "__main__":
    # 初始化参数
    # start_date='20230525'
    start_date='20250829'
    end_date='20250829'
    exchange=''
    adj=None
    freq='D'
    pool_name='pool3'
    factor_base_path=rf'D:\Code\自研策略\打板策略\Tools\Historydata\Factor\{pool_name}'
    label_base_path=r'D:\Code\自研策略\打板策略\Tools\Historydata\Label'
    # 是否计算label
    save_label_TF=False
    # 是否覆保存（覆盖）ic回测结果
    save_ic=False
    StockData=StockData(start_date=start_date,end_date=end_date,frequence=freq)
    Test=Test(start_date=start_date,end_date=end_date,frequence=freq)
    
    label_list=['T12open','T1opentoT2close','T1closeT2open','T12close','T15close']
    subfolder=label_list+['feature']
    def load_fac_if_valid(factor_path: str):
        try:
            if not os.path.exists(factor_path):
                return None
            df = pd.read_feather(factor_path)
            if df is None or len(df) == 0:
                return None
            # 只保留所需列并排序去重
            if 'ts_code' not in df.columns or 'fac' not in df.columns:
                return None
            df = df.loc[:, ['ts_code', 'fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
            # fac 列不应全为0或NA
            fac_series = df['fac']
            # 将0视为无效值，与NA一起检查是否全无有效值
            if fac_series.replace(0, pd.NA).dropna().empty:
                return None
            return df
        except Exception:
            return None
    def FactorResult(base_poolstr,fac,fac_name,label_name,date,T1date,T2date,pool_name):
        labeltest=StockData.Callabel(label_name=label_name,base_poolstr=base_poolstr,date=date,start_date=T1date,end_date=T2date,pool_name=pool_name)
        result=Test.ForwardTest(date=date,fac_name=fac_name,label_name=label_name,fac=fac,label=labeltest,pool_name=pool_name)
        Result_date=pd.DataFrame([[fac_name,label_name,date,result]],columns=['fac','label_name','date','ic'])
        return Result_date

    def LabelResult(ResultAll,base_poolstr,fac,fac_name,label_name,date,T1date,T2date,save_ic,pool_name):
        key=f"result_{fac_name}_{label_name}"
        Result_date=FactorResult(base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_name=label_name,date=date,T1date=T1date,T2date=T2date,pool_name=pool_name)
        print(Result_date)
        ResultAll[key]=pd.concat([ResultAll[key],Result_date],axis=0)
        if save_ic==True:
            folder_path=rf'Result/{fac_name}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            ResultAll[key].to_csv(rf'Result/{fac_name}/{pool_name}_{fac_name}_{label_name}.csv')
            

    def feature_label_save(ResultAll,base_pool,base_poolstr,fac,fac_name,label_list,date,AllDate,save_label,save_ic,pool_name):
        if len(fac)==0:
            fac=base_pool.copy()
            fac['fac']=0
        fac.to_feather((rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'))
        if save_label==True:
            index=AllDate.index(date)
            for label_name in label_list:
                if label_name=='T15close':
                    T1date=AllDateList[index+1]
                    T2date=AllDateList[index+6]
                else:
                    T1date=AllDateList[index+1]
                    T2date=AllDateList[index+2]
                LabelResult(ResultAll=ResultAll,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_name=label_name,date=date,T1date=T1date,T2date=T2date,save_ic=save_ic,pool_name=pool_name)

    # 提取测试时间
    AllDate=pd.read_feather('AshareCalendar.fea')
    AllDate['cal_date']=AllDate['cal_date'].astype(int)
    AllDate=AllDate.sort_values(by='cal_date',ascending=True).reset_index(drop=True)
    AllDateList=AllDate['cal_date'].to_list()

    test_date=StockData.AshareCalendar(exchange='',start_date=start_date,end_date=end_date)
    test_date['cal_date']=test_date['cal_date'].astype(int)
    test_date=test_date.sort_values(by='cal_date',ascending=True).reset_index(drop=True)
    test_datelist=test_date['cal_date'].to_list()

    # 记录因子结果
    ResultAll=dict()
    fac_list=os.listdir(rf'Factor/{pool_name}')
    label_list=['T12open','T1opentoT2close','T1closeT2open','T12close']
    for fac in fac_list:
        for label in label_list:
            key = f"result_{fac}_{label}"  # 生成字典键名
            ResultAll[key] = pd.DataFrame(columns=['fac','label_name','date','ic'])  # 创建空DataFrame并存入字典
    
    for i in range(len(test_datelist)):
        date=test_datelist[i]
        
        if date in [20241011,20240930,20240821,20240708,20240624,20230726,20240222,20241008,20241009,20241011]:
            continue
        print(date)
        index=AllDateList.index(date)
        pre_date=AllDateList[index-1]
        pre_date2=AllDateList[index-2]

        # 回调博弈策略：
        # 考虑个股在首板或连板后首次分歧后，回调产生资金上的博弈机会，趁分歧带来的低成本进行尾盘抢筹

        # 处理股票池
        finalpool=StockData.Stockpool(pool_name=pool_name,date=int(date),AllDateList=AllDateList)
        base_pool=pd.DataFrame(finalpool,columns=['ts_code'])
        base_poolstr=','.join(finalpool)
        if len(base_pool)>=1000 or len(base_pool)==0:
            continue

        print(base_pool)

        '''fac1:龙虎榜净买入额与换手率之比'''
        fac_name='DTTurn'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            DTALL=StockData.DTlist(date)
            DTBase=DTALL.loc[DTALL['ts_code'].isin(base_pool['ts_code']),['ts_code','net_amount','turnover_rate']]
            DTBase['fac']=DTBase['net_amount']/DTBase['turnover_rate']
            if len(DTBase['fac'])==0:
                DTBase=base_pool.copy()
                DTBase['fac']=0
            fac=DTBase.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac2:当天(或近两天内)个股特大单、大单占比'''
        fac_name='Large_ratio'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            MoneyFlow=StockData.MoneyFlow(ts_code=base_poolstr,start_date=date,end_date=date)
            MoneyFlow['fac']=(MoneyFlow['buy_lg_amount']+MoneyFlow['buy_elg_amount'])/MoneyFlow['net_mf_amount']
            fac=MoneyFlow.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac3:当天(或近两天内)个股特大单、大单买卖盘占比'''
        fac_name='Large_bs_ratio'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            MoneyFlow=StockData.MoneyFlow(ts_code=base_poolstr,start_date=date,end_date=date)
            MoneyFlow['fac']=(MoneyFlow['buy_lg_amount']+MoneyFlow['buy_elg_amount'])/(MoneyFlow['sell_lg_amount']+MoneyFlow['sell_elg_amount'])
            fac=MoneyFlow.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)
        

        '''fac4:龙虎榜净买入额与换手率之比'''
        fac_name='DTMulTurn'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            DTALL=StockData.DTlist(date)
            DTBase=DTALL.loc[DTALL['ts_code'].isin(base_pool['ts_code']),['ts_code','net_amount','turnover_rate']]
            DTBase['fac']=DTBase['net_amount']*DTBase['turnover_rate']
            fac=DTBase.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)


        '''fac5:龙虎榜净买入额占比与净买入/流通市值之比'''
        fac_name='DTnetrate_FV'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            DTALL=StockData.DTlist(date)
            DTBase=DTALL.loc[DTALL['ts_code'].isin(base_pool['ts_code']),['ts_code','net_rate','net_amount','float_values']]
            DTBase['fac']=DTBase['net_rate']/(DTBase['net_amount']/DTBase['float_values'])
            fac=DTBase.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac6:筹码分布'''
        fac_name='Pareto85_Chip'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            ChipAll=pd.DataFrame()
            for ts_code in base_pool['ts_code']:
                Chip=StockData.DailyChip(ts_code=ts_code,start_date=date,end_date=date)
                ChipAll=pd.concat([ChipAll,Chip],axis=0)
            ChipAll['fac']=ChipAll['cost_85pct']/ChipAll['cost_15pct']
            fac=ChipAll.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac7:大买卖预期占比'expression': '(buy_elg_amount * sell_lg_amount) / total_mv '''
        fac_name='Buysell_Exratio'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            MoneyFlow=StockData.MoneyFlow(ts_code=base_poolstr,start_date=date,end_date=date)
            MV=StockData.StockDailyBase(ts_code=base_poolstr,start_date=date,end_date=date)
            Money_MV=pd.merge(MoneyFlow,MV,on=['ts_code'])
            Money_MV['fac']=(Money_MV['sell_lg_amount']*Money_MV['buy_elg_amount'])/(Money_MV['total_mv'])
            fac=Money_MV.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac8:净价率比 'expression': 'net_rate / close' '''
        fac_name='Netrate_Close'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            DTALL=StockData.DTlist(date)
            DTBase=DTALL.loc[DTALL['ts_code'].isin(base_pool['ts_code']),['ts_code','net_rate','close']]
            DTBase['fac']=DTBase['net_rate']/DTBase['close']
            fac=DTBase.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)
        
        '''fac9:融资买入比MarginBuy_ratio'''
        fac_name='MarginBuy_ratio'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            MarginAll=StockData.Margin(ts_code=base_poolstr,start_date=date,end_date=date)
            Margin=MarginAll.loc[MarginAll['ts_code'].isin(base_pool['ts_code']),['ts_code','rzmre','rzche','trade_date']]
            Margin['fac']=Margin['rzmre']/Margin['rzche']
            
            if len(Margin['fac'])==0:
                Margin=base_pool.copy()
                Margin['fac']=0
            fac=Margin.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)
        

        '''fac10:融资融券余额比MarginZQ_ratio'''
        fac_name='MarginZQ_ratio'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            MarginAll=StockData.Margin(ts_code=base_poolstr,start_date=date,end_date=date)
            Margin=MarginAll.loc[MarginAll['ts_code'].isin(base_pool['ts_code']),['ts_code','rzye','rzrqye','trade_date']]
            Margin['fac']=Margin['rzye']/Margin['rzrqye']
            if len(Margin['fac'])==0:
                Margin=base_pool.copy()
                Margin['fac']=0
            fac=Margin.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac11:振动升降指标'''
        fac_name='asi_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea' 
        TechFactorAll=StockData.TechFactor_Tushare(trade_date=date) 
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactorAll=StockData.TechFactor_Tushare(trade_date=date)
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','asi_bfq','trade_date']]
            TechFactor['fac']=TechFactor['asi_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac12:真实波动'''
        fac_name='atr_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','atr_bfq','trade_date']]
            TechFactor['fac']=TechFactor['atr_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac13:BBI多空指标'''
        fac_name='bbi_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','bbi_bfq','trade_date']]
            TechFactor['fac']=TechFactor['bbi_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac14:BIAS乖离率'''
        fac_name='bias1_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','bias1_bfq','trade_date']]
            TechFactor['fac']=TechFactor['bias1_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac14:BIAS乖离率2'''
        fac_name='bias2_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','bias2_bfq','trade_date']]
            TechFactor['fac']=TechFactor['bias2_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac15:BIAS乖离率3'''
        fac_name='bias3_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','bias3_bfq','trade_date']]
            TechFactor['fac']=TechFactor['bias3_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac16:BOLL指标_low'''
        fac_name='boll_lower_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','boll_lower_bfq','trade_date']]
            TechFactor['fac']=TechFactor['boll_lower_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac17:BRAR情绪指标'''
        fac_name='brar_ar_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','brar_ar_bfq','trade_date']]
            TechFactor['fac']=TechFactor['brar_ar_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac18:CCI指标'''
        fac_name='cci_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','cci_bfq','trade_date']]
            TechFactor['fac']=TechFactor['cci_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)
        
        '''fac19:CR价格动量指标'''
        fac_name='cr_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','cr_bfq','trade_date']]
            TechFactor['fac']=TechFactor['cr_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac20:平行线差指标'''
        fac_name='dfma_dif_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','dfma_dif_bfq','trade_date']]
            TechFactor['fac']=TechFactor['dfma_dif_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac21:动向指标'''
        fac_name='dmi_adx_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','dmi_adx_bfq','trade_date']]
            TechFactor['fac']=TechFactor['dmi_adx_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac22:区间震荡线'''
        fac_name='dpo_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','dpo_bfq','trade_date']]
            TechFactor['fac']=TechFactor['dpo_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)
        
        '''fac23:指数移动平均'''
        fac_name='ema_bfq_10'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','ema_bfq_10','trade_date']]
            TechFactor['fac']=TechFactor['ema_bfq_10']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)


        '''fac24:简易波动指标'''
        fac_name='emv_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','emv_bfq','trade_date']]
            TechFactor['fac']=TechFactor['emv_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac25:EMA指数平均数指标'''
        fac_name='expma_12_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','expma_12_bfq','trade_date']]
            TechFactor['fac']=TechFactor['expma_12_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac26:KDJ指标'''
        fac_name='kdj_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','kdj_bfq','trade_date']]
            TechFactor['fac']=TechFactor['kdj_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac27:肯特纳交易通道'''
        fac_name='ktn_down_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','ktn_down_bfq','trade_date']]
            TechFactor['fac']=TechFactor['ktn_down_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac28:梅斯线'''
        fac_name='mass_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','mass_bfq','trade_date']]
            TechFactor['fac']=TechFactor['mass_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac29:MFI指标'''
        fac_name='mfi_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','mfi_bfq','trade_date']]
            TechFactor['fac']=TechFactor['mfi_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac30:能量潮指标'''
        fac_name='obv_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','obv_bfq','trade_date']]
            TechFactor['fac']=TechFactor['obv_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac31:投资者对股市涨跌产生心理波动的情绪指标'''
        fac_name='psy_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','psy_bfq','trade_date']]
            TechFactor['fac']=TechFactor['psy_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)


        '''fac32:变动率指标'''
        fac_name='roc_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','roc_bfq','trade_date']]
            TechFactor['fac']=TechFactor['roc_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac33:RSI指标'''
        fac_name='rsi_bfq_6'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','rsi_bfq_6','trade_date']]
            TechFactor['fac']=TechFactor['rsi_bfq_6']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac34:唐安奇通道(海龟)交易指标'''
        fac_name='taq_up_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','taq_up_bfq','trade_date']]
            TechFactor['fac']=TechFactor['taq_up_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)


        '''fac35:三重指数平滑平均线'''
        fac_name='trix_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','trix_bfq','trade_date']]
            TechFactor['fac']=TechFactor['trix_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac35:VR容量比率'''
        fac_name='vr_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','vr_bfq','trade_date']]
            TechFactor['fac']=TechFactor['vr_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac36:W&R 威廉指标'''
        fac_name='wr_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','wr_bfq','trade_date']]
            TechFactor['fac']=TechFactor['wr_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)

        '''fac36:薛斯通道II'''
        fac_name='xsii_td1_bfq'
        print(fac_name)
        CreateFolder(base_path=factor_base_path,main_folder=fac_name,subfolders=subfolder)
        factor_path = rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea'  
        fac = load_fac_if_valid(factor_path)
        if fac is None:
            TechFactor=TechFactorAll.loc[TechFactorAll['ts_code'].isin(base_pool['ts_code']),['ts_code','xsii_td1_bfq','trade_date']]
            TechFactor['fac']=TechFactor['xsii_td1_bfq']
            fac=TechFactor.loc[:,['ts_code','fac']].drop_duplicates(subset=['ts_code']).sort_values(by='ts_code').reset_index(drop=True)
        feature_label_save(ResultAll=ResultAll,base_pool=base_pool,base_poolstr=base_poolstr,fac=fac,fac_name=fac_name,label_list=label_list,date=date,AllDate=AllDateList,save_label=save_label_TF,save_ic=save_ic,pool_name=pool_name)