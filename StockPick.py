from __future__ import print_function, absolute_import
from Tools.Tushare_apidata import *
from Tools.Test import *
from Tools.Pick import *
pd.set_option('display.max_rows', None)

import tushare as ts
from gm.api import *
# from FactorAllTest import faclist
print(ts.__version__)

os.chdir(r'D:\Code\自研策略\打板策略\Tools\Historydata')
if __name__ == "__main__":
    # faclist=os.listdir('Factor')
    label_list=['T12open','T1opentoT2close','T1closeT2open','T12close','T15close']
    label_num=1
    label_name=label_list[label_num]
    print(label_name)
    start_date='20240525'
    end_date='20250818'
    trade_date='20250825'
    pool_name='pool3' # 记得修改
    freq='D'
    StockData=StockData(start_date=start_date,end_date=end_date,frequence=freq)
    Test=Test(start_date=start_date,end_date=end_date,frequence=freq)
    Pick=Pick(start_date=start_date,end_date=end_date,frequence=freq)
    selectclass='select0525'
    facresult=pd.read_csv(rf'Result\FactorSelect\{pool_name}_{label_name}_{selectclass}_20230525_20240525',index_col=0)
    faclist=facresult['fac_name'].to_list()
    AllDate=pd.read_feather('AshareCalendar.fea')
    AllDate['cal_date']=AllDate['cal_date'].astype(int)
    AllDate=AllDate.sort_values(by='cal_date',ascending=True).reset_index(drop=True)
    AllDateList=AllDate['cal_date'].to_list()

    # # 打板选股策略测试
    # test_date=StockData.AshareCalendar(exchange='',start_date=start_date,end_date=end_date)
    # test_date['cal_date']=test_date['cal_date'].astype(int)
    # test_date=test_date.sort_values(by='cal_date',ascending=True).reset_index(drop=True)
    # test_datelist=test_date['cal_date'].to_list()
    # WinPctAll=pd.DataFrame(columns=['date','winpct','avgrtn'])
    # Method_name='策略3' # 记得修改
    # print(Method_name)
    # for i in range(len(test_datelist)):
    #     date=test_datelist[i]
    #     if date in [20241011,20240930,20240821,20240708,20240624,20230726,20240222,20241008,20241009,20241011]:
    #         continue
    #     print(date)
    #     index=AllDateList.index(date)
    #     test_date=StockData.AshareCalendar(exchange='',start_date=start_date,end_date=end_date)
    #     test_date['cal_date']=test_date['cal_date'].astype(int)
    #     test_date=test_date.sort_values(by='cal_date',ascending=True).reset_index(drop=True)
    #     T1date=AllDateList[index+1]
    #     # 提取对应股票池
    #     finalpool=StockData.Stockpool(pool_name=pool_name,date=date,AllDateList=AllDateList)
    #     base_pool=pd.DataFrame(finalpool,columns=['ts_code'])
    #     base_poolstr=','.join(finalpool)
    #     # 因子数据存储结果
    #     ResultAll=base_pool.copy()
    #     for fac_name in faclist:
    #         fac=pd.read_feather(rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{date}.fea').loc[:,['ts_code','fac']]
    #         fac=fac.rename(columns={'fac':fac_name})
    #         ic=facresult.loc[facresult['fac_name']==fac_name,'ic'].to_list()[0]
    #         # fac[fac_name]=fac[fac_name]*np.sign(ic)
    #         # ic加权因子
    #         # fac[fac_name]=fac[fac_name]*(ic)
    #         fac[fac_name]=(fac[fac_name]-np.nanmean(fac[fac_name]))/np.nanstd(fac[fac_name])
    
    #         ResultAll=pd.merge(ResultAll,fac,on='ts_code',how='outer')
    #     ResultAll=ResultAll.fillna(0)
    #     # ResultAll=ResultAll.dropna()
    #     # 策略1：对因子zscore归一化
    #     ResultAll=Pick.StockPick(ResultAll=ResultAll,faclist=faclist,method_name=Method_name)
    #     StockPick=ResultAll.sort_values(by=['score'],ascending=False).reset_index(drop=True)
    #     print(StockPick)
    #     # if pool_name=='pool3':
    #     #     # if label_name=='T1opentoT2close':
    #     #         StockPick=StockPick.loc[(StockPick['score']>=0)].sort_values(by=['score'],ascending=False).reset_index(drop=True)
    #     Stocknum=5 # 待设定策略参数
    #     StockPick=StockPick.loc[:,'ts_code'].to_list()[:Stocknum]
    #     # 排除当日收盘无法买入的股票
    #     LUDTALL=StockData.LimitUD(trade_date=T1date,limit_type='U')
    #     LUDTALL['last_time']=LUDTALL['last_time'].astype(int)
    #     if label_num>=2:
    #         LUSTOCK=LUDTALL.loc[LUDTALL['last_time']<=140000,'ts_code'].to_list()
    #     else:
    #         LUSTOCK=LUDTALL.loc[LUDTALL['last_time']<=93100,'ts_code'].to_list()
    #     StockPick=list(set(StockPick)-set(LUSTOCK))
    #     print(StockPick)
    #     if len(StockPick)==0:
    #         continue
    #     label=pd.read_feather(rf'Label/{pool_name}/{label_name}/{label_name}_{date}.fea').loc[:,['ts_code','label']]
    #     label=label.drop_duplicates(subset='ts_code')
    #     benchmark=0.001 # 待设定策略参数
    #     loss_benchmark=-0.15
    #     WinPct=Test.CalWinPct(date=date,Stocklist=StockPick,label=label,benchmark=benchmark,loss_benchmark=loss_benchmark)
        
    #     WinPctAll=pd.concat([WinPctAll,WinPct],axis=0)
    #     print(WinPctAll)
    #     WinPctAll.to_csv(rf'Result/WinPct/{Method_name}_{label_name}_{Stocknum}_{benchmark}_{loss_benchmark}.csv')

    # 当天选股结果(第二天买入)
    print(trade_date)
    index=AllDateList.index(int(trade_date))
    test_date=StockData.AshareCalendar(exchange='',start_date=start_date,end_date=trade_date)
    test_date['cal_date']=test_date['cal_date'].astype(int)
    test_date=test_date.sort_values(by='cal_date',ascending=True).reset_index(drop=True)

    # 提取对应股票池
    finalpool=StockData.Stockpool(pool_name=pool_name,date=int(trade_date),AllDateList=AllDateList)
    base_pool=pd.DataFrame(finalpool,columns=['ts_code'])
    base_poolstr=','.join(finalpool)

    ResultAll=base_pool.copy()
    faclist_final=faclist.copy()
    for fac_name in faclist:
        fac=pd.read_feather(rf'Factor/{pool_name}/{fac_name}/feature/{fac_name}_{trade_date}.fea').loc[:,['ts_code','fac']]
        fac=fac.rename(columns={'fac':fac_name})
        ic=facresult.loc[facresult['fac_name']==fac_name,'ic'].to_list()[0]
        fac[fac_name]=fac[fac_name]*np.sign(ic)
        if len(fac)==0:
            faclist_final.remove(fac_name)
            continue
        fac[fac_name]=(fac[fac_name]-np.nanmean(fac[fac_name]))/np.nanstd(fac[fac_name])
        ResultAll=pd.merge(ResultAll,fac,on='ts_code',how='outer')
    ResultAll['score']=np.nanmean(ResultAll[faclist_final], axis=1)
    ResultAll=ResultAll.drop_duplicates(subset='score').reset_index(drop=True)
    ResultAll[faclist_final]=ResultAll[faclist_final].fillna(0)
    StockPick=ResultAll.sort_values(by=['score'],ascending=False).reset_index(drop=True)
    print(StockPick)
    # if pool_name=='pool3':
    #     if label_name=='T1opentoT2close':
    #         StockPick=StockPick.loc[(StockPick['score']>=0.4)].sort_values(by=['score'],ascending=False).reset_index(drop=True)
    Stocknum=5 # 待设定策略参数
    StockPick=StockPick.loc[:,'ts_code'].to_list()[:Stocknum]
    print(StockPick)
    StockPickAll=pd.DataFrame(StockPick,columns=['ts_code'])
    StockPickAll['date']=trade_date
