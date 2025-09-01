from __future__ import print_function, absolute_import
from Tools.Tushare_apidata import *
from Tools.Test import *
import tushare as ts
from gm.api import *
print(ts.__version__)
import pandas as pd
os.chdir(r'D:\Code\自研策略\打板策略\Tools\Historydata')

pool_name='pool3'
faclist=os.listdir(rf'Factor/{pool_name}')
label_name='T12open'
label_name='T1opentoT2close'
label_list=['T12open','T1opentoT2close','T1closeT2open','T12close']
# label_list=['T1closeT2open']

start_date=20230525
end_date=20240525
freq='d'

Test=Test(start_date=start_date,end_date=end_date,frequence=freq)
# ResultAll=pd.DataFrame(columns=['fac_name','label_name','ic','ir','icir'])
selectclass='select0525'
for label_name in label_list:
    ResultAll=pd.DataFrame(columns=['fac_name','label_name','ic','ir','icir','period'])
    for fac_name in faclist:
        # if fac_name not in ['DTTurn','Large_ratio','Large_bs_ratio','DTMulTurn','DTnetrate_FV','Buysell_Exratio','Netrate_Close','MarginBuy_ratio','MarginZQ_ratio']:
        #     continue
        Result=Test.FactorTest(fac_name=fac_name,label_name=label_name,start_date=start_date,end_date=end_date,pool_name=pool_name)
        Result['period']=rf'{start_date}_{end_date}'
        ResultAll=pd.concat([ResultAll,Result],axis=0)
        ResultAll=ResultAll.loc[abs(ResultAll['ic'])>=0.02,:]
        ResultAll.to_csv(rf'Result\FactorSelect\{pool_name}_{label_name}_{selectclass}_{start_date}_{end_date}')
    print(ResultAll)