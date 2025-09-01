from __future__ import print_function, absolute_import
from Tools.Tushare_apidata import *
import tushare as ts
from gm.api import *
print(ts.__version__)

os.chdir(r'D:\Code\自研策略\打板策略\Tools\Historydata')
if __name__ == "__main__":
    # 初始化参数
    start_date='20210527'
    end_date='20260629'
    exchange=''
    
    adj=None
    freq='1MIN'
    StockData=StockData(start_date=start_date,end_date=end_date,frequence=freq)

    # 基础数据保存
    AllStock=StockData.AllStock(exchange='')
    Calendar=StockData.AshareCalendarAll(exchange='',start_date='20200101',end_date=end_date)
    


