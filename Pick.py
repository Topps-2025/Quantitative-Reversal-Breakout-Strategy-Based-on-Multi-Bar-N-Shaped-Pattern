'''组合优化'''
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



class Pick:
    def __init__(self,start_date='20200101',end_date='20210131',frequence='D'):
        self.start_date = start_date
        self.end_date = end_date
        self.frequence = frequence

    def zscore(factor_df, factor_names):
        """
        向量化版本的因子标准化函数
        
        参数:
        factor_df: DataFrame，包含所有因子数据
        factor_names: list，需要标准化的因子名称列表
        
        返回:
        标准化后的DataFrame
        """
        # 复制一份数据避免修改原始数据
        result = factor_df.copy()
            
            # 对每个因子进行标准化
        for factor_name in factor_names:
            factor_values = result[factor_name]
            
            # 计算加权均值和标准差
            weighted_mean = np.nansum(factor_values)
            std = np.nanstd(factor_values)
            
            # 如果标准差接近0，保持原值
            if std <= 1e-8:
                continue
                
            # 标准化
            result.loc[:, factor_name] = (factor_values - weighted_mean) / std
                
        return result

    def StockPick(self,ResultAll,faclist,method_name):
        ResultAll['score']=np.nanmean(ResultAll[faclist], axis=1)
        
        return ResultAll

