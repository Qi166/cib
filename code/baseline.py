#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 08:09:43 2018

@author: qi
"""

%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import f1_score

pd.set_option('display.max_columns',None)
# In[0]


PATH_DATA_RAW = './input/'
entbase_raw = pd.read_csv(PATH_DATA_RAW + "1baseinfo.csv")
alter_raw = pd.read_csv(PATH_DATA_RAW + "2alterinfo.csv")
branch_raw = pd.read_csv(PATH_DATA_RAW + "3branchinfo.csv")
invest_raw = pd.read_csv(PATH_DATA_RAW + "4investinfo.csv")
right_raw = pd.read_csv(PATH_DATA_RAW + "5rightinfo.csv")
project_raw = pd.read_csv(PATH_DATA_RAW + "6projectinfo.csv")
case_raw = pd.read_csv(PATH_DATA_RAW + "7caseinfo.csv")
dishonest_raw = pd.read_csv(PATH_DATA_RAW + "8dishonestinfo.csv")
recruit_raw = pd.read_csv(PATH_DATA_RAW + "9jobinfo.csv")

train_raw = pd.read_csv(PATH_DATA_RAW + "train.csv")
test_raw = pd.read_csv(PATH_DATA_RAW + "evaluation_public.csv")

# In[1] base

###类别特征交互特征（频数）
def get_interaction_feature(df, feature_A, feature_B):
    tmp1=pd.Series(le.fit_transform(df[feature_A]))
    tmp2=pd.Series(le.fit_transform(df[feature_B]))
    feature_A_list = sorted(tmp1.unique())
    feature_B_list = sorted(tmp2.unique())
    count = 0
    mydict = {}
    for i in feature_A_list:
        mydict[int(i)] = {}
        for j in feature_B_list:
            mydict[int(i)][int(j)] = count
            count += 1
    return df.apply(lambda x: mydict[int(x[feature_A])][int(x[feature_B])], axis=1)


entbase =entbase_raw.copy()
entbase['CreateYear'] = 2018- entbase['CreateYear']
entbase['FeatureSum'] = entbase[['Feature1','Feature2','Feature3','Feature4']].sum(axis=1)
entbase['Feature3CapitalRat']=entbase['Feature3']/entbase['RegisteredCapital']
entbase['Feature3CreatYearRat']=entbase['Feature3']/entbase['CreateYear']
entbase['RegisteredCapitalCreatYearRat']=entbase['RegisteredCapital']/entbase['CreateYear']

entbase["CreateYearRegisteredCapital"] = get_interaction_feature(entbase, "CreateYear", "RegisteredCapital")
entbase["CreateYearFeature1"] = get_interaction_feature(entbase, "CreateYear", "Feature1")
entbase["CreateYearFeature2"] = get_interaction_feature(entbase, "CreateYear", "Feature2")
entbase["CreateYearFeature3"] = get_interaction_feature(entbase, "CreateYear", "Feature3")
entbase["CreateYearFeature4"] = get_interaction_feature(entbase, "CreateYear", "Feature4")
entbase["CreateYearFeature5"] = get_interaction_feature(entbase, "CreateYear", "Feature5")


# 不同行业平均资产规模不同，将注册资本转换为企业在其所属行业内的相对资产规模
avg_capital_tradetype = entbase.groupby('TradeType', as_index=False).RegisteredCapital.mean()
avg_capital_tradetype.columns=['TradeType','AvgTradeTypeCapital']   # 替换字段名，避免和原字段重名
avg_capital_type = entbase.groupby('Type', as_index=False).RegisteredCapital.mean()
avg_capital_type.columns=['Type','AvgTypeCapital']   # 替换字段名，避免和原字段重名

# 拼表
entbase = pd.merge(entbase, avg_capital_tradetype, on='TradeType', how='left')
entbase = pd.merge(entbase, avg_capital_type, on='Type', how='left')
entbase['TradeTypeCapitalSize'] = entbase['RegisteredCapital'] / entbase['AvgTradeTypeCapital']
entbase['TypeCapitalSize'] = entbase['RegisteredCapital'] / entbase['AvgTypeCapital']


# In[2] Alter

alter_raw.drop_duplicates(inplace=True)
alter = alter_raw.copy()
#正则表达式，取出数字部分
alter['AlterBefore'] = alter['AlterBefore'].str.extract('(?P<AlterBefore>[0-9]+)', expand=True)
alter['AlterAfter'] = alter['AlterAfter'].str.extract('(?P<AlterAfter>[0-9]+)', expand=True)
alter.drop_duplicates(inplace=True)
alter['AlterBefore'] = alter['AlterBefore'][alter['AlterBefore'].isnull()==False].apply(lambda x: int(x))
alter['AlterAfter'] = alter['AlterAfter'][alter['AlterAfter'].isnull()==False].apply(lambda x: int(x))
alter['ALTDIFF'] = alter['AlterAfter'] - alter['AlterBefore']
#check 如赛题描述，仅AlterNumber为“05”和“27”有数据
alter['AlterNumber'][alter['AlterBefore'].isnull() == False].unique()

#为了拿到同一变更事项下的统计特征，将alter表按照 05  27 拆分
alter_05 = alter[alter['AlterNumber'] == '05']
alter_05.head()
alter_05_sum = alter_05.groupby(['EID'])['ALTDIFF'].sum().reset_index()
alter_05_sum.head()
#重命名 （防止表连接时名字起冲突）
alter_05_sum.columns = ['EID', 'alt_05_sum']
alter_05 = pd.merge(alter_05,alter_05_sum,how='left',on='EID')
alter_05.head()
#但是该表中，有多条记录，如何按照EID聚合呢？对于这类于时间有关的问题，可以按照时间先后顺序取得最合适的一行记录
alter_05[alter_05['EID'] == 851646734]
def get_latest_alter(alter):
    del alter['AlterNumber']
    alter = alter.sort_values(['EID','AlterDate'], ascending=False)
    alter['cumcount'] = alter.groupby(['EID']).cumcount()
    alter = alter[alter['cumcount'] == 0]
    del alter['cumcount']
    return alter
alter_05 = get_latest_alter(alter_05)
alter_05[alter_05['EID'] == 851646734] #check

###时间戳转化为时间跨度
def translate_year(date):
    year = int(date[:4])
    month = int(date[-2:])
    return (year-2010)*12 + month
alter_05['alter_05_year'] = alter_05['AlterDate'].str.split('-').str[0].apply(lambda x: int(x))
alter_05['alter_05_month'] = alter_05['AlterDate'].str.split('-').str[1].apply(lambda x: int(x))
del alter_05['AlterDate']
alter_05.head()
alter_05.rename(columns=lambda x:x.replace('ALTDIFF','ALTDIFF05'), inplace=True)

alter_27 = alter[alter['AlterNumber'] == '27']
alter_27_sum = alter_27.groupby(['EID'])['ALTDIFF'].sum().reset_index()
alter_27_sum.columns = ['EID', 'alt_27_sum']
alter_27 = pd.merge(alter_27,alter_27_sum,how='left',on='EID')
alter_27 = get_latest_alter(alter_27)
alter_27['alter_27_year'] = alter_27['AlterDate'].str.split('-').str[0].apply(lambda x: int(x))
alter_27['alter_27_month'] = alter_27['AlterDate'].str.split('-').str[1].apply(lambda x: int(x))
del alter_27['AlterDate']
alter_27.rename(columns=lambda x:x.replace('ALTDIFF','ALTDIFF27'), inplace=True)


le = preprocessing.LabelEncoder()
le.fit(alter['AlterNumber'])
alter['AlterNumber'] = le.transform(alter['AlterNumber']) 

alter = alter.sort_values(['EID','AlterDate'], ascending=True)
alter['alt_year'] = alter['AlterDate'].str.split('-').str[0].apply(lambda x: int(x))
alter['alt_month'] = alter['AlterDate'].str.split('-').str[1].apply(lambda x: int(x))


#将以上信息聚合到alter主表中，我们就得到了，一个企业（EID）下，2015年的变更总次数
#将这些步骤整合成函数：
def get_alter_year_count(df,col_name,year_col_name,year):
    df['tmp'] = 1
    df_tmp = df[df[year_col_name] == year].groupby(['EID'])['tmp'].sum()
    df_tmp = pd.DataFrame({'EID':df_tmp.index, col_name:df_tmp.values})
    del df['tmp']
    df = pd.merge(df, df_tmp, how='left', on='EID')
    #对于2015年没记录的企业，空值填充为0 （实际意义）
    df[col_name] = df[col_name].fillna(0)
    return df

alter = get_alter_year_count(alter,'alter_count_2015','alt_year',2015)

def divide_alter_no(df, num):
    col_name = 'alter_count_' + str(num)
    df[col_name] = 0
    df[col_name][df['AlterNumber'] == num] = 1
    df_tmp = df.groupby(['EID'])[col_name].sum()
    df_tmp = pd.DataFrame({'EID':df_tmp.index, col_name:df_tmp.values})
    del df[col_name]
    df = pd.merge(df, df_tmp, how='left', on='EID')
    return df

# for i in range(12):
#     alter = divide_alter_no(alter, i)
    
alter['alter_count'] = alter.groupby(['EID']).cumcount() + 1
idx = alter.groupby(['EID'])['alter_count'].transform(max) == alter['alter_count']
alter = alter[idx]

alter = alter.drop(['AlterNumber','AlterDate','AlterBefore','AlterAfter'], axis = 1)

alter_05 = alter_05.drop(['AlterBefore','AlterAfter'], axis = 1)
alter_27 = alter_27.drop(['AlterBefore','AlterAfter'], axis = 1)

alter = pd.merge(alter, alter_05, how='left', on='EID')
alter = pd.merge(alter, alter_27, how='left', on='EID')


# In[3] Branch

branch_raw = branch_raw.drop_duplicates()
branch = branch_raw.copy()

#先检查注册年份 和 倒闭年份 直接 有没有逻辑错误
branch[branch.CloseYear < branch.RegYear].head()
wrong_index = branch.CloseYear < branch.RegYear
branch.loc[wrong_index,'CloseYear'], branch.loc[wrong_index, 'RegYear'] = \
branch.loc[wrong_index,'RegYear'], branch.loc[wrong_index,'CloseYear']

del branch['BranchEID']
branch = branch.sort_values(['EID','RegYear'], ascending=True)
branch['branch_count'] = branch.groupby(['EID']).cumcount() + 1
branch.head()

def get_annual_close(df, col_name, year_col_name, year):
    df[col_name] = 0
    df[col_name][(df[year_col_name].isnull() == False) & (df[year_col_name] <= year)] = 1
    df_tmp = df.groupby(['EID'])[col_name].sum()
    df_tmp = pd.DataFrame({'EID':df_tmp.index, col_name:df_tmp.values})
    del df[col_name]
    df = pd.merge(df, df_tmp, how='left', on='EID')
    return df
branch = get_annual_close(branch, 'branch_end_count_2015', 'CloseYear', 2015)
branch[branch['EID'] == 103798]
idx = branch.groupby(['EID'])['branch_count'].transform(max) == branch['branch_count']
branch = branch[idx]
branch.rename(columns=lambda x:x.replace('SameProvince','branch_same'), inplace=True)

branch['RegYear'] = 2018 - branch['RegYear']
branch['CloseYear'] = 2018 - branch['CloseYear']


# In[4] Dishonest

dishonest_raw = dishonest_raw.drop_duplicates()
dishonest_raw.shape
dishonest = dishonest_raw.copy()

# 处理日期格式
import re
dishonest['RegYear'] = dishonest['RegDate'].apply(lambda x: int(x[:4]))
dishonest['RegDate'] = dishonest['RegDate'].apply(lambda x: re.sub("\D", "", x)).apply(lambda x: int(x))
dishonest['CloseDate'] = dishonest['CloseDate'].str.replace('-','')
dishonest[dishonest.CloseDate.notnull()].head()
dishonest.RegYear.describe()
# 将非空的结束日期转化为数值
dishonest['CloseDate'] = dishonest['CloseDate'][dishonest['CloseDate'].notnull()].apply(lambda x: int(x))
# 最近三年每年内总共发生多少次失信记录
# 最近三年总共发生多少次失信
# 发生的失信记录中，有多少至今还未关闭

# 统计某一年发生的失信记录数
def dishonest_count(df, group_key, filter_key, filter_value, agg_col, new_col_name):
    new_df = df[df[filter_key]==filter_value].groupby(group_key).nunique()[[agg_col]]
    # 重命名列，以避免merge时冲突
    new_df.rename(columns=lambda x:x.replace(agg_col, new_col_name), inplace=True)
    return pd.merge(df, new_df, left_on=group_key, right_index=True,how='left').fillna({new_col_name:0})

# 统计还未注销的失信记录数
def dishonest_unclose_count(df, group_key, filter_key, agg_col, new_col_name):
    new_df = df[df[filter_key].isnull()].groupby(group_key).nunique()[[agg_col]]
    # 重命名列，以避免merge时冲突
    new_df.rename(columns=lambda x:x.replace(agg_col, new_col_name), inplace=True)
    return pd.merge(df, new_df, left_on=group_key, right_index=True,how='left').fillna({new_col_name:0})
dishonest = dishonest_count(dishonest, 'EID', 'RegYear', 2013, 'DishonestID', 'dis_count_2013')
dishonest[(dishonest.EID==542420088) & (dishonest.RegYear==2013)]
dishonest = dishonest_count(dishonest, 'EID', 'RegYear', 2014, 'DishonestID', 'dis_count_2014')
dishonest = dishonest_count(dishonest, 'EID', 'RegYear', 2015, 'DishonestID', 'dis_count_2015')
dishonest = dishonest_unclose_count(dishonest, 'EID', 'CloseDate', 'DishonestID', 'dis_count_unclose')
# 只取最近发生的失信记录中，取还没有注销的记录
dishonest['dis_count'] = dishonest.sort_values(['EID','RegDate','CloseDate'], ascending=[True, True, False])\
                            .groupby(['EID']).cumcount() + 1
idx = dishonest.groupby(['EID'])['dis_count'].transform(max) == dishonest['dis_count']
dishonest = dishonest[idx]
# 重命名
dishonest.rename(columns=lambda x:x.replace('RegDate','LastDisRegDate'), inplace=True)
dishonest.rename(columns=lambda x:x.replace('CloseDate','LastDisCloseDate'), inplace=True)
dishonest.rename(columns=lambda x:x.replace('RegYear','LastDisRegYear'), inplace=True)


# In[5] Merge tables
del train_raw['EndDate']
del test_raw['PROB']

total = pd.concat([train_raw,test_raw])

total = pd.merge(total, entbase, how='left', on='EID')
total = pd.merge(total, alter, how='left', on='EID')
total = pd.merge(total, branch, how='left', on='EID')
total = pd.merge(total, dishonest, how='left', on='EID')
total = total.apply(pd.to_numeric, errors='ignore')

train = total[total['Y'].isnull() == False]
test = total[total['Y'].isnull() == True]
del test['Y']

train[['AvgTradeTypeCapital','TradeTypeCapitalSize','RegisteredCapital','Y']].corr()

#del train['TradeTypeCapitalSize']
#del test['TradeTypeCapitalSize']


# In[99]

from util import *
# lgb
l_params = {'metric': 'auc', 'learning_rate' : 0.1, \
          'max_depth': -1, 'max_bin': 20,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction': 0.9,\
          'bagging_freq': 2, 'num_leaves': 15,\
         'boosting_type': 'gbdt'}

train_cv = train.copy()
train_cv.shape

X = train_cv.drop(['EID', 'Y'], axis=1)
y = train_cv['Y']

lgb_cv, f1_th = evaluate(X, y)

###Pred
model = lgb.train(l_params, lgb.Dataset(X, label=y), len(lgb_cv['auc-mean']))
pred = model.predict(test.drop(['EID'], axis=1))

'''
test['PROB'] = pred
sub = pd.DataFrame({'EID':test.EID, 'PROB':test.PROB})
sub.insert(1, 'Y', 0)

test_raw = pd.read_csv(PATH_DATA_RAW + "evaluation_public.csv")
del test_raw['Y']
del test_raw['PROB']

test_raw = test_raw.merge(sub,how='left',on='EID')

test_raw['Y'][test_raw['PROB'] >= f1_th] = 1
test_raw.to_csv('../submit/baseline_publish.csv',index=False, encoding = 'utf-8')
'''
