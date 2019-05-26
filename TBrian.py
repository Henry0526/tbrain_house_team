import pandas as pd
# from pandas.core import datetools
import numpy as np
import seaborn as sns;sns.set(color_codes=True)
import matplotlib.pyplot as plt 

import sklearn 
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2           # 類別 預測 類別
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif      # ANOVA
from sklearn import linear_model


from pandas.plotting import scatter_matrix

import statsmodels.api as sm


pd.set_option('display.max_rows', 50) 
pd.set_option('display.max_columns', 250)
pd.set_option('display.float_format', lambda x: f"{x:.6f}")  # 讓顯示出來的數字不要使用科學記號

np.set_printoptions(suppress=True)  # 讓顯示出來的數字不要使用科學記號

plt.rcParams['font.sans-serif']=['Monospace 11']

# 節省記憶體用的
train = pd.read_csv("./train.csv", header=0)
test  = pd.read_csv("./test.csv", header=0)
submit_test = pd.read_csv("./submit_test.csv", header=0)

ALL_COLUMN = train.columns.values

IMPORTANT_COLUMNS = [   
    'building_use', 
    'parking_area',                # 車位面積(轉)
    'txn_floor', 
    'land_area',                   # 土地面積(轉)
    'building_area',               # 建物面積(轉)
    'village_income_median', 
    'town_population_density',
    'city', 
    
    
    'town', 
    'village', 
    'town_population', 
    'town_area',
    
    'txn_dt', 
    'total_floor',
    
    'building_material',  
    'building_type', 
    'building_complete_dt', 
    
    'parking_way',                # 車位停放方式
    'parking_price',              # 車位價格(轉)
    
    'lat',                        # 緯度(轉)
    'lon',                        # 經度(轉)
    
    'doc_rate', 
    'master_rate', 
    'bachelor_rate',
    'jobschool_rate', 
    'highschool_rate', 
    'junior_rate', 
    'elementary_rate',
    'born_rate', 
    'death_rate', 
    'marriage_rate', 
    'divorce_rate', 
    
]

# 14 種周遭建築物代稱   補0是為了之後要哪一類就用那個數字就好了
NEIGHBOR_BUILDING_TYPE    = ['0', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV']
# 14種周遭建築物距離範圍
NEIGHBOR_BUILDING_DISTANT = ['50', '500', '1000', '5000', '10000']

NEIGHBOR_FORMAT = [
    "{}_{}",             # {1} 類建築物在方圓 {2} 公里內有多少
    "{}_index_{}",       # 方圓 {2} 公里內有無 {1} 類建築物
    "{}_MIN"             # 該房屋與最近的 {1} 類別之距離
]


NEIGHBOR_COLUMN = []
for i in NEIGHBOR_BUILDING_TYPE:
    NEIGHBOR_COLUMN.append(NEIGHBOR_FORMAT[2].format(i))
    for j in NEIGHBOR_BUILDING_DISTANT:
        NEIGHBOR_COLUMN.append(NEIGHBOR_FORMAT[0].format(i, j))
        NEIGHBOR_COLUMN.append(NEIGHBOR_FORMAT[1].format(i, j))
        

NEIGHBOR_INDEX_COLUMN = []
for t in NEIGHBOR_BUILDING_TYPE[1:]:
    for i in NEIGHBOR_BUILDING_DISTANT:
        NEIGHBOR_INDEX_COLUMN.append(NEIGHBOR_FORMAT[1].format(t, i))

# IMPORTANT_COLUMNS = IMPORTANT_COLUMNS + AREA_NEIGHBOR
# ==========================================================================================
# 要將型態轉為 -- dummy Variable (Catorgory)

CATORGORY_COLUMN = [
    "building_material",
    "city",
    "town",
    "building_type",
    "building_use",
    "parking_way",
]

CONTINOUS_COLUMN = [name for name in ALL_COLUMN if name not in CATORGORY_COLUMN and name not in NEIGHBOR_INDEX_COLUMN]
# ==========================================================================================
Y_COLUMN = "total_price"

# 有 NA 的值
# txn_floor
# parking_area
# village_income_median 
# parking_price 

# train.isnull().values.any()

# Train 補 Missing Value
train['txn_floor'] = train['txn_floor'].fillna(round(train.groupby(['building_use','building_type','city'])['txn_floor'].transform('mean')))
train['txn_floor'] = train['txn_floor'].fillna(round(train.groupby(['building_type','city'])['txn_floor'].transform('mean')))
train['txn_floor'] = train['txn_floor'].fillna(round(train.groupby(['city'])['txn_floor'].transform('mean')))
# train["txn_floor"] = train["txn_floor"].fillna(value = 0.0)

train["parking_area"] = train["parking_area"].fillna(value=0)

train['village_income_median'] = train['village_income_median'].fillna(round(train.groupby(['city','town','village'])['village_income_median'].transform('mean')))
train['village_income_median'] = train['village_income_median'].fillna(round(train.groupby(['city','town'])['village_income_median'].transform('mean')))
train['village_income_median'] = train['village_income_median'].fillna(round(train.groupby(['city'])['village_income_median'].transform('mean')))
# train["village_income_median"] = train["village_income_median"].fillna(train["village_income_median"].mean())

train["parking_price"] = train["parking_price"].fillna(value=0)

# Test 補 Missing Value
test['txn_floor'] = test['txn_floor'].fillna(round(test.groupby(['building_use','building_type','city'])['txn_floor'].transform('mean')))
test['txn_floor'] = test['txn_floor'].fillna(round(test.groupby(['building_type','city'])['txn_floor'].transform('mean')))
test['txn_floor'] = test['txn_floor'].fillna(round(test.groupby(['city'])['txn_floor'].transform('mean')))
# test["txn_floor"] = test["txn_floor"].fillna(value = 0.0)
test["parking_area"] = test["parking_area"].fillna(value=0)

test['village_income_median'] = test['village_income_median'].fillna(round(test.groupby(['city','town','village'])['village_income_median'].transform('mean')))
test['village_income_median'] = test['village_income_median'].fillna(round(test.groupby(['city','town'])['village_income_median'].transform('mean')))
test['village_income_median'] = test['village_income_median'].fillna(round(test.groupby(['city'])['village_income_median'].transform('mean')))
# test["village_income_median"] = test["village_income_median"].fillna(value=test["village_income_median"].mean())

test["parking_price"] = test["parking_price"].fillna(value=0)