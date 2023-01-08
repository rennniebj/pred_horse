import os.path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from datetime import timedelta 
from datetime import datetime

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler

import shap
from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier, Pool

from hyperopt import fmin, hp, tpe

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



def Load_Data(hist_r):
  data=pd.DataFrame()
  for hist in hist_r:
      dfname=pd.ExcelFile(hist)
      sn=dfname.sheet_names
      df=pd.read_excel(hist)
      if len(sn)>1:
          for i in sn[1:]:
              temp_df=pd.read_excel(hist,sheet_name=i)
              df=pd.concat([df,temp_df])
      df.reset_index(drop=True, inplace=True)
      data=pd.concat([data,df])
  return data

def get_info(dff):
  All_hist=pd.DataFrame()
  Horse_info=[]
  for i in dff.Horse.unique():
      Df_dict={}
      a=dff.loc[dff.Horse==i]
      Df_dict['Horse']=i
      a_sort=a.sort_values('Date.1')
      a_sort['Sectional_range']=a_sort['Sectional'].expanding().max()-a_sort['Sectional'].expanding().min()
      a_sort['Sectional_median']=a_sort['Sectional'].expanding().median()
      a_sort['Previous_Sectional']=a_sort['Sectional'].shift(1)
      a_sort['MPS_range']=a_sort['MPS'].expanding().max()-a_sort['MPS'].expanding().min()
      a_sort['MPS_median']=a_sort['MPS'].expanding().median()
      a_sort['Previous_MPS']=a_sort['MPS'].shift(1)

      Df_dict['Sectional_range']=a_sort['Sectional'].max()-a_sort['Sectional'].min()
      Df_dict['Previous_Sectional']=a_sort['Sectional'].values[-1]
      Df_dict['Sectional_median']=a_sort['Sectional'].median()

      Df_dict['MPS_range']=a_sort['MPS'].max()-a_sort['MPS'].min()
      Df_dict['Previous_MPS']=a_sort['MPS'].values[-1]
      Df_dict['MPS_median']=a_sort['MPS'].median()

      Df_dict['Age']=a_sort['Age'].median()
      Horse_info.append(Df_dict)
      All_hist=pd.concat([All_hist,a_sort])
  return All_hist,Horse_info

def Days_last_rest(df):
    restdays=[]
    for i,j in df.iterrows():
        try:
            pos=j.LastTen.rfind('x')
            if pos != -1:
                lastrestday=len(j.LastTen)-pos
                restdays.append(lastrestday)
            else:
                lastrestday=len(j.LastTen)
                restdays.append(lastrestday)
        except:
            lastrestday=np.nan
            restdays.append(lastrestday)
    df['lastrestday']=restdays
    return df

def handi(df):
    handicaped=[]
    for i,j in df.iterrows():
        pos=j.ShortName.find('HCP')
        if pos != -1:
            handicaped.append(1)
        else:
            handicaped.append(0)
    df['handicaped']=handicaped
    return df

def dis_class(df):
    type_race=[]
    for i,j in df.iterrows():
        if j.Distance>= 400 and j.Distance<= 1300:
            type_race.append('Sprint')
        else:
            type_race.append('distance')
    df['dis_class']=type_race
    return df

def dis_finsh(df):
    type_top=[]
    for i,j in df.iterrows():
        if j.Finish>= 0 and j.Finish<= 3:
            type_top.append('Top_3')
        else:
            type_top.append('Below 11th')
    df['rank']=type_top
    return df

    # Get the current date and time
now = datetime.now()
def days_since(date):
  delta = now - date
  return delta.days
