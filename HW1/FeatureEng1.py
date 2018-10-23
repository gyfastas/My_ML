#This is the Feature Engneering part of our project
#Author: 郭远帆，罗乙然，唐荣俊
#rev.0 2018.10.18
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#去除没有用的特征，构建新特征
def FeatureSelect(df):
    #构建参加人数的特征，并且去除较少人参加的部分
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df = df[df['playersJoined']>49]
    #合并击倒和击杀数
    df['kills'] = df['kills']+df['DBNOs']
    #由于每一场人数不同，需要先对kills和damageDealt进行归一化处理
    df['killsNorm'] = df['kills'] * ((100 - df['playersJoined']) / 100 + 1)
    df['damageDealtNorm'] = df['damageDealt'] * ((100 - df['playersJoined']) / 100 + 1)
    df['distance'] = df['walkDistance']+df['swimDistance']+df['rideDistance']
    #drop掉不用的特征
    cols_to_drop = ['Id','groupId','numGroups','matchId','maxPlace','winPoints','damageDealt','killPoints']
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    df = df[cols_to_fit]

    return df








