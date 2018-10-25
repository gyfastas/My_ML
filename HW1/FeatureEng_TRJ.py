'''
Machine Learning HW1
FeatureEng for Decision Tree

Author: 唐荣俊
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#去除没有用的特征，构建新特征
def FeatureSelect1(df):
    #构建参加人数的特征，并且去除较少人参加的部分
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df = df[df['playersJoined']>49]
    #合并击倒和击杀数以及其他参数
    df['kills'] = df['kills']+0.7*df['DBNOs']+0.5*df['roadKills']
    #由于每一场人数不同，需要先对kills和damageDealt进行归一化处理
    df['killsNorm'] = df['kills'] * ((100 - df['playersJoined']) / 100 + 1)
    df['damageDealtNorm'] = df['damageDealt'] * ((100 - df['playersJoined']) / 100 + 1)
    #df['distance'] = df['walkDistance']+df['swimDistance']+df['rideDistance']
    #drop掉不用的特征
    cols_to_drop = ['groupId','numGroups','roadKills', 'matchId','teamKills', 'vehicleDestroys', 'maxPlace','winPoints','kills','damageDealt','DBNOs','killPoints','longestKill']

    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    df = df[cols_to_fit]

    return df








