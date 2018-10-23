'''
Machine Learning HW1
This is the SVR Part of our project

Author: GYF
'''
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('./datas/train.csv')
#我们对数据进行简单的预处理以去除异常数据
#构建参加人数的特征，并且去除较少人参加的部分
df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
df = df[df['playersJoined'] > 49]
#由于每一场人数不同，需要先对kills和damageDealt进行归一化处理
df['killsNorm'] = df['kills'] * ((100 - df['playersJoined']) / 100 + 1)
df['damageDealtNorm'] = df['damageDealt'] * ((100 - df['playersJoined']) / 100 + 1)

