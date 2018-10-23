'''
Machine Learning HW1
This is the SVR Part of our project

Author: GYF
'''
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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

cols_to_drop = ['Id','groupId','numGroups','matchId']
cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
df = df[cols_to_fit]
sample_num = 20000
df = df.sample(sample_num)
Y_Column = ['winPlacePerc']

cols_to_fit = [col for col in df.columns if col not in Y_Column]
X_Train = df[cols_to_fit]
Y_Train = df[Y_Column]

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Train,Y_Train,test_size= 0.25,random_state=0)

#训练线性模型
model = LinearSVR()
model.fit(X_Train,Y_Train)
model2 = NuSVR()
model2.fit(X_Train,Y_Train)

#预测
y_Pred = model.predict(X_Test)
y_Pred2 = model2.predict(X_Test)
M = mean_absolute_error(Y_Test,y_Pred)
M2 = mean_absolute_error(Y_Test,y_Pred2)
print(M,M2)

