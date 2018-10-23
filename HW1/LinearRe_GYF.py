#This is the LinearRegression part of our project
#
#Author: 郭远帆
#rev.0 2018.10.28
import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import FeatureEng1
#读取数据
train = pd.read_csv('./datas/train.csv')
test = pd.read_csv('./datas/test.csv')


train = FeatureEng1.FeatureSelect(train)
#完成特征提取,采用线性模型fit并得到误差
#预测列
Y_Column = ['winPlacePerc']

cols_to_fit = [col for col in train.columns if col not in Y_Column]
X_Train = train[cols_to_fit]
Y_Train = train[Y_Column]

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Train,Y_Train,test_size= 0.25,random_state=0)
#训练线性模型
model = LinearRegression()
model.fit(X_Train,Y_Train)

#预测
y_Pred = model.predict(X_Test)
M = mean_absolute_error(Y_Test,y_Pred)
print(M)



