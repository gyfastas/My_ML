'''
Machine Learning HW1
Test on LinearRegression

Author: 郭远帆
'''
import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import SGDRegressor
import FeatureEng_GYF
#读取数据
train = pd.read_csv('./datas/train.csv')
test = pd.read_csv('./datas/test.csv')


train = FeatureEng_GYF.FeatureSelect(train)

#完成特征提取,采用线性模型fit并得到误差
#预测列
Y_Column = ['winPlacePerc']

cols_to_fit = [col for col in train.columns if col not in Y_Column]
X_Train = train[cols_to_fit]
Y_Train = train[Y_Column]

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Train,Y_Train,test_size= 0.25,random_state=0)
#训练线性模型
transformer = Normalizer()
X_Train = transformer.fit_transform(X_Train)
Y_Train = transformer.fit_transform(Y_Train)
X_Test  = transformer.fit_transform(X_Test)
Y_Test  = transformer.fit_transform(Y_Test)
model = SGDRegressor(alpha=0.00001)
model.fit(X_Train,Y_Train)

#预测
y_Pred = model.predict(X_Test)
#计算平均绝对误差
M = mean_absolute_error(Y_Test,y_Pred)
print(M)



