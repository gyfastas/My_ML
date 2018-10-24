'''
Machine Learning HW1
Test on GDLinear

Author: 郭远帆
'''
from sklearn.preprocessing import Normalizer
from GDLinear import GDLinear
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import FeatureEng_GYF
#读取数据
train = pd.read_csv('./datas/train.csv')
test = pd.read_csv('./datas/test.csv')


train = FeatureEng1.FeatureSelect(train)
#完成特征提取,采用线性模型fit并得到误差
#预测列
train = train.sample(1000)
transformer = Normalizer()
Y_Column = ['winPlacePerc']

cols_to_fit = [col for col in train.columns if col not in Y_Column]
X_Train = train[cols_to_fit]
Y_Train = train[Y_Column]
X_Train = transformer.fit_transform(X_Train)
Y_Train = transformer.fit_transform(Y_Train)
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Train,Y_Train,test_size= 0.25,random_state=0)
#训练线性模型
model = GDLinear(alpha = 0.00001,max_iter =5000)
model.fit(X_Train,Y_Train)

#预测
y_Pred = model.predict(X_Test)
#计算平均误差
M = mean_absolute_error(Y_Test,y_Pred)
print(M)



