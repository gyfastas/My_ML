'''
Machine Learning HW1
Test on DecisionTree

Author: 唐荣俊
'''
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import os
import seaborn
import pydotplus
import FeatureEng_TRJ
os.environ["PATH"] += os.pathsep + 'E:/graphviz-2.38/release/bin/'
#读取数据
train = pd.read_csv('./datas/train.csv')
test = pd.read_csv('./datas/test.csv')


train = Featurecatch.FeatureSelect1(train)
test  = Featurecatch.FeatureSelect1(test)

train = train.set_index(['Id'])
test = test.set_index(['Id'])
#预测列
Y_Column = ['winPlacePerc']

cols_to_fit = [col for col in train.columns if col not in Y_Column]
X_Train = train[cols_to_fit]
Y_Train = train[Y_Column]

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Train,Y_Train,test_size= 0.3,random_state=0)

#训练决策树模型
#X_Train,X_Test0,Y_Train,Y_Test0 = train_test_split(X_Train,Y_Train,test_size= 0.0005,random_state=0)

model = tree.DecisionTreeRegressor(max_depth=15,min_samples_split=10,min_samples_leaf=10)
model.fit(X_Train,Y_Train)
#iris = train
#with open("tree1.dot", 'w') as f:
#    f = tree.export_graphviz(model, out_file=f)
#预测
y_Pred = model.predict(X_Test)
M = mean_absolute_error(Y_Test,y_Pred)

print(M)

#plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
#plt.plot(Y_Test,y_Pred, color="cornflowerblue",
#         label="max_depth=10")
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
#plt.xlabel("Test")
#plt.ylabel("Pred")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()