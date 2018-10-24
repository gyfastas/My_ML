from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import FeatureEng_LYR
#读取训练集
train = pd.read_csv('./datas/train.csv')
#对训练集中的数据
train = FeatureEng1.FeatureSelect(train)
#将预测目标与其他特征分开为X和Y
Y_Column = ['winPlacePerc']
cols_to_fit = [col for col in train.columns if col not in Y_Column]
X_Train = train[cols_to_fit]
Y_Train = train[Y_Column]
#将训练集按3:7分为测试集与训练集
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_Train,Y_Train,test_size= 0.3,random_state=0)
#设定模型的参数
regr = RandomForestRegressor(max_features=20,random_state=10,n_estimators=50,min_samples_leaf=50,min_samples_split=100,n_jobs=-1,oob_score=True)
#训练模型
regr.fit(X_Train,Y_Train)
#袋外得分
print(regr.oob_score_)

print(regr.feature_importances_)
#将模型应用于测试集
Y_Pred=regr.predict(X_Test)
print(mean_absolute_error(Y_Test,Y_Pred))
