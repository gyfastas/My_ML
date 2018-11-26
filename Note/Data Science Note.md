# Data Science Note 

## 工具包常用方法学习(1)

### Pandas

### Seaborn



用seaborn 画 直方图:

```
#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['longestKill'])
```



### Matplotlib

```
f, ax = plt.subplots(figsize=(8, 6))
df_train['revives'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()
```

value_counts可以直接数出个数，接着用plt来画也很方便



### SkLearn



### XGBoost





### 模型与调参方法学习



