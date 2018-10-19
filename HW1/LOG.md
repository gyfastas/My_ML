# My_ML

## HW1 Titanic 

### 想法

1. 使用python sklearn
2. 使用stacking 和 ensembling技术，通过不同模型来增强鲁棒性



## 使用的工具包

### Pandas

### seaborn

### Matplotlib

### sklearn



## 项目计划

第一阶段(10.15-10.20): 熟悉基本的工具，参考Kaggle上他人的Kernel完成数据的清洗和初步特征提取

第二阶段(10.20-10.25):特征工程，搭建简单机器学习模型(使用Python sklearn工具包)

第三阶段(10.25-10.29):模型改进与结果可视化；采用集成学习的stacking方法提升模型的鲁棒性，攥写报告



## Log

### 10.17

 配置Pycharm, Github, 下载数据集

 Pycharm: http://www.jetbrains.com/pycharm/ 专业版 社区版随意

 Github: 下载Github客户端，Clone此项目 https://github.com/gyfastas/My_ML

 在Pycharm中安装包:

 file-setting-project interpreter- 点击+号，找到对应的包安装即可

![1539741536660](D:\GIT\My_ML\HW1\1539741536660.png)

  

Goal:

1.参考Kaggle 上的相关kernel跑一遍代码，确保数据集能够打开和可视化

2.查看数据的缺失（清理数据），初步讨论数据的特征，明确预测任务

3.*简单清洗数据，获得能够用于训练的数据集



### 10.18

Goal: 与昨日相同

#### 任务说明(gyf)

<最终目标>: 对于测试集中的ID,预测其在该场比赛中的排名)

 针对第一次作业，提出以下简化任务的方法:

1.简化为二分类问题，吃鸡与否（难度较大，而且预计模型会预估得非常不准...） 

2.简化为二分类问题，成绩是否良好（单排前10，双排前7，四排前4） （难度稍小）

3.简化为多分类问题，在2的基础上加入几个等级（良好，中等，落地成盒等...）

4.按照原本任务要求（回归问题） 给出排名值（从0到1）

 

#### 数据观察与思考

https://www.kaggle.com/lhideki/pubg-finish-placement-prediction/data

- **DBNOs** - Number of enemy players knocked.
- **assists** - Number of enemy players this player damaged that were killed by teammates.
- **boosts** - Number of boost items used.
- **damageDealt** - Total damage dealt. Note: Self inflicted damage is subtracted.
- **headshotKills** - Number of enemy players killed with headshots.
- **heals** - Number of healing items used.
- **killPlace** - Ranking in match of number of enemy players killed.
- **killPoints** - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.)
- **killStreaks** - Max number of enemy players killed in a short amount of time.
- **kills** - Number of enemy players killed.
- **longestKill** - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
- **matchId** - Integer ID to identify match. There are no matches that are in both the training and testing set.
- **revives** - Number of times this player revived teammates.
- **rideDistance** - Total distance traveled in vehicles measured in meters.
- **roadKills** - Number of kills while in a vehicle.
- **swimDistance** - Total distance traveled by swimming measured in meters.
- **teamKills** - Number of times this player killed a teammate.
- **vehicleDestroys** - Number of vehicles destroyed.
- **walkDistance** - Total distance traveled on foot measured in meters.
- **weaponsAcquired** - Number of weapons picked up.
- **winPoints** - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.)
- **groupId** - Integer ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
- **numGroups** - Number of groups we have data for in the match.
- **maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
- **winPlacePerc** - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

通过竞赛给出的数据说明，为简化任务，可以值观地提取几个重要的特征：

1.killPoints - 这可以说是杀人榜排名，一般我们可以认为杀敌数越多越容易获得胜利

2.kills：kills与killPoints相关性较大，需要处理

3.damageDealt: 造成的伤害量，这个与kills, killPoints等相关性较高，需要处理

4.DBNOS: 击倒的玩家数量，这个跟kills也有较大的相关性...

- **headshotKills** - Number of enemy players killed with headshots. 爆头数从一定程度上说明玩家水平高低，但是不够直观，要么去除要么作为较弱特征
- **heals** - Number of healing items used 治疗用品的数量衡量参战次数（没打赢也不可能用药...）比较重要
- **boosts**-饮料和止痛药使用的次数，也能从一定程度上刻画生存的时间

5.Distances  = Walk + Ride + Swim Distances ： 把三个距离加和起来作为一个特征: 移动距离

6.winPoints: 胜利排名...这个当然重要了....



问题:

对于winPoints这个特征，如果这些是匹配的比赛，winPoints高的和winPoints高的在同一场就无法作为一个良好的特征了(killPoints 同理)



第一次去除的特征:

1.**roadKills** - Number of kills while in a vehicle  弱特征，没有实际作用

2.

- **teamKills** - Number of times this player killed a teammate.
- **vehicleDestroys** - Number of vehicles destroyed  弱特征，没有实际作用

3.

- **assists** - Number of enemy players this player damaged that were killed by teammates. 助攻数特征较弱
- **weaponsAcquired** - Number of weapons picked up. 没什么卵用

**maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.  没什么卵用，基本上是拿来判别这场比赛的性质（单排 双排 四排）



#### 从Kernel中得到的信息

https://www.kaggle.com/rejasupotaro/effective-feature-engineering 此Kernel给出了有效的特征工程方法，从中提取到了一些有用的信息:

Killplace 相当重要(通过feature_importance观察而至)

https://www.kaggle.com/deffro/eda-is-fun

需要对kills damage等进行归一化处理



#### 模型选择

LinearRegression:

