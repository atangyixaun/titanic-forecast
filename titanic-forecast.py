import pandas as pd
import numpy as np
from sklearn.liner_model import LinerRegression  #线性回归
from sklearn.cross_validation import KFold  #交叉验证
from sklearn import cross_validation
from sklearn.liner_model import LogisticRegression

titanic= pd.read_csv("titanic_train.csv")  #文件脚本存放在同一文件夹下
titanic.decsribe()
#补充缺失值,中位数值
titanic["age"] = titanic["age"].fillna(titanic["age"].median())
titanic["Embarked"] =titanic["Embarked"].fillna("S") 
#把文本值转换为数值型
print(titanic["sex"].unique())  #性别类型，列表显示
titanic.loc[titanic["sex"]=="male","sex"] =0 
titanic.loc[titanic["sex"]=="female","sex"] =1  #参数为行标签和列标签
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2

#测试集数据整理
titanic_test = pandas.read_csv("test.csv")
titanic.loc[titanic["sex"]=="male","sex"] =0
titanic.loc[titanic["sex"]=="female","sex"] =1
titanic["age"] = titanic["age"].fillna(titanic["age"].median())
titanic["Embarked"] =titanic["Embarked"].fillna("S")
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2

predicators = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
alg_liner = LinerRegression()
kf = KFold(titanic.shape[0],n_folds=3,random_state=1) #行数，次数
predictions =[]
for train,test in kf:
	train_data = (titanic[predicators].iloc[train])#选取特定列的内容
	train_target = titanic["Survived"].iloc[train]
	alg_liner.fit(train_data,train_target)  #训练训练集
	test_data = alg_liner.predict(titanic[predicators].iloc[test])
	predictions.append(test_data)  #为值
	
predictions[predictions>0.5]=1
predictions[predictions<=0.5]=0
accuracy = sum(predictions[predictions==titanic["Survived"]])/len(predictions)
#准确率偏低


alg_logi = LogisticRegression(random_state =1)
scores = cross_validation.cross_val_score(alg_logi,titanic[predictors],titanic["Survived"],cv=3)  #scores 为一组数据，代表每一次切分的准确率
print(scores.mean())

#随机森林 测试
from sklearn.ensemble import RandomForestClassifier  
predicators = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
alg_RF = RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=3)
kf = cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores = cross_validation.cross_val_score(alg_RF,titanic[predictors],titanic["Survived"],cv=kf)  #准确率
print(scores.mean())

#由于各模型的准确率都不高，继续挖掘特征值
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x: len(x))
import re
def get_title(name):
	title_search = re.search("(A-Za-z)\.",name)
	if title_search:
		return title_search.group(1)
	#return ""
	
titles = titanic["Name"].apply(get_title)
title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6}
for k,v in title_mapping.items():
	titles[titles==k]=v
	
#加入原数据
titanic["Title"]=titles
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pylot as plt
predicators = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title","NameLength"]
#评估特征重要性
selector = SelectKBest(f_classif,k=5)
selector.fit(titanic[predicators],titanic["Survived"])
#获得原始p-value值，并转化为分数
scores = -np.log10(selector.pvalues)

#绘图
plt.bar(range(len(predicators)),scores)
plt.xticks(range(len(predicators),predicators,rotation='vertical')
plt.show()
#根据图谱选择最重要的四个
predicators = ['Pclass','Sex','Fare','Title']
alg = RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=3)
kf = cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)  #准确率
print(scores.mean())