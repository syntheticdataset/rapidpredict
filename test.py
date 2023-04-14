from rapidpredict.supervised import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y= data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.25,random_state =123)

clf = rapidclassifier(verbose= 0,ignore_warnings=True, custom_metric=None)
# print("clf" ,clf)
models , predictions = clf.fit(X_train, X_test, y_train, y_test)



compareModels_bargraph(predictions["F1 Score"] ,models.index)
# 
compareModels_boxplot(predictions["F1 Score"] ,models.index)


plot_target(y)