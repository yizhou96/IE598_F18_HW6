#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yizhouwang
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


iris = datasets.load_iris()
X = iris.data
y = iris.target
 
##1   
trainscore = []
testscore = []

for rs in range(1,11):  
    print('random state=',rs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rs)
  
    tree = DecisionTreeClassifier(criterion='gini', 
                                  max_depth=4, 
                                  random_state=1)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    trainscore.append(accuracy_score(y_train,y_train_pred))
    y_test_pred = tree.predict(X_test)
   
    print('train accuracy score: ',accuracy_score(y_train,y_train_pred))
    
    print('test accuracy score: ',accuracy_score(y_test,y_test_pred))
    testscore.append(accuracy_score(y_test,y_test_pred))

print("In-sample mean score: ",np.mean(trainscore))
print("in-sample standard deviation: ",np.std(trainscore))

print("Out-sample mean score: ",np.mean(testscore))
print("Out-ample standard deviation: ",np.std(testscore))

##2
accuracy = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

kfold = KFold(n_splits=10,random_state=1).split(X_train, y_train)
tree = DecisionTreeClassifier(criterion='gini', 
                                  max_depth=4, 
                                  random_state=1)
tree.fit(X_train,y_train)

accuracy = cross_val_score(estimator = tree , X = X_train , y = y_train, cv=10,n_jobs=1)
print("cross_val_score: ",accuracy)
print("mean scores: ",np.mean(accuracy))
print("standard deviation: ",np.std(accuracy),"\n")

print("My name is YizhouWang")
print("My NetID is yizhouw4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")