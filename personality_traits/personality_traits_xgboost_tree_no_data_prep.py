# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:14:00 2019

@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import *

dataset = pd.read_csv('personality_data.csv', header=0, sep='\t')

"""
Data cleansing
----------
"""

#dataset = dataset.dropna() #drop any null data


X = dataset.iloc[:,0:-6].values
y_age = dataset.iloc[:,-6].values
y_gender = dataset.iloc[:,-5].values
y_accuracy = dataset.iloc[:,-4].values
y_elapsed = dataset.iloc[:,-1].values



# split into training and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size = 0.1)

"""
Model
--------
"""


"""
# grid search - best params listed here. could still try increase learning rate, max depth, as the grid took the highest of those

xgb_model = xgb.XGBClassifier()

params = {
        'min_child_weight': [25],
        'gamma': [2.2],
        'subsample': [0.88],
        'colsample_bytree': [0.8],
        'max_depth': [9],
        'learning_rate': [0.12]
        }

clf = GridSearchCV(xgb_model, params, n_jobs=5,
                   scoring='roc_auc',
                   verbose=2, refit=True)

"""

clf = xgb.XGBClassifier()

clf.fit(X_train, y_gender_train)

# new y test and y_pred from best fit

# When using grid search, y_pred = clf.best_estimator_.predict(X_test)
y_pred = clf.predict(X_test)


# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_gender_test, y_pred)

# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf, X = X_train, y = y_gender_train, cv = 10) 
accuracies.mean()
accuracies.std()




