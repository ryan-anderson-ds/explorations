# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 06:46:53 2019
@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd

#Loading data from preprocessed CSVs
dataset_X_reimported = pd.read_csv('Encoded_X.csv')
dataset_y_reimported = pd.read_csv('Encoded_y - rating.csv')
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here

X = dataset_reimported.iloc[:, 1:-2].values
y = dataset_reimported.iloc[:, -1].values

# Removing zero revenues from the data
# I did this now in data prep because I realised it later, and data prep takes a good 30 minutes. 
y_removed = []
X_removed = []
for l in range(0,len(y)):
    if y[l] !=0:
        y_removed.append(y[l])
        X_removed.append(X[l])
X = np.array(X_removed)
y = np.array(y_removed)

# Splitting the dataset into the Training set and Test set
# I have a fairly large dataset of +- 4000 entries, so I'm going with 10% test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

#This regressor was picked with gridsearch over many parameters - took 4 hours
from xgboost import XGBRegressor
regressor = XGBRegressor(colsample_bytree= 0.6, gamma= 0.7, max_depth= 4, min_child_weight= 5,
                         subsample = 0.8, objective='reg:squarederror')
regressor.fit(X, y)

y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred) 
