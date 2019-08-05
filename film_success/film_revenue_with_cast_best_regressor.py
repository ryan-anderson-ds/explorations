# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 06:46:53 2019
@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading data from preprocessed CSVs
dataset_X_reimported = pd.read_csv('Encoded_X.csv')
dataset_y_reimported = pd.read_csv('Encoded_y - revenue.csv')
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here

X = dataset_reimported.iloc[:, 1:-2].values
y = dataset_reimported.iloc[:, -1].values

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

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured revenue')
ax.set_ylabel('Predicted revenue')
plt.title('Measured versus predicted revenue')
plt.ylim((50000000, 300000000))   # set the ylim to bottom, top
plt.xlim(50000000, 300000000)     # set the ylim to bottom, top
plt.show()