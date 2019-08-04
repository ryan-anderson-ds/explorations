# -*- coding: utf-8 -*-
"""
Created on Sun Aug 4 10:54:52 2019
@author: rian-van-den-ander
"""

import pandas as pd
from encode_json_column import encode_json_column

# Importing the dataset
dataset = pd.read_csv('tmdb_5000_movies.csv')
dataset_credits = pd.read_csv('tmdb_5000_credits.csv')
dataset = pd.concat([dataset, dataset_credits], axis=1)

dataset = encode_json_column(dataset, 22,"name", 500, 1)

y = dataset.iloc[:, 18].values #12 for revenue, 18 for rating
X = dataset.iloc[:, 23:].values
X_names = dataset.columns[23:].values

from xgboost import XGBRegressor
regressor = XGBRegressor(colsample_bytree= 0.6, gamma= 0.7, max_depth= 4, min_child_weight= 5,
                         subsample = 0.8, objective='reg:squarederror')
regressor.fit(X, y)

importances = {}

count = 0
for feature_importance in regressor.feature_importances_:
    if feature_importance > 0.002:
        feature_name = X_names[count]
        importances[feature_name] = feature_importance
    count+=1
    
import operator
sorted_importances = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)
