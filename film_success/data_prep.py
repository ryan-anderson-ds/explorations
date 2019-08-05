# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 06:46:53 2019
@author: rian-van-den-ander
"""

# ---- IMPORTS AND INPUTS -----

import numpy as np
import pandas as pd
from encode_json_column import encode_json_column
from datetime import datetime

# Importing the dataset
dataset = pd.read_csv('tmdb_5000_movies.csv')
dataset_credits = pd.read_csv('tmdb_5000_credits.csv')
dataset = pd.concat([dataset, dataset_credits], axis=1)

# meaning out 0 budgets - there are a lot, so this is better than removing the rows
dataset['budget']=dataset['budget'].replace(0,dataset['budget'].mean())

X = dataset.iloc[:, :].values
y = dataset.iloc[:, 12].values  #12 is revenue, 18 is rating - I changed this to make different y source files for next step 

# picking independent variables
X = X[:,[0,1,4,9,11,13,14,22]]

# Removing zero REVENUES from the data - revenue is super important
# I could (and have) adjusted for inflation, but it made scant difference to model performance
y_removed = []
X_removed = []
for l in range(0,len(y)):
    if y[l] !=0:
        y_removed.append(y[l])
        X_removed.append(X[l])
y = np.array(y_removed)
X = np.array(X_removed)

# Ajusting inflation to 2019 at average inflation - 3.22%
# do this only if using revenue (12 y index)
avg_inflation = 1.01322
year_now = 2019
for l in range(0,len(y)):
    try:
        film_year = int(X[l,4][0:4])
        y[l] = y[l]*(avg_inflation ** (year_now-film_year))
    except:
        X[l,4] = 0

# converting film date to day of year
# i am arguably losing the 'year' which might be slightly correlated with film success
# but that opens up a whole new can of worms about ratings and revenues by year
for l in range(0,len(y)):
    film_date = X[l,4]
    try:
        datetime_object = datetime.strptime(film_date, '%Y-%m-%d')
        X[l,4] = datetime_object.timetuple().tm_yday
    except:
        X[l,4] = 0

dataset =  pd.DataFrame(X)

# encoding genres. 
# using name because "id" overlaps with "id" in the next encoding, and so on
dataset = encode_json_column(dataset, 1,"name")

# encoding keywords
# limiting to 100 codes, and removing anything not within those 100
# yes, it is column 1 now, since last column 1 was removed by previous encoding
dataset = encode_json_column(dataset, 1, "name", 100, 1)

# encoding production companies.
# limiting to 100 codes, and removing anything not within those 100
dataset = encode_json_column(dataset, 1,"name", 100, 1)

# encoding all spoken languages
dataset = encode_json_column(dataset, 3,"iso_639_1")

# encoding cast
# encoding 'just' top 500 cast
dataset = encode_json_column(dataset, 3,"name", 500, 1)

#saving to CSVs as a checkpoint to be used in regressors
dataset.to_csv(r'Encoded_X.csv')
dataset_y = pd.DataFrame(y)
dataset_y.to_csv(r'Encoded_y.csv')