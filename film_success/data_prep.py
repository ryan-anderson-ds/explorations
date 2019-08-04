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

# Easiest to get right to dropping NaN, because there are several rows of bad data which can't 
# be used by the algorithm. Since I'm working with around 5000 rows to start, dropping around 50 NaNs is not a problem.

# meaning out 0 budgets
dataset['budget']=dataset['budget'].replace(0,dataset['budget'].mean())

# ---- DATA PREPARATION -----

X = dataset.iloc[:, :].values
y = dataset.iloc[:, 18].values #12 is revenue, 18 is rating

# picking independent variables
X = X[:,[0,1,4,9,11,13,14,22]]

# Removing zero revenues from the data
y_removed = []
X_removed = []
for l in range(0,len(y)):
    if y[l] !=0:
        y_removed.append(y[l])
        X_removed.append(X[l])
y = np.array(y_removed)
X = np.array(X_removed)

# converting film date to day of year. i've already adjusted for year through inflation
# i am arguably losing the 'year' which might be slightly correlated with film success
for l in range(0,len(y)):
    film_date = X[l,4]
    try:
        datetime_object = datetime.strptime(film_date, '%Y-%m-%d')
        X[l,4] = datetime_object.timetuple().tm_yday
    except:
        X[l,4] = 0

# after some basic processing, encoding the 
dataset =  pd.DataFrame(X)

# encoding genres. im not using ID because there'd be an overlap with other ids
# All genres
dataset = encode_json_column(dataset, 1,"name")

# encoding keywords
# limiting to 100 codes, and removing anything not within those 100
# yes, it is column 1 now, since last column 1 was removed by the encoder
dataset = encode_json_column(dataset, 1, "name", 100, 1)

# encoding production companies.
# limiting to 100 codes, and removing anything not within those 50

dataset = encode_json_column(dataset, 1,"name", 100, 1)

# encoding spoken languages - this has now become column 3
# encoding all spoken languages
dataset = encode_json_column(dataset, 3,"iso_639_1")


# encoding cast - this has now become column 3
# encoding 'just' top 500 cast
dataset = encode_json_column(dataset, 3,"name", 500, 1)

#saving to CSVs as a checkpoint
dataset.to_csv(r'Encoded_X.csv')
dataset_y = pd.DataFrame(y)
dataset_y.to_csv(r'Encoded_y.csv')