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
# Easiest to get right to dropping NaN, because there are several rows of bad data which can't 
# be used by the algorithm. Since I'm working with around 5000 rows to start, dropping around 50 NaNs is not a problem.

# ---- DATA PREPARATION -----

X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 12].values #revenue

# setting y to profit of film
y = y - X[:,0]

# picking independent variables
X = X[:,[0,1,11,13,14]]
# TODO: ADD BACK PRODUCTION COMPANIES (9), KEYWORDS AND ACTORS. BUT LIMIT ENCODING TO 20 EACH

# Removing zero revenues from the data
y_removed = []
X_removed = []
for l in range(0,len(y)):
    if y[l] !=0:
        y_removed.append(y[l])
        X_removed.append(X[l])
y = np.array(y_removed)
X = np.array(X_removed)

# Ajusting inflation to 2019 at average inflation - 3.22%
avg_inflation = 1.01322
year_now = 2019
for l in range(0,len(y)):
    film_year = int(X[l,2][0:4]) # TODO: THIS WILL CHANGE IF I ADD BACK PRODUCTION COMPANIES, KEYWORDS AND THEY'RE BEFORE 11
    y[l] = y[l]*(avg_inflation ** (year_now-film_year))

# converting film date to day of year. i've already adjusted for year through inflation
# i am arguably losing the 'year' which might be slightly correlated with film success
for l in range(0,len(y)):
    film_date = X[l,2]  # TODO: THIS WILL CHANGE IF I ADD BACK PRODUCTION COMPANIES, KEYWORDS AND THEY'RE BEFORE 11
    try:
        datetime_object = datetime.strptime(film_date, '%Y-%m-%d')
        X[l,2] = datetime_object.timetuple().tm_yday
    except:
        X[l,2] = 0

# after some basic processing, encoding the 
dataset =  pd.DataFrame(X)

# encoding genres
dataset = encode_json_column(dataset, 1,"id")

# encoding spoken languages - this has now become column 3
 # TODO: THIS WILL CHANGE IF I ADD BACK PRODUCTION COMPANIES, KEYWORDS AND THEY'RE BEFORE 11
 # TODO: MUST THEN TEST AGAIN
dataset = encode_json_column(dataset, 3,"iso_639_1")

# this results in 108 independent variables from 6, in a dataset of size 20. it's a lot.
# Saving to CSVs as the above takes quite a while, and I'd like to use this as a checkpoint:
dataset.to_csv(r'Encoded_X.csv')
dataset_y = pd.DataFrame(y)
dataset_y.to_csv(r'Encoded_y.csv')

# --- CHECKPOINT ----
# reloading back from CSVs, since previous step can take a long time

dataset_X_reimported = pd.read_csv('Encoded_X.csv') #this fucks out a bit. index is included, so i need to skip that
dataset_y_reimported = pd.read_csv('Encoded_y.csv') #this fucks out a bit. index is included, so i need to skip that
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here

X = dataset_reimported.iloc[:, 1:].values
y = dataset_y_reimported.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# I have a fairly large dataset of +- 4000 entries, so I'm going with 10% test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

np.isnan(X_train).any()

from sklearn.svm import SVR
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(X_train, y_train)

y_pred = svr_regressor.predict(X_test)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

