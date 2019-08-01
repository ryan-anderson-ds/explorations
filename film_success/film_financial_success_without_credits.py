# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 06:46:53 2019
@author: rian-van-den-ander
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('tmdb_5000_movies.csv')
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 12].values #revenue

#setting y to profit of film
y = y - X[:,0]

#picking independent variables
X = X[:,[0,1,9,11,13,14]]

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
    film_year = int(X[l,3][0:4])
    y[l] = y[l]*(avg_inflation ** (year_now-film_year))


#some independent variables need work

#TODO: 1 is genre list. must be split up and categorised
#TODO: 9 is production companies. must be split up and categorised
#TODO: 11 is release date. how do i include this? maybe a day of year
#TODO: 14 is spoken languages. must be split up and categorised
    #NB: This does not have ID, must categorise on another


# TODO: Encoding categorical data
    """
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
"""

# TODO: Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# TODO: Feature scaling? 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()