# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:34:59 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:14:00 2019

@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

dataset = pd.read_csv('personality_data.csv', header=0, sep='\t')

"""
Data cleansing
----------
"""

dataset = dataset.dropna() #drop any null data
dataset = dataset[dataset.age < 100] # Removing bogus age
dataset = dataset[dataset.gender.isin([1,2])] # removing non specific genders
dataset = dataset[dataset.accuracy > np.percentile(dataset.accuracy,5)] # removing very low accuracies
dataset = dataset[dataset.accuracy <= 100] # Removing very high accuracies
dataset = dataset[dataset.elapsed <= 5000] # Removing very high accuracies
dataset = dataset[dataset.elapsed > 300] # Removing very high accuracies

X = dataset.iloc[:,0:-6].values
y_age = dataset.iloc[:,-6].values
y_gender = dataset.iloc[:,-5].values
y_accuracy = dataset.iloc[:,-4].values
y_elapsed = dataset.iloc[:,-1].values


"""
Exploratory Data Analysis
----------
"""

# Distribution of age
plt.hist(y_age)
plt.xlabel('Age');
plt.ylabel('Respondents');
plt.title('Respondent age')
plt.show()

# Distribution of gender
plt.hist(y_gender, bins=[1,2])
bins = np.arange(1, y_gender.max() + 1.5) - 0.5
fig, ax = plt.subplots()
_ = ax.hist(y_gender, bins)
ax.set_xticks(bins + 0.5)
plt.xlabel('Gender');
plt.ylabel('Respondents');
plt.title('Respondent gender')
plt.show()

# Distribution of accuracy
plt.hist(y_accuracy)
plt.xlabel('Accuracy');
plt.ylabel('Respondents');
plt.title('Respondent accuracy')
plt.show()

# Distribution of time elapsed
plt.hist(y_elapsed, bins=np.arange(0, 5000, 10))
plt.xlabel('Elapsed');
plt.ylabel('Respondents');
plt.title('Respondent survey time')
plt.show()


"""
Data preparation
----------
"""

# split into training and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size = 0.1)

"""
Model
--------
"""

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 50, 200]}]

# The best parameters are {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

"""
GridSearch parameters:
    - can't specify scoring for SVC, must be hinge
    - cross validation = 3 just to have 3 shots at each, in case it's lucky
"""
model = SVC(kernel='rbf',gamma=1e-3,C=10)
model.fit(X_train, y_gender_train)


# new y test and y_pred from best fit

# When using grid search, y_pred = clf.best_estimator_.predict(X_test)
y_pred = model.predict(X_test)


# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_gender_test, y_pred)





