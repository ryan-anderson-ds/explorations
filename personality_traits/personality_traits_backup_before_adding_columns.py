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
from sklearn.model_selection import *

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

# Averaging out responses for each personality letter
# unfortunately B is a twit and has 13 questions. There goes my solution elegance

"""

X_new = []

for row in X:
    newrow = []
    for trait in np.arange(0,16,1):
        if(trait==0): # B has 13 answers for some reason
            newrow.append(round(np.mean([row[10*trait],
                     row[10*trait + 1],
                     row[10*trait + 2],
                     row[10*trait + 3],
                     row[10*trait + 4],
                     row[10*trait + 5],
                     row[10*trait + 6],
                     row[10*trait + 7],
                     row[10*trait + 8],
                     row[10*trait + 9],
                     row[10*trait + 10],
                     row[10*trait + 11],
                     row[10*trait + 12]]),2))
        elif(trait==1): # B has 13 answers for some reason
            newrow.append(round(np.mean([row[16*trait],
                     row[16*trait + 1],
                     row[16*trait + 2],
                     row[16*trait + 3],
                     row[16*trait + 4],
                     row[16*trait + 5],
                     row[16*trait + 6],
                     row[16*trait + 7],
                     row[16*trait + 8],
                     row[16*trait + 9],
                     row[16*trait + 10],
                     row[16*trait + 11],
                     row[16*trait + 12]]),2))
        else: # so for each next trait, we must add 3 counts on to account for B's greediness
            newrow.append(round(np.mean([row[10*trait + 3],
             row[10*trait + 1 + 3],
             row[10*trait + 2 + 3],
             row[10*trait + 3 + 3],
             row[10*trait + 4 + 3],
             row[10*trait + 5 + 3],
             row[10*trait + 6],
             row[10*trait + 7 + 3],
             row[10*trait + 8 + 3],
             row[10*trait + 9 + 3]]),2))
    newrow = np.array(newrow)
    X_new.append(newrow)
X = np.array(X_new)

"""

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

clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.8, gamma=2.2,
       learning_rate=0.12, max_delta_step=0, max_depth=9,
       min_child_weight=25, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=0.88, verbosity=1)

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




