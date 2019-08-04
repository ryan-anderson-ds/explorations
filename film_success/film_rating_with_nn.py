# -*- coding: utf-8 -*-
"""
Created on Tue Aug 4 06:46:53 2019
@author: rian-van-den-ander

!pip install keras
!pip install tensorflow
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

dataset_X_reimported = pd.read_csv('Encoded_X.csv')
dataset_y_reimported = pd.read_csv('Encoded_y - revenue.csv')
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna()

X = dataset_reimported.iloc[:, 1:-1].values
y = dataset_reimported.iloc[:, -1].values

# Splitting into training and test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Feature Scaling - ABSOLUTELY has to be done for Neural Networks
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential #required to initialise neural network
from keras.layers import Dense #required to build the layers of the ANN
classifier = Sequential()

#applying rules of thumb:
#1. halve the dimensions each time
#2. use relu for hidden layers and input layers, linear depending on your output
classifier.add(Dense(output_dim = 400, init = 'uniform', activation = 'relu', input_dim = 810))
classifier.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu', input_dim = 400))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim = 200))
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 100))
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 100))
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu', input_dim = 50))
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 25))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 6))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

#for mean squared optimisation
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred) 
