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

from keras import backend as BK
def mapping_to_target_range( x, target_min=1, target_max=10 ) :
    x02 = BK.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min

dataset_X_reimported = pd.read_csv('Encoded_X.csv')
dataset_y_reimported = pd.read_csv('Encoded_y - rating.csv')
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna()

X = dataset_reimported.iloc[:, 1:-2].values
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


#TODO: FIND THAT CODE THAT GOES BACK AND PICKS THE BEST ONE AFTER MANY MANY RUNS

#applying rules of thumb:
#1. halve the dimensions each time
#2. use relu for hidden layers and input layers, linear depending on your output

"""
Input layer: 
-----
sigmoid / tanh: to quickly discard inputs that dont matter (since my XGBoost said only 160 inputs count)
I got better performance from tanh. The difference between these is supposedly about learning speed, though, so could try sigmoid again with more epochs

"""
classifier.add(Dense(units = 400, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 809))

"""
hidden layers: 
-----
- relu is most used 
- sigmoid again was VERY bad
https://blog.paperspace.com/vanishing-gradients-activation-function/ says elu is best of both worlds

"""
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'elu', input_dim = 400))
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'elu', input_dim = 200))
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'elu', input_dim = 100))
classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'elu', input_dim = 50))
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'elu', input_dim = 25))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'elu', input_dim = 12))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'elu', input_dim = 6))

# A 1-10 output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = mapping_to_target_range ))

#for mean squared optimisation
classifier.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')
""" 
Loss:
-----
mse is default, and generally you dont stray from it. for regression, you can also go with mean_squared_logarithmic_error for 
ballsier guessing, or mean_absolute_error when you have many outliers

Optimizer:
-----
Adam is the normally used ones
I got better results from adadelta, across the board (with different loss functions, activations) - but i think that's because it made
the model converge on an average 6.1 for all

"""

classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
score = r2_score(y_test, y_pred) 
