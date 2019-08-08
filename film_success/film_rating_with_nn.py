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
from keras.callbacks import ModelCheckpoint

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
nn = Sequential()

"""
DATA: 10955 columns. 3000 rows. Essentially ALL cast, ALL crew, ALL keywords, and then the normal dataset data

"""


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
nn.add(Dense(units = 5000, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 10955))

"""
hidden layers: 
-----
- relu is most used 
- sigmoid again was VERY bad
https://blog.paperspace.com/vanishing-gradients-activation-function/ says elu is best of both worlds

"""
nn.add(Dense(units = 2500, kernel_initializer = 'uniform', activation = 'elu', input_dim = 5000))
nn.add(Dense(units = 1200, kernel_initializer = 'uniform', activation = 'elu', input_dim = 2500))
nn.add(Dense(units = 600, kernel_initializer = 'uniform', activation = 'elu', input_dim = 1200))
nn.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'elu', input_dim = 600))
nn.add(Dense(units = 150, kernel_initializer = 'uniform', activation = 'elu', input_dim = 300))
nn.add(Dense(units = 75, kernel_initializer = 'uniform', activation = 'elu', input_dim = 150))
nn.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'elu', input_dim = 75))
nn.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'elu', input_dim = 40))
nn.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'elu', input_dim = 20))
nn.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'elu', input_dim = 10))
nn.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'elu', input_dim = 5))

# A 1-10 output layer
nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear' ))

#for mean squared optimisation
nn.compile(optimizer = 'adam', loss = 'mse', metrics=['mse', 'mean_absolute_error'])
""" 
Loss:
-----
mse is default, and generally you dont stray from it.
 for regression, you can also go with mean_squared_logarithmic_error for 
ballsier guessing,
or mean_absolute_error when you have many outliers

Optimizer:
-----
Adam is the normally used ones
I got better results from adadelta, across the board (with different loss functions, activations) - but i think that's because it made
the model converge on an average 6.1 for all

"""

"""
Defining a checkpoint of best result, and fitting

"""
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

nn.fit(X_train, y_train, batch_size = 20, epochs = 500)

# Predicting the Test set results
y_pred = nn.predict(X_test)
score = r2_score(y_test, y_pred) 

"""
# LOADING THE BEST MODEL
# Load weights file of the best model :
weights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint 
nn.load_weights(weights_file) # load it
nn.compile(loss='mse', optimizer='adam', metrics=['mse','mean_absolute_error'])

#THEN TEST AGAIN

y_pred = nn.predict(X_test)
score = r2_score(y_test, y_pred) 
"""

