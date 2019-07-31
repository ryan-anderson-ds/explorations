#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:09:17 2019
@author: rian-van-den-ander
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from collections import defaultdict

# Importing the dataset
dataset = pd.read_csv('tmdb_5000_movies.csv')
X = dataset.iloc[:, 0:20].values

#just working with one column
X = X[:,1]

#to convert just one row to json

encodedcolumns = defaultdict()


#TODO: THIS IDEA OF ADDING NEW COLUMN DOESNT SEEM TO BE WORKING
# check that they all seem to be going to row '94' of x
# actually, since this is running only on one row of X, it should probably 
# result in an X of around size 4, one with json and three with 1s

for json_features in json.loads(X[1]):
    
    json_features = json_features["id"] #pick the test id
    
    if json_features not in encodedcolumns:
        new_column_pos = len(X[1])
        
        X[new_column_pos] = 0
        encodedcolumns[json_features] = new_column_pos
       
    X[encodedcolumns[json_features]]=1
    
#TODO: when this is done, remove the last column to avoid encoding trap

#TODO: And then remove the Json clumn

#TODO: Now apply this to all the rows in X
    
#TODO and then within the bigger X, not just X starting with 1 column
    
#TODO and then for multiple json types.  (What did I mean by this?!)

#TODO: Make this an encoding library, accepting X and an index and returning new encoded X!

#TODO: And then add this to writeup while it's still fresh

"""

x2 = []

#adding in the new dimension
for row in X:
    x2.append([np.array(jsondata) for jsondata in row])
#transforming back into numpy array
x2 = np.array(x2)

#AN EASIER WAY TO DO IT MIGHT JUST BE TO GET THE LIST OF POSSIBLE GENRES AND
#LOOP THROUGH TO SEE IF THIS HAS THAT ONE
#BUT THIS IS NOT SUPER EXTENSABLE 

"""