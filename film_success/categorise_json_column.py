#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:09:17 2019
@author: rian-van-den-ander
"""

import pandas as pd
import json

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except:
        return False
    return True

def categorise_json_column(pandas_data_frame, json_column_index=0, json_column_categorical_name="id"):
    
    #input variables
    json_column_index=1
    json_column_categorical_name="id"
    
    X = pandas_data_frame.iloc[:, 0:20].values
    
        
    #keep track of whether a column has been encoded already, else we'd reset all the values to 0
    encodedcolumns = []
    count = 0
    
    #for each row in the data
    for row in X:
        
        if(is_json(row[json_column_index])): #some data is just not json. ignore
           
            
            for json_features in json.loads(row[json_column_index]):
                
                featureid = json_features[json_column_categorical_name] #pick the test id
                
                if featureid not in encodedcolumns:
                    encodedcolumns.append(featureid)
                    pandas_data_frame[featureid]=0
                   
                pandas_data_frame[featureid][count] = 1
    
        count+=1
    
    #drop the last column from dataframe to avoid the dummy variable trap
    pandas_data_frame = dataset.drop(encodedcolumns[len(encodedcolumns)-1], 1)
    
    #drop the original json column
    pandas_data_frame = dataset.drop(dataset.columns[json_column_index], 1)
    
    return pandas_data_frame


#TESTING IT:
# Importing the dataset
dataset = pd.read_csv('tmdb_5000_movies.csv')
json_column_index=1
json_column_categorical_name="id"

dataset = categorise_json_column(dataset,json_column_index,json_column_categorical_name)


#TODO: Make this an encoding library, accepting X and an index and returning new encoded X!
#Take away hardcoded row [1], make it a list of rows so it can do all
    #NB: IT CANT DO ALL AT ONCE BECAUSE THEY MAY HAVE SAME ID

#TODO: ARE ALL MY JSON COLUMNS "ID, X"? Should I just take the first feature? :/

#TODO: And then add this to writeup while it's still fresh

#TODO: AND THEN TEST THAT THIS WORKS 

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