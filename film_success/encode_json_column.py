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

def encode_json_column(pandas_data_frame, json_column_index=0, json_id_column="id"):
        
    X = pandas_data_frame.iloc[:, :].values
            
    #keep track of whether a column has been encoded already, else we'd reset all the values to 0
    encodedcolumns = []
    count = 0
    
    #for each row in the data
    for row in X:
        
        if(is_json(row[json_column_index])): #some data is just not json. ignore
            
            #for each feature in the json
            for json_features in json.loads(row[json_column_index]):
                
                #pick out its id (the json identifier you specifc in json_id_column)
                featureid = json_features[json_id_column]
                
                #if this id hasn't been seen yet, add it to the dataframe with default 0
                if featureid not in encodedcolumns:
                    encodedcolumns.append(featureid)
                    pandas_data_frame[featureid]=0
                   
                #else just set it to 1 here
                pandas_data_frame[featureid][count] = 1
    
        count+=1

    #drop the original json column
    pandas_data_frame = pandas_data_frame.drop(dataset.columns[json_column_index], 1)
    
    return pandas_data_frame