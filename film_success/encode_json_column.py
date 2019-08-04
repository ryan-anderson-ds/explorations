#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:09:17 2019
@author: rian-van-den-ander
"""


""" #example usage
dataset = pd.read_csv('tmdb_5000_movies.csv')
dataset = dataset[["revenue","genres"]]
dataset = dataset[0:2]

encode_json_column(dataset,1,"id",5,1)
"""

import pandas as pd
import json
import operator

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except:
        return False
    return True

def encode_json_column(pandas_data_frame, json_column_index=0, json_id_column="id", encodinglimit = 1000, remove_non_encoded = 1):
      
    X = pandas_data_frame.iloc[:, :].values

    #create a list of codes you want to take, based on encodinglimit
    all_encodedcolumns = {}
    
    for row in X:                    
        if(is_json(row[json_column_index])): #some data is just not json. ignore            
            #for each feature in the json
            for json_features in json.loads(row[json_column_index]):
                #pick out its id (the json identifier you specifc in json_id_column)
                featureid = json_features[json_id_column]                
                #if this id hasn't been seen yet, add it to the dataframe with default 0
                if featureid not in all_encodedcolumns:
                    all_encodedcolumns[featureid] = 1                   
                #else just set it to 1 here
                all_encodedcolumns[featureid] += 1

    top_encodedcolumns = sorted(all_encodedcolumns.items(), key=operator.itemgetter(1), reverse=True)
    
    if encodinglimit < len(top_encodedcolumns):
        top_encodedcolumns = top_encodedcolumns[:encodinglimit]        

    top_encodedcolumns = dict(top_encodedcolumns)

    #keep track of whether a column has been encoded into the dataframe already, else we'd reset all the values to 0
    df_encodedcolumns = []
    count = 0
    
    #for each row in the data
    for row in X:
        
        #keep track of whether this row can be kept or not, based on if it has an encoded value
        has_an_encoded_value = 0
        
        if(is_json(row[json_column_index])): #some data is just not json. ignore
            
            #for each feature in the json
            for json_features in json.loads(row[json_column_index]):
                
                #pick out its id (the json identifier you specifc in json_id_column)
                featureid = json_features[json_id_column]
                                
                if featureid in top_encodedcolumns:

                    #if this id hasn't been seen yet, add it to the dataframe with default 0
                    if featureid not in df_encodedcolumns:
                        df_encodedcolumns.append(featureid)
                        pandas_data_frame[featureid]=0
                   
                    #else just set it to 1 here
                    pandas_data_frame[featureid][count] = 1
                    
                    has_an_encoded_value = 1
    
        if has_an_encoded_value == 0 & remove_non_encoded == 1:
            pandas_data_frame.drop(pandas_data_frame.index[count])
        else:
          count+=1

    #drop the original json column
    pandas_data_frame = pandas_data_frame.drop(pandas_data_frame.columns[json_column_index], 1)
    
    return pandas_data_frame