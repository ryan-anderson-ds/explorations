
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 1 10:32:00 2019

@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd

"""
Data
----
https://openpsychometrics.org/_rawdata/
Questions in Rosenberg Self-Esteem Scale format
47974 items

Scoring: "5 items are reverse scored. Give “Strongly Disagree” 1 point, “Disagree” 2 points,
“Agree” 3 points, and “Strongly Agree” 4 points. Sum scores for all ten items. Keep scores
on a continuous scale. Higher scores indicate higher self-esteem"

Q1. I feel that I am a person of worth, at least on an equal plane with others.	
Q2. I feel that I have a number of good qualities.	
Q3. All in all, I am inclined to feel that I am a failure. REVERSE SCORED
Q4. I am able to do things as well as most other people.	
Q5. I feel I do not have much to be proud of.	REVERSE SCORED
Q6. I take a positive attitude toward myself.	
Q7. On the whole, I am satisfied with myself.	
Q8. I wish I could have more respect for myself. REVERSE SCORED
Q9. I certainly feel useless at times. REVERSE SCORED
Q10. At times I think I am no good at all. REVERSE SCORED

"""

dataset = pd.read_csv('self_esteem_data.csv', header=0, sep='\t')


"""
Data cleansing and engineering
----------
"""

dataset = dataset.dropna() #drops 4 rows of null data
self_esteem = []

#defining a function to reverse the scores for the negatively scored questions
def reverse(x):
    x = int(x)
    x_out = 0
    if x==4:
        x_out = 1
    elif x==1:
        x_out = 4
    else:
        x_out = 4 - x
    return x_out
        

#Scoring self esteem per row
for index, row in dataset.iterrows():    
    score = int(row['Q1']) + int(row['Q2']) + int(reverse(row['Q3'])) + int(row['Q4']) + int(reverse(row['Q5'])) + int(row['Q6']) + int(row['Q7']) + int(reverse(row['Q8'])) + int(reverse(row['Q9'])) + int(reverse(row['Q10']))
    self_esteem.append(score)

dataset['self_esteem'] = self_esteem


# TODO: MAKE A DATAFRAME RATHER SINCE IT WILL HAVE 3 ROWS countries = dict()

for index, row in dataset.iterrows():    
    
    #todo: average self esteem per country
    #dataframe will have a 2d array in it, with average so far, and number of entries
    
    



#todo: put on map, clearly noting not enough data (<arbitrarily chosen as 50 data points)


