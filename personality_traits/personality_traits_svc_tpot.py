
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 1 10:32:00 2019

@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd

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
Data engineering
----------
"""

# Adding interaction between personality items to X

X_new = []

for row in X:
    averaged_personality_traits = []
    for trait in np.arange(0,16,1):
        if(trait==0): # B has 13 answers for some reason
            averaged_personality_traits.append(round(np.mean([row[10*trait],
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
            averaged_personality_traits.append(round(np.mean([row[16*trait],
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
            averaged_personality_traits.append(round(np.mean([row[10*trait + 3],
             row[10*trait + 1 + 3],
             row[10*trait + 2 + 3],
             row[10*trait + 3 + 3],
             row[10*trait + 4 + 3],
             row[10*trait + 5 + 3],
             row[10*trait + 6],
             row[10*trait + 7 + 3],
             row[10*trait + 8 + 3],
             row[10*trait + 9 + 3]]),2))
        
    averaged_personality_traits = np.array(averaged_personality_traits)
    
    personality_trait_interactions = []
    
    outercount = 0
    for averaged_personality_trait_outer in averaged_personality_traits:
        innercount = 0
        for averaged_personality_trait_inner in averaged_personality_traits:
            if outercount != innercount:
                if averaged_personality_trait_inner == 0:
                    personality_trait_interactions.append(1)
                else:
                    personality_trait_interactions.append(round(averaged_personality_trait_outer/averaged_personality_trait_inner,2))
            innercount+=1        
        outercount+=1
    
    personality_trait_interactions = np.array(personality_trait_interactions)
    X_new.append(personality_trait_interactions)

X_new = np.array(X_new)
X = np.append(X, X_new, axis=1)

# split into training and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size = 0.1)

"""
Model
--------
"""

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()


tpot = TPOTClassifier(verbosity=2, n_jobs = -1)
tpot.fit(X_train, y_gender_train)
print(tpot.score(X_test, y_gender_test))
tpot.export('tpot_best_pipeline.py')


