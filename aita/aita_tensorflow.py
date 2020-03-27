#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:11:53 2020

@author: rian-van-den-ander
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential

df = pd.read_csv('../../data/aita/aita_clean.csv')

X = df['body'].values
y = df['is_asshole'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

vectorizer = CountVectorizer()
vectorizer.fit(X_train.astype('U'))

X_train = vectorizer.transform(X_train.astype('U'))
X_test  = vectorizer.transform(X_test.astype('U'))

