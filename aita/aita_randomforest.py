#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:13:41 2020
@author: rian-van-den-ander
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('../../data/aita/aita_clean.csv')

X = df['body'].values
y = df['is_asshole'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

vectorizer = CountVectorizer()
vectorizer.fit(X_train.astype('U'))

X_train = vectorizer.transform(X_train.astype('U'))
X_test  = vectorizer.transform(X_test.astype('U'))

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score) #72%

n=20
feature_names = vectorizer.get_feature_names()
coefs_with_fns = sorted(zip(classifier.feature_importances_, feature_names))
coefs_with_fns = coefs_with_fns[-20:]
print(coefs_with_fns)