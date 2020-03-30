#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:13:41 2020
@author: rian-van-den-ander
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('../../data/aita/aita_clean.csv')

X = df['body'].values
y = df['is_asshole'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

vectorizer = CountVectorizer()
vectorizer.fit(X_train.astype('U'))

X_train = vectorizer.transform(X_train.astype('U'))
X_test  = vectorizer.transform(X_test.astype('U'))

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score) #68%

n=20
feature_names = vectorizer.get_feature_names()
coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))