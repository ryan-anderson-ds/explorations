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
from keras import layers

df = pd.read_csv('../../data/aita/aita_clean.csv')

X = df['body'].values
y = df['is_asshole'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

vectorizer = CountVectorizer()
vectorizer.fit(X_train.astype('U'))

X_train = vectorizer.transform(X_train.astype('U'))
X_test  = vectorizer.transform(X_test.astype('U'))

input_dim = X_train.shape[1] # set feature dimensions
model = Sequential()
model.add(layers.Dense(1000, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(200, input_dim=1000, activation='relu'))
model.add(layers.Dense(40, input_dim=200, activation='relu'))
model.add(layers.Dense(6, input_dim=40, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

res = model.fit(X_train, y_train, epochs=20, verbose=True, validation_data=(X_test, y_test), batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))