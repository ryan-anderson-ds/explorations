#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:11:53 2020
@author: rian-van-den-ander
"""

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.models import Sequential
from keras import layers
from keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath,encoding='utf-8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

df = pd.read_csv('../../data/aita/aita_clean.csv')

""" Delete many not_asshole items to balance dataset """
df_asshole = df[df['is_asshole'] == 1]
df_not_asshole =df[df['is_asshole'] == 0]
df_not_asshole = df_not_asshole[:26511]
df = df_not_asshole.append(df_asshole)

X = df['body'].values
X= X.astype(str)
y = df['is_asshole'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(X_train)

#https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/91240
num_words=9000
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= num_words} # <= because tokenizer is 1 indexed
tokenizer.word_index[tokenizer.oov_token] = num_words + 1

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

maxlen = 1000
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

input_dim = X_train.shape[1] 

embedding_dim = 50
embedding_matrix = create_embedding_matrix('../../data/embedding/glove/glove.6B.50d.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
embedding_accuracy = nonzero_elements / vocab_size
print('embedding accuracy: ' + str(embedding_accuracy))

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(layers.Conv1D(256, 5, activation='relu'))
model.add(Dropout(0.3))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
opt = optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

res = model.fit(X_train, y_train, epochs=50, verbose=True, validation_data=(X_test, y_test), batch_size=100)

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['not_asshole','asshole']
print(classification_report(y_test, y_pred, target_names=target_names))

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))