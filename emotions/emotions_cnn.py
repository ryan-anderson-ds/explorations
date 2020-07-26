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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.layers import Dropout
from keras.models import Sequential
from keras import layers
from keras import optimizers

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

df1 = pd.read_csv('../../data/emotions/goemotions_1.csv')
df2 = pd.read_csv('../../data/emotions/goemotions_2.csv')
df3 = pd.read_csv('../../data/emotions/goemotions_3.csv')

frames = [df1, df2, df3]

df = pd.concat(frames)

X = df['text'].values
X= X.astype(str)
y = df.iloc[:,9:].values

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

maxlen = 20
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

input_dim = X_train.shape[1] 

embedding_dim = 300
embedding_matrix = create_embedding_matrix('../../data/embedding/glove/glove.6B.300d.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
embedding_accuracy = nonzero_elements / vocab_size
print('embedding accuracy: ' + str(embedding_accuracy))

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(layers.Conv1D(256, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(28, activation='sigmoid'))
opt = optimizers.Adam(lr=0.0002)
model.compile(optimizer=opt, loss='binary_crossentropy')
model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
         ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
res = model.fit(X_train, y_train, epochs=15, verbose=True, callbacks=callbacks, validation_data=(X_test, y_test), batch_size=100)

y_pred = model.predict(X_test)

thresholds=[0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for val in thresholds:
    pred=y_pred.copy()
  
    pred[pred>=val]=1
    pred[pred<val]=0
  
    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f1 = f1_score(y_test, pred, average='micro')
   
    print("Threshold: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(val, precision, recall, f1))

column_names = list(df.columns[9:])
f1_scores = []
threshold = 0.25
for i in range(0,28):
    emotion_prediction = y_pred[:,i]
    emotion_prediction[emotion_prediction>=threshold]=1
    emotion_prediction[emotion_prediction<threshold]=0
    emotion_test = y_test[:,i]
    precision = precision_score(emotion_test, emotion_prediction)
    recall = recall_score(emotion_test, emotion_prediction)
    f1 = f1_score(emotion_test, emotion_prediction)
    f1_scores.append(f1)
    print("Emotion: {}, Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(column_names[i], precision, recall, f1))


import matplotlib.pyplot as plt
fig = plt.figure()
plt.bar(column_names,f1_scores)
plt.xticks(rotation=90)
plt.show()
