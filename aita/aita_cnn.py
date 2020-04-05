#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:11:53 2020

@author: rian-van-den-ander
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


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

X = df['body'].values
X= X.astype(str)

y = df['is_asshole'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(X_train)

#there is a bug with num_words. you have to manually set the tokenizer word index like this
#https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/91240
#I've set it to 9000 because I start seeing many zero values after there (many emojis, misspellings, etc)
#but does this affect accuracy at all?
#No, it doesn't seem to really
#Howeverm  it does make the model train 3* faster
#But it does make embedding accuracy go from 51% to 95%, which is a good thing to prevent overtraining
num_words=9000
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= num_words} # <= because tokenizer is 1 indexed
tokenizer.word_index[tokenizer.oov_token] = num_words + 1

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

#500: accuracy 72.5%
#1000: 72.8%
#3000: 72.8%
maxlen = 1000
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

input_dim = X_train.shape[1] # set feature dimensions

#Using glove:
#50 dimensions: 72.8%
#100 dimensions: 72.54
#lower amount of dimensions not an option

#Glove: 51% embedding accuracy, 72.54% model accuracy
#Fasttext: 56%, 72.44
#fasttext: embedding_matrix = create_embedding_matrix('../../data/embedding/fasttext/wiki-news-300d-1M-subword.vec', tokenizer.word_index, embedding_dim)
embedding_dim = 50
embedding_matrix = create_embedding_matrix('../../data/embedding/glove/glove.6B.50d.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
embedding_accuracy = nonzero_elements / vocab_size
print('embedding accuracy: ' + str(embedding_accuracy))

#TODO: prevent such early overfitting. why is it as such?!

#using embeddings_regularizer to prevent overfitting:  https://towardsdatascience.com/preventing-deep-neural-network-from-overfitting-953458db800a
#regularizer with 0.1: model accuracy=73%. each model trains equally.
#regularizer with 0.3: 73%.
#so taking it out

#alternatively, try dropout: https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
#model accuracy=

#TODO: fiddle with the size of these layers. maybe overnight grid search.
model = Sequential()
#regularizer: model.add(layers.Embedding(vocab_size, embedding_dim, embeddings_regularizer=keras.regularizers.l2(l=0.3), input_length=maxlen, trainable=True))
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#batch size - 10:73%. 50: 73%. 100: 72.5%
res = model.fit(X_train, y_train, epochs=20, verbose=True, validation_data=(X_test, y_test), batch_size=50)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))