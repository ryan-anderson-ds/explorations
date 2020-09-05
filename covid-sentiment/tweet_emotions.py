# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:33:40 2020
@author: Rian van den Ander
"""

import pandas as pd
import datetime
from datetime import datetime as dt
df = pd.read_json ('../../data/covid19_one_hundred_million_unique_tweets/tweet_ids/all_tweets_500_pd.jsonl', encoding='utf8', lines=True)

"""
take english tweets only
"""
df = df[df['lang']=='en']


"""
dirty logic to make one dataframe of tweets per day
"""
current_day = '0000-00-00'
dates = []
tweets = []
days_tweets = pd.DataFrame()
first_run = 1
for index, row in df.iterrows():
    date = str(row['created_at'])[:10]
    if(date > current_day):
        dates.append(date)
        current_day = date
        
        if(first_run == 1):
            first_run = 0
        else:
            tweets.append(days_tweets)
            
        days_tweets = pd.DataFrame()
        days_tweets = days_tweets.append(row)
    else:
        days_tweets = days_tweets.append(row)
tweets.append(days_tweets)        


"""
fit emotion detection model to reddit dataset
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

maxlen = 50
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


"""
now run on twitter per day
"""


















