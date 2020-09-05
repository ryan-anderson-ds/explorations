# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:33:40 2020
@author: Rian van den Ander
"""

import pandas as pd
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
load my emotion detection model
"""

import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df_emotion_model = pd.read_csv('../../data/emotions/goemotions_1.csv')
emotion_names = list(df_emotion_model.columns[9:-1])
model = keras.models.load_model("model_file")

"""
run model on tweets
"""

average_emotions = []
    
for days_tweets in tweets:
    
    """
    Preprocess tweets
    """
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(days_tweets['full_text'])
    
    num_words=9000
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= num_words} # <= because tokenizer is 1 indexed
    tokenizer.word_index[tokenizer.oov_token] = num_words + 1
    
    X = tokenizer.texts_to_sequences(days_tweets)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    maxlen = 50
    X_padded = pad_sequences(X, padding='post', maxlen=maxlen)
    
    
    """
    predict emotions for each tweet this day
    """
    y = model.predict(X_padded)
    days_emotion_percent = []           

    threshold = 0.05
    for i in range(0,27):
        emotion_prediction = y[:,i]
        emotion_prediction[emotion_prediction>=threshold]=1
        emotion_prediction[emotion_prediction<threshold]=0
        percent = round(len(emotion_prediction[emotion_prediction==1])/len(emotion_prediction),2)
        days_emotion_percent.append(percent)
    
    average_emotions.append(days_emotion_percent)
    

""" 
graph an emotion
"""

average_emotions_df =pd.DataFrame.from_records(average_emotions)















