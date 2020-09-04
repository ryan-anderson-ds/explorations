# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:33:40 2020
@author: Rian van den Ander
"""

import pandas as pd
df = pd.read_json ('../../data/covid19_one_hundred_million_unique_tweets/tweet_ids/all_tweets_500_pd.jsonl', encoding='utf8', lines=True)

#take english tweets only
df = df[df['lang']=='en']


#todo
#one list of tweets per day
for index, row in df.iterrows():
    print(row['c1'], row['c2'])