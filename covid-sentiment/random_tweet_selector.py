# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:47:54 2020
@author: Rian van den Ander
"""

import random
from os import listdir
from os.path import isfile, join

path = '../../data/covid19_one_hundred_million_unique_tweets/tweet_ids/'
tweetfiles = [f for f in listdir(path) if isfile(join(path, f))]
tweet_ids = []

for file in tweetfiles:
    C = 1000
    fpath = file
    buffer = []
    
    f = open(path+file, 'r', encoding='utf8')
        
    for line_num, line in enumerate(f):
        n = line_num + 1.0
        r = random.random()
        if n <= C:
            buffer.append(line.strip())
        elif r < C/n:
            loc = random.randint(0, C-1)
            buffer[loc] = line.strip()
            
    for item in buffer:
        tweet_ids.append(item)
        
    
all_tweets_filename = 'all_tweets_2000_pd.csv'
all_tweets_file = open(path+all_tweets_filename, 'w+', encoding='utf8')
all_tweets_file.writelines(["%s\n" % item  for item in tweet_ids])