#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:07:18 2019
@author: rian-van-den-ander
"""

import pandas as pd
import json
import os
import matplotlib.pyplot as plt

""" Reading the json as a dict. The tinder data is not valid json
 this creates a dictionary of size 7, one with tinder campaigns, one 
 with messages, one with swipes, etc """
with open('tinder_data.json', encoding="utf8") as json_data:
    data = json.load(json_data)
    
# just interested in the 'app_opens' usage data
right_swipes = pd.DataFrame.from_dict(data["Usage"]["swipes_likes"], orient='index')
left_swipes = pd.DataFrame.from_dict(data["Usage"]["swipes_passes"], orient='index')
matches = pd.DataFrame.from_dict(data["Usage"]["matches"], orient='index')


"""I only wanted days where i wasn't just picking from top of the stack, where i know people who like you go
so i took out days with less than 10 left swipes - NB these days account for jumps up and down in data"""

swipy_day_right_swipes = []
swipy_day_left_swipes = []
swipy_day_matches = []

x=0

for index, row in left_swipes.iterrows():
    if(row[0] >= 10):
        swipy_day_right_swipes.append(right_swipes.iloc[x])
        swipy_day_left_swipes.append(left_swipes.iloc[x])
        swipy_day_matches.append(matches.iloc[x])
    x+= 1

right_swipes = pd.DataFrame(swipy_day_right_swipes)
left_swipes = pd.DataFrame(swipy_day_left_swipes)
matches = pd.DataFrame(swipy_day_matches)

# again, i've smoothed this out a bit with a rolling window.
swipes = right_swipes / 50
swipes = swipes.rolling(window=10, min_periods = 1).mean()

# also smoothing out matches
matches_total = matches / swipes
matches_total = matches_total.rolling(window=10, min_periods = 1).mean()

#plotting result
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 300/my_dpi), dpi=my_dpi)
plt.tight_layout()
plt.plot(matches_total.index[5:], matches_total.iloc[5:,0], marker='', color='orange',linewidth=2, label = 'matches per 50 right swipes')
plt.title("Tinder matches per day controlled for swipes",fontdict={'family': 'serif',
        'weight': 'bold',
        'size': 10,
        })
plt.legend(loc='upper left')
plt.xlim(0,155)
plt.style.use('seaborn-paper')
plt.tick_params(
    axis='x',
    which='both',
    bottom=False, 
    top=False) 
plt.axes().set_xticks(plt.axes().get_xticks()[::30])