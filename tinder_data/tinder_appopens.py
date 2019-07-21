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
usage_data = pd.DataFrame.from_dict(data["Usage"]["app_opens"], orient='index')

""" i found that the daily wasn't that useful. it obviously spiked on weekends
but it was erratic. how about a rolling monthly mean? """
mean = usage_data.rolling(window=30, min_periods = 1).mean()


#plotting result
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 300/my_dpi), dpi=my_dpi)
plt.plot(usage_data.index, mean.iloc[:,0], color='orange', marker='', linewidth=2)
plt.title("Tinder app opens", fontdict={'family': 'serif',
        'weight': 'bold',
        'size': 10,
        })
plt.xlim(0,258)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False) # labels along the bottom edge are off

plt.axes().set_xticks(plt.axes().get_xticks()[::30])
