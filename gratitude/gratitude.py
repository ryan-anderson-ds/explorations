# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:14:00 2019

@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data.csv', header=0)

"""
Data cleansing
--------------
"""
dataset = dataset.fillna(0)


"""
Most common themes?
----------------------
"""
themes = dataset.columns[2:]
counts = dataset.sum(axis = 0, skipna = True)[2:]
df = pd.DataFrame({'themes': themes, 'counts': list(counts)}, columns=['themes', 'counts'])
df = df.sort_values(by='counts', ascending=False, na_position='first')

sns.set(style="dark")
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="counts", y="themes", data=df,
            label="Total", color="b")
ax.xaxis.tick_top()
sns.despine(left=True, bottom=True)
ax.set_ylabel('')    
ax.set_xlabel('')








