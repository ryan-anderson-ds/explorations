# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9 2019
@author: rian-van-den-ander
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

release_dates = pd.read_csv('netflix.csv', header=0)
imdb_indices = pd.read_csv('title.basics.tsv.gz', compression='gzip', sep='\t', header=0)
imdb_ratings =  pd.read_csv('title.ratings.tsv', sep='\t', header=0)

indexMerge = pd.merge(release_dates, imdb_indices, how='inner', left_on='Offering', right_on='primaryTitle')
ratingsMerge = pd.merge(indexMerge, imdb_ratings, how='inner', left_on='tconst', right_on='tconst')
ratings = ratingsMerge[ratingsMerge.numVotes > 500] #killing duplicates on netflix or small shows. results in almost same size dataset

#pulling out year
ratings['Year'] = ratings['Date'].map(lambda x: datetime.datetime.strptime(x, '%B %d, %Y').year)


#avg ratings
meanratings = ratings.groupby('Year').mean()
meanratings['Year'] = meanratings.index

sns.set_style("white")

# Draw Stripplot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
b = sns.stripplot(ratings["Year"], ratings["averageRating"], jitter=0.25, color="#c66c6c", size=8, ax=ax, linewidth=.5)
b.set_facecolor('#ffd4d6')
b.tick_params(axis='both', which='major', labelsize=15)
b.tick_params(axis='both', which='minor', labelsize=15)
b.set_xlabel("Year", fontsize=30,color="#e50914")
b.set_ylabel("Rating", fontsize=30,color="#e50914")

x = plt.gca().axes.get_xlim()
             
b.plot([0,1,2,3,4,5,6,7],meanratings['averageRating'],color='#ca9999',linewidth=5)
          
plt.title('Netflix: Quantity vs Quality', fontsize=22)
plt.show()






