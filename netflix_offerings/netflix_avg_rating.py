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

#total offerings
totalofferings = pd.DataFrame(ratings.groupby('Year').size())

graphdata = pd.merge(meanratings, totalofferings, how='inner', left_index=True, right_index=True)


graphdata = graphdata[['averageRating',0]]
graphdata.rename(columns={0:'numOfferings'}, 
                 inplace=True)
graphdata = graphdata.reset_index()

#notes:
#a few duplicates so not 100% accurate
#release date of things, so if they kept running it may raise or lower the average

colors = [plt.cm.rainbow(i/float(len(graphdata)-1)) for i in range(len(graphdata))]
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
plt.scatter(graphdata.Year, graphdata.averageRating, s=graphdata.numOfferings*50, c=colors) #graphdata.numOfferings 5-150

for i, txt in enumerate(graphdata['numOfferings']):
    ax.annotate(txt, (graphdata['Year'][i]-50, graphdata['averageRating'][i]-50))

plt.title('Netflix: Quantity vs Quality', fontsize=22)
plt.show()





OldRange = (308 - 1)  
NewRange = (150 - 10)  

OldValue = 50
NewValue = (((OldValue - 1) * NewRange) / OldRange) + 10


