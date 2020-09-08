# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:33:40 2020
@author: Rian van den Ander
"""

import pandas as pd
df = pd.read_json ('../../data/covid19_one_hundred_million_unique_tweets/tweet_ids/all_tweets_2000_pd.jsonl', encoding='utf8', lines=True)

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
Global covid data
"""
covid_df = pd.read_csv ('../../data/covid19_one_hundred_million_unique_tweets/covid-tests-cases-deaths-per-million.csv', encoding='utf8')
covid_df['Date']= covid_df['Date'].astype('datetime64[ns]') 
covid_df2 = covid_df[(covid_df['Date'] >= "2020-01-24") & (covid_df['Date'] <= "2020-08-29")]
covid_df4 = covid_df2.groupby('Date', as_index=False).sum()

last_val = 0
for index, row in covid_df4.iterrows():
    temp = row['Confirmed deaths per million (deaths per million)']
    covid_df4.loc[index, 'Confirmed deaths per million (deaths per million)'] = row['Confirmed deaths per million (deaths per million)'] - last_val
    last_val = temp
covid_df4.loc[0, 'Confirmed deaths per million (deaths per million)'] = 0 #since this value will be inaccurate anyways


"""
US Covid data
"""
covid_df = pd.read_csv ('../../data/covid19_one_hundred_million_unique_tweets/covid-tests-cases-deaths-per-million.csv', encoding='utf8')
covid_df['Date']= covid_df['Date'].astype('datetime64[ns]') 
covid_df2 = covid_df[(covid_df['Date'] >= "2020-01-24") & (covid_df['Date'] <= "2020-08-29")]
covid_df2 = covid_df2[(covid_df2['Entity'] == "United States")]
covid_usa = covid_df2.groupby('Date', as_index=False).sum()

last_val = 0
for index, row in covid_usa.iterrows():
    temp = row['Confirmed deaths per million (deaths per million)']
    covid_usa.loc[index, 'Confirmed deaths per million (deaths per million)'] = row['Confirmed deaths per million (deaths per million)'] - last_val
    last_val = temp
covid_usa.loc[0, 'Confirmed deaths per million (deaths per million)'] = 0 #since this value will be inaccurate anyways


""" 
Prepare emotion values for graph
"""

average_emotions_df = pd.DataFrame.from_records(average_emotions)
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
dates = pd.to_datetime(pd.Series(dates), format='%Y-%m-%d')
seaborn_df = pd.DataFrame(columns=["x","g"])

for x in range(0,len(emotion_names)):

    emotion_name = emotion_names[x]
    
    if(emotion_name == 'disgust'):            
        ysmoothed = gaussian_filter1d(average_emotions_df[x], sigma=30)
    else:            
        ysmoothed = gaussian_filter1d(average_emotions_df[x], sigma=12)
        
    total_y = sum(ysmoothed)
    
    count = 0
    for y in ysmoothed:
        days_percent = y/total_y
        probability = days_percent/0.001
        bucket_assignments = int(round(probability))
        
        if(days_percent > 0.004): #a cutoff point for the emotion
            for assignment in range(0,bucket_assignments):
                seaborn_df = seaborn_df.append({"x":count,"g":emotion_name},ignore_index=True)
        count+=1

"""
Prepare covid values for graph
"""

covid_smoothed = gaussian_filter1d(covid_df4['Confirmed deaths per million (deaths per million)'], sigma=12)
total_y = sum(covid_smoothed)
count = 0
for y in covid_smoothed:
    days_percent = y/total_y
    probability = days_percent/0.0001
    if(days_percent < 0):
        days_percent = 0
    bucket_assignments = int(round(probability))
    for assignment in range(0,bucket_assignments):
        seaborn_df = seaborn_df.append({"x":count,"g":'global covid-19 deaths/population'},ignore_index=True)
    count+=1

covid_usa_smoothed = gaussian_filter1d(covid_usa['Confirmed deaths per million (deaths per million)'], sigma=12)
total_y = sum(covid_usa_smoothed)
count = 0
for y in covid_usa_smoothed:
    days_percent = y/total_y
    probability = days_percent/0.001 #lowering this smooths out the graphs
    if(days_percent < 0):
        days_percent = 0
    bucket_assignments = int(round(probability))
    for assignment in range(0,bucket_assignments):
        seaborn_df = seaborn_df.append({"x":count,"g":'USA covid-19 deaths/population'},ignore_index=True)
    count+=1


"""
Graph
"""

seaborn_df_arranged_in_order = seaborn_df[seaborn_df["g"]=="confusion"].append(
    [seaborn_df[seaborn_df["g"]=="amusement"],
    seaborn_df[seaborn_df["g"]=="approval"],
    seaborn_df[seaborn_df["g"]=="disapproval"],
    seaborn_df[seaborn_df["g"]=="caring"],
    seaborn_df[seaborn_df["g"]=="fear"],
    seaborn_df[seaborn_df["g"]=="gratitude"],
    seaborn_df[seaborn_df["g"]=="anger"],
    seaborn_df[seaborn_df["g"]=="annoyance"],
    seaborn_df[seaborn_df["g"]=="embarrassment"],
    seaborn_df[seaborn_df["g"]=="disappointment"],
    seaborn_df[seaborn_df["g"]=="disgust"],
    seaborn_df[seaborn_df["g"]=="global covid-19 deaths/population"],
    seaborn_df[seaborn_df["g"]=="USA covid-19 deaths/population"],    
    ])
    

"""
amusement
approval
disapproval
confusion
caring 
fear 
gratitude
anger
annoyance
embarassment
disappointment
disgust
global covid-19 death rate
USA covid-19 death rate
"""


import seaborn as sns
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(14, rot=-.25, light=.7)
g = sns.FacetGrid(seaborn_df_arranged_in_order, row="g", hue="g", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, 0, label, fontweight="bold", color=color,
            ha="right", va="baseline", transform=ax.transAxes)

g.map(label, "x")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.50)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)