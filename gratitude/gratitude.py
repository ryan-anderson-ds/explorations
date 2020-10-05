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
f, ax = plt.subplots(figsize=(6, 15))

sns.set_color_codes("pastel")
sns.barplot(x="counts", y="themes", data=df,
            label="Total", color="b")
ax.xaxis.tick_top()
sns.despine(left=True, bottom=True)
ax.set_ylabel('')    
ax.set_xlabel('')

"""
Generalised gratefulness
----------------------
"""
themes_generalised = ['Surroundings and city','People and romance','Peace','Feelings','Basic needs or normality','Accomplishment'
                      ,'Possessions and financial security','Experiences']
counts_generalised = []

counts_generalised.append(
        int(df.loc[df['themes']=='Living in Amsterdam / Netherlands']['counts'])
        +int(df.loc[df['themes']=='Belonging where I live']['counts'])
        +int(df.loc[df['themes']=='Beauty or tranquility of surroundings']['counts'])
        +int(df.loc[df['themes']=='Good weather']['counts'])
        )

#I joined romance in here because it's not fair to put gratefulness for that in its own group in a year I spent mostly without partners
counts_generalised.append(
        int(df.loc[df['themes']=='A good, or improved, social life']['counts'])
        +int(df.loc[df['themes']=='A specific friend or family member']['counts'])
        +int(df.loc[df['themes']=='General positivity in the world']['counts'])
        +int(df.loc[df['themes']=='Colleagues']['counts'])
        +int(df.loc[df['themes']=='Feeling romantically wanted']['counts'])
        +int(df.loc[df['themes']=='Physical intimacy']['counts'])
        )

#this one is tricky to place. It could be an experience (but if anything, its lack of experience). It could be a feeling, which is the closest match
counts_generalised.append(
        int(df.loc[df['themes']=='Free time, relaxation or lack of responsibility']['counts'])
        )

counts_generalised.append(
        int(df.loc[df['themes']=='Hope for the future']['counts'])
        +int(df.loc[df['themes']=='Flow']['counts'])
        +int(df.loc[df['themes']=='Noticing strong positive emotions in myself']['counts'])
        )

counts_generalised.append(
        int(df.loc[df['themes']=='Good health']['counts'])
        +int(df.loc[df['themes']=='Physical safety']['counts'])
        +int(df.loc[df['themes']=='Post-lockdown normality']['counts'])
        +int(df.loc[df['themes']=='The worst is over / it could be worse']['counts'])
        )


counts_generalised.append(
        int(df.loc[df['themes']=='Physical ability or accomplishment']['counts'])
        +int(df.loc[df['themes']=='Being skilled at things']['counts'])
        +int(df.loc[df['themes']=='Work accomplishment']['counts'])
        )

counts_generalised.append(
        int(df.loc[df['themes']=='Apartment']['counts'])
        +int(df.loc[df['themes']=='Financial and employment situation']['counts'])
        +int(df.loc[df['themes']=='Possessions, technology, or advanced society']['counts'])
        )

counts_generalised.append(
        int(df.loc[df['themes']=='Food']['counts'])
        +int(df.loc[df['themes']=='A unique experience']['counts'])
        +int(df.loc[df['themes']=='Music']['counts'])
        +int(df.loc[df['themes']=='The little things (e.g. fresh sheets)']['counts'])
        +int(df.loc[df['themes']=='Unexpected good luck']['counts'])
        +int(df.loc[df['themes']=='Running']['counts'])
        +int(df.loc[df['themes']=='Computer gaming']['counts'])
        +int(df.loc[df['themes']=='Release (e.g. cool swim after heat)']['counts'])
        )

df_generalised = pd.DataFrame({'themes': themes_generalised, 'counts': list(counts_generalised)}, columns=['themes', 'counts'])
df_generalised = df_generalised.sort_values(by='counts', ascending=False, na_position='first')

sns.set(style="dark")
f, ax = plt.subplots(figsize=(6, 5))
sns.set_color_codes("pastel")
sns.barplot(x="counts", y="themes", data=df_generalised,
            label="Total", color="b")
ax.xaxis.tick_top()
sns.despine(left=True, bottom=True)
ax.set_ylabel('')    
ax.set_xlabel('')




"""
Area chart
----------------------
"""

# Data
df = dataset
df['DATE'] =pd.to_datetime(df['DATE'])
df = df.sort_values(by='DATE')
x=df['DATE'].drop_duplicates()

x=x[17:]

df_grouped = df.groupby(['DATE']).sum()
df = df_grouped
df = df.iloc[17:,:]


df['Surroundings and city'] = (df['Living in Amsterdam / Netherlands'] + df['Belonging where I live']
    + df['Beauty or tranquility of surroundings'] + df['Good weather'])

df['People and romance'] = (df['A good, or improved, social life'] + df['A specific friend or family member']
    + df['General positivity in the world'] + df['Colleagues'] + df['Feeling romantically wanted'] + df['Physical intimacy'])

df['Peace'] = (df['Free time, relaxation or lack of responsibility'])

df['Feelings'] = (df['Hope for the future'] + df['Flow']
    + df['Noticing strong positive emotions in myself'])

df['Basic needs or normality'] = (df['Good health'] + df['Physical safety']
    + df['Post-lockdown normality'] + df['The worst is over / it could be worse'])

df['Accomplishment'] = (df['Physical ability or accomplishment'] + df['Being skilled at things']
    + df['Work accomplishment'])

df['Possessions and financial security'] = (df['Apartment'] + df['Financial and employment situation']
    + df['Possessions, technology, or advanced society'])

df['Experiences'] = (df['Food'] + df['A unique experience']
    + df['Music'] + df['The little things (e.g. fresh sheets)'] + df['Unexpected good luck'] + df['Running']
    + df['Computer gaming'] + df['Release (e.g. cool swim after heat)'])

#TODO: now for each date, update totals. I guess a loop to build this up

df['Surroundings and city_sum'] = df['Surroundings and city'].rolling(min_periods=1, window=398).sum()
df['People and romance_sum'] = df['People and romance'].rolling(min_periods=1, window=398).sum()
df['Peace_sum'] = df['Peace'].rolling(min_periods=1, window=398).sum()
df['Feelings_sum'] = df['Feelings'].rolling(min_periods=1, window=398).sum()
df['Basic needs or normality_sum'] = df['Basic needs or normality'].rolling(min_periods=1, window=398).sum()
df['Accomplishment_sum'] = df['Accomplishment'].rolling(min_periods=1, window=398).sum()
df['Possessions and financial security_sum'] = df['Possessions and financial security'].rolling(min_periods=1, window=398).sum()
df['Experiences_sum'] = df['Experiences'].rolling(min_periods=1, window=398).sum()

y = [df['Surroundings and city_sum'],df['People and romance_sum'],df['Peace_sum'],df['Feelings_sum'],
     df['Basic needs or normality_sum'],df['Accomplishment_sum'],df['Possessions and financial security_sum'],
     df['Experiences_sum']]

sns.set(style="dark")
f, ax = plt.subplots(figsize=(10, 6))
sns.set_color_codes("dark")

plt.stackplot(x,y, labels=['Surroundings and city','People and romance','Peace','Feelings','Basic needs or normality',
                           'Accomplishment','Possessions and financial security','Experiences'])
plt.legend(loc='upper left')
plt.show()




