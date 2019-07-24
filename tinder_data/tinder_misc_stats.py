# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:40:21 2019

@author: User
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:07:18 2019
@author: rian-van-den-ander
"""

import json

""" Reading the json as a dict. The tinder data is not valid json
 this creates a dictionary of size 7, one with tinder campaigns, one 
 with messages, one with swipes, etc """
with open('tinder_data.json', encoding="utf8") as json_data:
    data = json.load(json_data)
   
conversation_count = 0


for x in data["Messages"]:
 
    if len(x["messages"]) >= 5:
        conversation_count += 1

match_count = 0

for x in data["Usage"]["matches"].values():
    match_count += x

right_swipe_count = 0

for x in data["Usage"]["swipes_likes"].values():
    right_swipe_count += x

left_swipe_count = 0

for x in data["Usage"]["swipes_passes"].values():
    left_swipe_count += x

#quick copypasta to read the amount of times my number was sent
file = open('tinder_data.json', 'r', encoding="utf8").read()
number_count = file.count("4293")
