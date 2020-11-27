import requests
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
from detoxify import Detoxify

'''
# Warcraft Forum Sentiment Analysis

The World of Warcraft is a magical one, full of brave soliders and noble knights. ⚔
These heroes fight evil and go for their weapons no matter how powerful the enemy is.

That these are not only attributes of the player characters, but the players themselves embody them can be seen in the official forums, 
where the noble-minded community gathers to welcome announcements from the development team with joy and gratitude, provides constructive feedback, and
player engage in philosophical discussions about the ...

no, actually it  is a shitshow 💩


To protect yourself from these verbal horrors, you can check the atmosphere in the topics here.

'''

def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


# Load blue posts: 
#     US = https://us.forums.blizzard.com/en/wow/groups/blizzard-tracker/posts.json
#     EU = https://eu.forums.blizzard.com/en/wow/groups/blizzard-tracker/posts.json
# Contains:
#   excerpt ... truncated post text
#   created ... date of posting
#   url     ... relative url
#   title   ... title of the post/topic
#   user.username ... blue poster 

def getPosts():
  blue_posts = pd.read_json('https://us.forums.blizzard.com/en/wow/groups/blizzard-tracker/posts.json')
  blue_posts.drop_duplicates(subset=['topic_id'], inplace=True)
  blue_posts.reset_index(drop=True, inplace=True)
  return blue_posts
  

last_blue_posts = getPosts()[['title', 'url']].head(9)
last_blue_posts.loc[len(last_blue_posts)] = ['Enter an URL', None]

title = st.radio( 
  "Entern an URL, or pick one of the most recent developer posts.", last_blue_posts)

url = last_blue_posts.loc[last_blue_posts['title'] == title]['url'].iloc[0]
if url is None:
  url = st.text_input('URL')
else:
  # https://us.forums.blizzard.com/en/wow/t/<thread_id>.json
  url = 'https://us.forums.blizzard.com/en/wow'+url


if not uri_validator(url):
  "URL is not valid"
  st.stop()

'Forum Link: ' + url

url = url + '.json'

post_ids = requests.get(url).json()['post_stream']['stream']

'## Posts'
with st.spinner('Loading model for sentinment analysis...'):
  tox = Detoxify('original')

for post_id in post_ids:
  # https://us.forums.blizzard.com/en/wow/posts/<post_id>.json
  post = requests.get('https://us.forums.blizzard.com/en/wow/posts/%s.json' % post_id).json()
  post_text = post['raw'].replace('\n', ' ')
  '* ' + post_text
  results = tox.predict(post_text)
  values = np.fromiter(results.values(), dtype=np.float32)
  values
