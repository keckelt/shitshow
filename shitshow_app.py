import requests
from urllib.parse import urlparse

import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

'''
# Warcraft Forum Shitshow Detector 💩

The World of Warcraft is a magical one, full of brave and noble knights. 🧙‍♂️🧝‍♀️

That the players themselves embody these attributes is evident in the official forums, 
where the community gathers to welcome announcements from the development team with joy and gratitude, 
and engages in philosophical discussions ...no, actually it  is a *shitshow* 💩

To protect yourself from these verbal horrors, you can check the atmosphere here first.
'''

def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

@st.cache(persist=True,show_spinner=False)
def get_classifier(id):
  with st.spinner('Loading model for sentinment analysis...'):
    classifier = pipeline(id)
  return classifier

@st.cache(show_spinner=False)
def get_post(post_id):
  return requests.get('https://us.forums.blizzard.com/en/wow/posts/%s.json' % post_id).json()

@st.cache(show_spinner=False)
def showBalloons():
  # display only once, by caching the function
  st.balloons()
  pass

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
  

last_blue_posts = getPosts()[['title', 'url']].head(5)
last_blue_posts.loc[len(last_blue_posts)] = ['Enter an URL 🔗', None]

title = st.radio( 
  "Entern an URL, or pick one of the most recent developer posts:", last_blue_posts)

url = last_blue_posts.loc[last_blue_posts['title'] == title]['url'].iloc[0]
if url is None:
  st.info('The model can only judge English posts.')
  url = st.text_input('URL')
else:
  # https://us.forums.blizzard.com/en/wow/t/<thread_id>.json
  url = 'https://us.forums.blizzard.com/en/wow'+url
  st.markdown('Forum Link: [%s](%s)' % (title, url))


if not uri_validator(url):
  "URL is not valid: " + url
  st.stop()


url = url + '.json'

post_ids = requests.get(url).json()['post_stream']['stream']

'## Analysis'
msg = 'There are %s posts in the selected thread.' % len(post_ids)
if len(post_ids) > 20:
  msg = msg + ' Will load and analyze the first 20.'
  post_ids = post_ids[0:20]

msg
try:
  classifier = get_classifier('sentiment-analysis')

  post_results = []
  progress_wrapper = st.empty()
  post_progress = progress_wrapper.progress(0)

  for i, post_id in enumerate(post_ids):
    # https://us.forums.blizzard.com/en/wow/posts/<post_id>.json
    post = get_post(post_id)
    post_text = post['raw'].replace('\n', ' ')
    # '* ' + post_text
    results = classifier(post_text)
    post_progress.progress((i+1)/len(post_ids))
    if results[0]['score'] > 0.75:
      post_results.append({
        'text': post_text,
        'url': 'https://us.forums.blizzard.com/en/wow/p/%s' % post['id'],
        'sentiment': results[0]['label'],
        'score': results[0]['score']
      })

  post_results = pd.DataFrame(post_results)
  # post_results
  chart = alt.Chart(post_results).mark_text(size=30).encode(
    x=alt.X('rank:O', title=None),
    y=alt.Y('sentiment:N', title=None),
    text='emoji:N',
    tooltip='text'
  ).transform_calculate(
      emoji="{'POSITIVE': '😀', 'NEGATIVE': '💩'}[datum.sentiment]"
  ).transform_window(
    rank='rank()',
    groupby=['sentiment']
  ).properties(
      width=600,
      height=300
  ).configure_view(
    strokeWidth=0.0
  )

  progress_wrapper.empty()
  chart
  showBalloons()
except BaseException as e:
  st.error('Somethiong went wrong, please try a diferent thread')
  st.exception(e)
