# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:22:59 2020

@author: acros
"""

import tweepy
import time
import pandas as pd
from datetime import datetime
import os
import json # The API returns JSON formatted text

consumer_key = '###################'
consumer_secret = '###################'
access_token = '###################'
access_token_secret = '###################'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)
NUM_TWEETS_TO_CAPTURE = 1000000
#OUTPUT_FILE = "streamed_tweets.txt"
#%%
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        #self.file = open(OUTPUT_FILE, "w")
        self.tweets = []
    def on_status(self, status):
        if self.num_tweets == NUM_TWEETS_TO_CAPTURE:
            return(False)
            
        if self.num_tweets % 100 == 0:
            print("Captured {} so far...".format(self.num_tweets))
        if hasattr(status, "retweeted_status"):  # Check if Retweet
            try:
                #print(status.retweeted_status.extended_tweet["full_text"])
                text = status.retweeted_status.extended_tweet['full_text']
            except AttributeError:
                #print(status.retweeted_status.text)
                text = status.retweeted_status.text
        else:
            try:
                #print(status.extended_tweet["full_text"])
                text = status.extended_tweet['full_text']
            except AttributeError:
                #print(status.text)
                text = status.text
                
        username = status.user.screen_name
        acctdesc = status.user.description
        location = status.user.location
        following = status.user.friends_count
        followers = status.user.followers_count
        totaltweets = status.user.statuses_count
        usercreatedts = status.user.created_at
        tweetcreatedts = status.created_at
        retweetcount = status.retweet_count
        hashtags = status.entities['hashtags']
        
        self.tweets.append([username, acctdesc, location, following, followers, totaltweets,
                         usercreatedts, tweetcreatedts, retweetcount, text, hashtags])
        self.num_tweets += 1
        
    #self.file.close()
        
#%%

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
words_to_track = ['police', 'police brutality', 'blm', 'blacklivesmatter',
                  'protests', 'antifa', 'liberals', 'whitelivesmatter',
                  'bluelivesmatter', 'racist', 'george floyd']
myStream.filter(track=words_to_track)

# Obtain timestamp in a readable format
cols = ['username', 'acctdesc', 'location', 'following',
        'followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts',
        'retweetcount', 'text', 'hashtags']
db_tweets = pd.DataFrame(myStreamListener.tweets, columns=cols)
to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')# Define working path and filename
#path = os.getcwd()
path = r"C:\\Users\\acros\\Documents\\data\\protest_tweets\\"
filename = path + to_csv_timestamp + '_georgefloydprotests.csv'# Store dataframe in csv with creation date timestamp
db_tweets.to_csv(filename, index = False)
