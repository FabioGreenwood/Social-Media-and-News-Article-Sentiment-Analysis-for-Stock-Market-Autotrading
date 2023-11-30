import numpy as np
import pandas as pd
import re

# this file is for removing tweets thats are not sueful for our experiment, this can be done by:
#1. removing tweets that dont mention the company in question

raw_tweet_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\data\twitter data\Tweets about the Top Companies from 2015 to 2020\Tweet.csv\Tweet.csv"
new_tweet_file_path = r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\data\twitter data\Tweets about the Top Companies from 2015 to 2020\Tweet.csv\apple.csv"
df = pd.read_csv(raw_tweet_file_path)
df['body'] = df['body'].str.lower()
print("original length : {}".format(len(df)))
max_other_company_mentions_allowed = 3


def process_tweet_1(tweet):
    
    tweet = re.sub(r'\.', "", tweet)
    tweet = re.sub(r',', "", tweet)
    tweet = re.sub(r'\$\d+', 'MONEYAMOUNT', tweet)
    tweet = re.sub(r'\$aapl', 'apple', tweet)
    tweet = re.sub(r'\$\w+', 'OTHERCOMPANY', tweet)
    return tweet

#remove tweets concernign apple
df = df[df['body'].str.contains(r'\$AAPL', case=False)]
df['body'] = df['body'].apply(process_tweet_1)


#remove commas and full sets
df = df[df['body'].str.count(r'\bOTHERCOMPANY\b', flags=re.IGNORECASE) < 2]

# Reset index after filtering
df = df.reset_index(drop=True)
print("new length      : {}".format(len(df)))
df.to_csv(new_tweet_file_path)

print("hello")