import keys
import tweepy
from textblob import TextBlob

API_Key = keys.apikey
API_SecretKey = keys.apiseckey
accessToken = keys.accesstoken
access_SecretToken = keys.accessseckey
auth = tweepy.OAuthHandler(API_Key, API_SecretKey)
auth.set_access_token(accessToken, access_SecretToken)

api = tweepy.API(auth)
public_tweets = api.search('Apple')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    if analysis.sentiment[0] > 0:
        print('Positive')
    elif analysis.sentiment[0] < 0:
        print('Negative')
    else:
        print('Neutral')
