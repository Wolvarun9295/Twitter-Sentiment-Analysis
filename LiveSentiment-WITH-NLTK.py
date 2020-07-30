import keys
import tweepy
import re
from tweepy import OAuthHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

API_Key = keys.apikey
API_SecretKey = keys.apiseckey
Access_Token = keys.accesstoken
Access_SecretToken = keys.accessseckey

authorizer = OAuthHandler(API_Key, API_SecretKey)
authorizer.set_access_token(Access_Token, Access_SecretToken)

API = tweepy.API(authorizer, timeout=15)
allTweets = []
search_query = 'AMD'

for tweetObject in tweepy.Cursor(API.search, q=search_query + " -filter:retweets", lang='en',
                                 result_type='recent').items(200):
    allTweets.append(tweetObject.text)

# Uncomment the following to download the stopwords if not present
# nltk.download('stopwords')
from nltk.corpus import stopwords

tweets = pd.read_csv(
    "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")

x = tweets.iloc[:, 10].values
y = tweets.iloc[:, 1].values

processedTweets = []

for tweet in range(0, len(x)):
    # Remove all the special characters
    processedTweet = re.sub(r'\W', ' ', str(x[tweet]))
    # remove all single characters
    processedTweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processedTweet)
    # Remove single characters from the start
    processedTweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processedTweet)
    # Substituting multiple spaces with single space
    processedTweet = re.sub(r'\s+', ' ', processedTweet, flags=re.I)
    # Removing prefixed 'b'
    processedTweet = re.sub(r'^b\s+', '', processedTweet)
    # Converting to Lowercase
    processedTweet = processedTweet.lower()
    processedTweets.append(processedTweet)

tfidfConverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
x = tfidfConverter.fit_transform(processedTweets).toarray()

textClassifier = RandomForestClassifier(n_estimators=100, random_state=0)
textClassifier.fit(x, y)

for tweet in allTweets:
    # Remove all the special characters
    processedTweet = re.sub(r'\W', ' ', tweet)
    # remove all single characters
    processedTweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processedTweet)
    # Remove single characters from the start
    processedTweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processedTweet)
    # Substituting multiple spaces with single space
    processedTweet = re.sub(r'\s+', ' ', processedTweet, flags=re.I)
    # Removing prefixed 'b'
    processedTweet = re.sub(r'^b\s+', '', processedTweet)
    # Converting to Lowercase
    processedTweet = processedTweet.lower()
    sentiment = textClassifier.predict(tfidfConverter.transform([processedTweet]).toarray())
    print(processedTweet, ":", sentiment)
