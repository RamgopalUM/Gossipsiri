import tweepy
import csv
import pandas as pd
####input your credentials here
consumer_key = 'Gq1ZPTPYEeuu2QJu51asnNC5O'
consumer_secret = 'LU9rJ70KDKEVyc2wP40nOTYs52TbMZhuRgxb1jhzS0TuV8Fz9j'
access_token = '1322751044175253504-fy5RAfAMjjCDES28RZUe4jB7zxORft'
access_token_secret = '8F5KmGZHnuqFxrrCYbACtEXXOqwIjnbLhRFSifW0DWjDH'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Open/Create a file to append data


csvFile = open('Tesla.csv', 'a')
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search, q="$TSLA", count=200,
                           lang="en").items():
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
