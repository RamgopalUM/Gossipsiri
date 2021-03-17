
"""
Input: Keyword
Output: return 10% live tweets given that keyword

"""

import json
import tweepy
from tweepy.auth import OAuthHandler


class MyStreamListener(tweepy.StreamListener):
    # initialise variable path to store the output tweet
    def __init__(self, path=None):
        self.path = path

    def on_data(self, tweets):
        # take in all the data (listening all tweets) and we can have it ~store it in JSON type of file
        tweet_data = json.loads(tweets)
        # take only the text excluding RT
        text = tweet_data['text']
        if text[0:2] != 'RT':
            print(tweet_data['text'])
            if self.path is not None:
                savefile = open(self.path, 'a', encoding="utf-8")
                savefile.write(tweet_data['text'])
                savefile.close()

        return True

    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        if status_code == 420:
            print(status_code)
            # returning False in on_error disconnects the stream
            return False


def main():
    consumer_key = 'Gq1ZPTPYEeuu2QJu51asnNC5O'
    consumer_secret = 'LU9rJ70KDKEVyc2wP40nOTYs52TbMZhuRgxb1jhzS0TuV8Fz9j'
    access_token = '1322751044175253504-fy5RAfAMjjCDES28RZUe4jB7zxORft'
    access_token_secret = '8F5KmGZHnuqFxrrCYbACtEXXOqwIjnbLhRFSifW0DWjDH'
    stock_list = ['$AAPL', '$TSLA', '$AMZN', '$INTC', '$MSFT', '$NVDA', '$QCOM', '$TWTR', '$LMT', '$GOOG']

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    myStreamListener = MyStreamListener(path="raw_tweets.txt")
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
    myStream.filter(track=stock_list, languages=['en'])


if __name__ == "__main__":
    main()
