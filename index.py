from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import malaya
from textblob import TextBlob
import pandas as pd
import tweepy
import re
import os


app = Flask(__name__)
CORS(app)

load_dotenv()
# get/create a .env file
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_SECRET")


@app.route('/')
def test():
    return "test"


@app.route('/scrape')
def scrape():
    # handling OAuth
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    # config
    tweet_mode = 'extended'
    result_type = 'recent'
    count = 100
    query = 'unifi'
    query2 = '@unifi OR @unifihelp'
    geocode = '4.586542,104.076119,450km'
    max_tweets = 10

    # data cleaning

    # unwanted authors
    uw_au = ['unifi', 'unifihelp', 'helpmeunifi']
    cleaner_regex = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"

    try:
        searched_tweets1 = [status._json for status in tweepy.Cursor(
            api.search, q=query, tweet_mode=tweet_mode,
            result_type=result_type, geocode=geocode,
            count=count).items(max_tweets)]

        searched_tweets2 = [status._json for status in tweepy.Cursor(
            api.search, q=query2, tweet_mode=tweet_mode,
            result_type=result_type, count=count).items(max_tweets)]

        searched_tweets = searched_tweets1 + searched_tweets2
        # print(searched_tweets)
        tweet_to_csv = []
        saved_tweet_id = []
        # regex to clear mentions, URL, etc
        for tweet in searched_tweets:

            if tweet['id'] not in saved_tweet_id:

                saved_tweet_id.append(tweet['id'])
                if tweet['user']['name'] not in uw_au:
                    tweet['full_text'] = ' '.join(
                        re.sub(cleaner_regex, " ", tweet['full_text']).split())
                    tweet_to_csv.append(tweet)

        if len(tweet_to_csv) > 0:
            csvFile = "tweet_concat" + ".csv"
            pd.DataFrame(tweet_to_csv).to_csv(csvFile, index=False)
            return jsonify(csvFile + " has been created."), 200
        else:
            return jsonify("Empty tweet."), 200

    except Exception as e:
        return jsonify("Some error"), 400


@app.route('/model', methods=['GET'])
def model_out():
    try:
        df = pd.read_csv('tweet_concat.csv')
        features = ['created_at', 'id', 'user', 'full_text']
        data = df[features]
        data.full_text = data.full_text.astype('str')
        text = data.full_text
        text_list = text.values.tolist()  # model requires list as input
        mn_lang = malaya.language_detection.multinomial()
        lang = mn_lang.predict_batch(text_list)

        # add lang column
        data['lang'] = lang
        data.lang.value_counts()  # ISSUE: model interprets malay as other

        english = data[data['lang'] == 'ENGLISH']
        english_text = english[['full_text']]
        # model requires list as input
        english_text_list = english_text.full_text.values.tolist()

        # other and indonesia mostly consist of malay based on observation
        malay = data[data['lang'] != 'ENGLISH']
        malay_text = malay[['full_text']]
        # model requires list as input
        malay_text_list = malay_text.full_text.values.tolist()

        english_sentiment = []
        for tweet in english_text_list:
            blob = TextBlob(tweet)
            analysis = blob.sentiment
            if analysis[0] >= 0:
                english_sentiment.append('positive')
            elif analysis[0] < 0:
                english_sentiment.append('negative')

        # add sentiment column
        english['sentiment'] = english_sentiment
        english.sentiment.value_counts()

        # sentiment analysis for malay
        malay_sentiment_xgb = malaya.sentiment.xgb()
        malay_sentiment = malay_sentiment_xgb.predict_batch(
            malay_text_list)  # get_proba=True

        # add sentiment column
        malay['sentiment'] = malay_sentiment
        malay.sentiment.value_counts()

        # lda = malaya.topic_model.lda(
        #     english_text_list, 10, stemming=False, vectorizer='skip-gram',
        #     ngram=(1, 4), skip=3)

        # lda2vec = malaya.topic_model.lda(
        #     malay_text_list, 10, stemming=False, vectorizer='skip-gram',
        #     ngram=(1, 4), skip=3)

        output = english.merge(malay, how='outer')
        return output.to_json(orient="records"), "Updated Successfully!"
    except FileNotFoundError:
        return jsonify('File not found'), 400


if __name__ == "__main__":
    app.run(debug=True)
