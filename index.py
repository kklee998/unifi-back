from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from dotenv import load_dotenv
import malaya
import pyLDAvis
from textblob import TextBlob
import pandas as pd
import tweepy
import re
import os
import csv


app = Flask(__name__)
CORS(app)
stopwords_eng = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

load_dotenv()
# get/create a .env file
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_SECRET")


@app.route('/')
def test():
    return "/scrape \n /model \n /chart \n /malay_sentiment \n /eng_sentiment \n /english \n /malay \n /dismiss \n /done", 200


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
    max_tweets = 20

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
            return '', '204 EMPTY TWEET'

    except Exception as e:
        return jsonify(str(e)), 400


@app.route('/model', methods=['GET'])
def model_out():
    try:
        df = pd.read_csv('tweet_concat.csv')
        features = ['created_at', 'id_str', 'user', 'full_text']
        data = df[features]
        data.full_text = data.full_text.astype('str')
        data.id_str = data.id_str.astype('str')
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

        # sentiment analysis and subjectivity analysis for english
        english_sentiment = []
        english_subjectivity = []

        for tweet in english_text_list:
            blob = TextBlob(tweet)
            analysis = blob.sentiment
            sentiment = analysis[0]
            subject = analysis[1]
            # scale of -1 to 1
            if sentiment > 0:
                english_sentiment.append('positive')
            elif sentiment < 0:
                english_sentiment.append('negative')
            else:
                english_sentiment.append('neutral')
            # scale of 0 to 1
            if subject > 0.6:
                english_subjectivity.append('subjective')
            elif subject < 0.4:
                english_subjectivity.append('objective')
            else:
                english_subjectivity.append('mixed')

        # add sentiment and subjectivity column
        english['sentiment'] = english_sentiment
        english['subjectivity'] = english_subjectivity
        english.sentiment.value_counts()

        english.subjectivity.value_counts()

        my_sentiment = []
        my_subjective = []
        # sentiment analysis for malay
        malay_sentiment_xgb = malaya.sentiment.xgb()
        malay_sentiment = malay_sentiment_xgb.predict_batch(
            malay_text_list, get_proba=True)  # get_proba=True

        # create list of sentiments(positive/negative/neutral)
        for item in malay_sentiment:
            if item['negative'] > 0.45 and item['negative'] < 0.55:
                my_sentiment.append("Neutral")
            elif item['negative'] > item['positive']:
                my_sentiment.append("Negative")
            else:
                my_sentiment.append("Positive")

        # subjectivity analysis for malay
        malay_subjective_xgb = malaya.subjective.xgb()
        malay_subjective = malay_subjective_xgb.predict_batch(
            malay_text_list, get_proba=True)
        # create list of subjectivity(positive/negative/neutral)
        for item in malay_subjective:
            if item['negative'] > 0.45 and item['negative'] < 0.55:
                my_subjective.append("Mixed")
            elif item['negative'] > item['positive']:
                my_subjective.append("Objective")
            else:
                my_subjective.append("Subjective")

        # add sentiment & subjective column
        malay['sentiment'] = my_sentiment
        malay['subjectivity'] = my_subjective
        malay.subjectivity.value_counts()

        output = english.merge(malay, how='outer')
        return output.to_json(orient="records"), 200
    except FileNotFoundError:
        return jsonify('File not found'), 404

    except Exception as e:
        return jsonify(str(e)), 400


@app.route('/chart')
def make_chart():
    try:
        # import dataset
        df = pd.read_csv('tweet_concat.csv')
        features = ['created_at', 'id_str', 'user', 'full_text']
        data = df[features]
        data.full_text = data.full_text.astype('str')
        data.id_str = data.id_str.astype('str')
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
        malay_sentiment_xgb = malaya.sentiment.xgb()
        malay_sentiment = malay_sentiment_xgb.predict_batch(
            malay_text_list)  # get_proba=True

        # add sentiment column
        malay['sentiment'] = malay_sentiment
        malay.sentiment.value_counts()

        # topic modeling for english
        lda = malaya.topic_model.lda(english_text_list, 10,
                                     stemming=False,
                                     vectorizer='skip-gram',
                                     ngram=(1, 4),
                                     skip=3,
                                     stop_words=stopwords_eng)
        prepared_data = lda.visualize_topics(notebook_mode=False)
        pyLDAvis.save_html(prepared_data, 'pylda.html')
        return send_file('pylda.html', attachment_filename='pylda.html'), 201

    except FileNotFoundError:
        return jsonify('File not found'), 404

    except Exception as e:
        print(str(e))
        return jsonify("CHART IS CURRENTLY UNAVAILABLE"), 400


@app.route('/malay_sentiment')
def malay_sentiment():
    try:

        # import dataset
        df = pd.read_csv('tweet_concat.csv')
        features = ['created_at', 'id_str', 'user', 'full_text']
        data = df[features]
        data.full_text = data.full_text.astype('str')
        data.id_str = data.id_str.astype('str')
        text = data.full_text
        text_list = text.values.tolist()  # model requires list as input

        # language detection
        mn_lang = malaya.language_detection.multinomial()
        lang = mn_lang.predict_batch(text_list)

        # add lang column
        data['lang'] = lang
        data.lang.value_counts()  # ISSUE: model interprets malay as other

        # other and indonesia mostly consist of malay based on observation
        malay = data[data['lang'] != 'ENGLISH']
        malay_text = malay[['full_text']]
        # model requires list as input
        malay_text_list = malay_text.full_text.values.tolist()

        malay_lda = malaya.topic_model.lda(
            malay_text_list, 10, stemming=False,
            vectorizer='skip-gram',
            ngram=(1, 4),
            skip=3)
        malay_topics = malay_lda.top_topics(5, top_n=10, return_df=True)

        # export dataframe for mobile deployment
        malay_topics

        return malay_topics.to_json(orient="records"), 200

    except Exception as e:
        return jsonify("Error has occured : " + str(e)), 400


@app.route('/eng_sentiment')
def eng_sentiment():
    try:

        # import dataset
        df = pd.read_csv('tweet_concat.csv')
        features = ['created_at', 'id_str', 'user', 'full_text']
        data = df[features]
        data.full_text = data.full_text.astype('str')
        data.id_str = data.id_str.astype('str')
        text = data.full_text
        text_list = text.values.tolist()  # model requires list as input

        # language detection
        mn_lang = malaya.language_detection.multinomial()
        lang = mn_lang.predict_batch(text_list)

        # add lang column
        data['lang'] = lang
        data.lang.value_counts()  # ISSUE: model interprets malay as other

        # separate into english and malay
        english = data[data['lang'] == 'ENGLISH']
        english_text = english[['full_text']]
        # model requires list as input
        english_text_list = english_text.full_text.values.tolist()

        # topic modeling for english

        english_lda = malaya.topic_model.lda(
            english_text_list, 10, stemming=False,
            vectorizer='skip-gram',
            ngram=(1, 4),
            skip=3,
            stop_words=stopwords_eng)

        english_topics = english_lda.top_topics(5, top_n=10, return_df=True)
        return english_topics.to_json(orient="records"), 200

    except Exception as e:
        return jsonify("Error has occured : " + str(e)), 400


@app.route('/malay')
def malay():
    try:

        # import dataset
        df = pd.read_csv('tweet_concat.csv')
        features = ['created_at', 'id_str', 'user', 'full_text']
        data = df[features]
        data.full_text = data.full_text.astype('str')
        data.id_str = data.id_str.astype('str')
        text = data.full_text
        text_list = text.values.tolist()  # model requires list as input

        # language detection
        mn_lang = malaya.language_detection.multinomial()
        lang = mn_lang.predict_batch(text_list)

        # add lang column
        data['lang'] = lang
        data.lang.value_counts()  # ISSUE: model interprets malay as other

        # other and indonesia mostly consist of malay based on observation
        malay = data[data['lang'] != 'ENGLISH']
        malay_text = malay[['full_text']]
        # model requires list as input
        malay_text_list = malay_text.full_text.values.tolist()
        # topic modeling for malay

        malay_lda = malaya.topic_model.lda(
            malay_text_list, 10, stemming=False,
            vectorizer='skip-gram',
            ngram=(1, 4),
            skip=3)
        malay_vis = malay_lda.visualize_topics(notebook_mode=False)

        # save to html file for deployment
        pyLDAvis.save_html(malay_vis, 'malay_vis.html')
        return send_file('malay_vis.html',
                         attachment_filename='malay_vis.html'), 201

    except Exception as e:
        return jsonify("Error has occured : " + str(e)), 400


@app.route('/english')
def english():
    try:

        # import dataset
        df = pd.read_csv('tweet_concat.csv')
        features = ['created_at', 'id_str', 'user', 'full_text']
        data = df[features]
        data.full_text = data.full_text.astype('str')
        data.id_str = data.id_str.astype('str')
        text = data.full_text
        text_list = text.values.tolist()  # model requires list as input

        # language detection
        mn_lang = malaya.language_detection.multinomial()
        lang = mn_lang.predict_batch(text_list)

        # add lang column
        data['lang'] = lang
        data.lang.value_counts()  # ISSUE: model interprets malay as other

        # separate into english and malay
        english = data[data['lang'] == 'ENGLISH']
        english_text = english[['full_text']]
        # model requires list as input
        english_text_list = english_text.full_text.values.tolist()

        # topic modeling for english

        english_lda = malaya.topic_model.lda(
            english_text_list, 10, stemming=False,
            vectorizer='skip-gram',
            ngram=(1, 4),
            skip=3,
            stop_words=stopwords_eng)

        english_vis = english_lda.visualize_topics(notebook_mode=False)

        # save to html file for deployment
        pyLDAvis.save_html(english_vis, 'english_vis.html')

        return send_file('english_vis.html',
                         attachment_filename='english_vis.html'), 201

    except Exception as e:
        return jsonify("Error has occured : " + str(e)), 400


@app.route('/done', methods=['GET', 'POST'])
def is_done():
    if request.method == 'POST':

        try:

            tweetid = request.json['id_str']
            with open('isdone.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([tweetid])

            return jsonify(request.json['id_str'] + " successfully added"), 200
        except Exception as e:
            return jsonify("Error has occured : " + str(e)), 400

    else:
        try:
            data = []
            with open('isdone.csv') as f:
                reader = csv.reader(f, delimiter='\n')
                for row in reader:
                    data.append(row[0])

            return jsonify(data), 200

        except OSError as e:
            return jsonify("file does not exist"), 404

        except Exception as e:
            return jsonify("Error has occured : " + str(e)), 400


@app.route('/dismiss', methods=['GET', 'POST'])
def is_dismiss():
    if request.method == 'POST':

        try:

            tweetid = request.json['id_str']
            with open('isdismiss.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([tweetid])

            return jsonify(request.json['id_str'] + " successfully added"), 200
        except Exception as e:
            return jsonify("Error has occured : " + str(e)), 400

    else:
        try:
            data = []
            with open('isdismiss.csv') as f:
                reader = csv.reader(f, delimiter='\n')
                for row in reader:
                    data.append(row[0])

            return jsonify(data), 200

        except OSError as e:
            return jsonify("file does not exist"), 404

        except Exception as e:
            return jsonify("Error has occured : " + str(e)), 400


@app.route('/reply', methods=['POST'])
def reply():
    form_data = request.json
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)

        api.update_status(
            status=form_data['text'], in_reply_to_status_id=form_data['id'])

        return jsonify("Status updated successfully"), 201

    except Exception as e:
        return jsonify("Error has occured : " + str(e)), 400


@app.route('/dm', methods=['POST'])
def dm():
    form_data = request.json
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)

        api.send_direct_message(
            user_id=form_data['id'], text=form_data['text'])

        return jsonify("Message sent successfully"), 201

    except Exception as e:
        return jsonify("Error has occured : " + str(e)), 400


if __name__ == "__main__":
    app.run(debug=True)
