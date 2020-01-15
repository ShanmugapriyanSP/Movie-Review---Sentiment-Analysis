from flask import render_template, request, flash, url_for
import re
import nltk
from werkzeug.utils import redirect
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sentiment_prediction import app
import pickle

def predict_review_type(review_test):
    sparser = pickle.load(open('sentiment_prediction/files/sparser.pkl', 'rb'))
    model = pickle.load(open('sentiment_prediction/files/model.pkl', 'rb'))

    review_test = re.sub('[^a-zA-Z]', ' ', review_test)
    review_test = review_test.lower()
    review_test = review_test.split()
    ps = PorterStemmer()
    review_test = [ps.stem(word) for word in review_test if not word in set(stopwords.words('english'))]
    review_test = ' '.join(review_test)
    review_test = [review_test]
    review_test = sparser.transform(review_test).toarray()
    result = model.predict(review_test)[0]
    if result == 0:
        result = 'Review is Negative'
    if result == 1:
        result = 'Review is Somewhat Negative'
    if result == 2:
        result = 'Review is Neutral'
    if result == 3:
        result = 'Review is Somewhat Positive'
    if result == 4:
        result = 'Review is Positive'
    #result = "Positive Review" if model.predict(review_test)[0] == 1 else "Negative Review"
    return result


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', title='ML - Sentiment Analysis')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        test_review = request.form['review']
        if test_review == '':
            result = ''
        else:
            result = predict_review_type(test_review)
    except:
        print('Exception')
        return redirect(url_for('home'))
    return render_template('home.html', result=result, title='Prediction')
