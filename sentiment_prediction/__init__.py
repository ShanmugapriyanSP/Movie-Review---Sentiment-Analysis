from flask import Flask


app = Flask(__name__)

from sentiment_prediction import routes

