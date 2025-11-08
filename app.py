

from flask import Flask, render_template, request
import pandas as pd
from flask import Flask, render_template, request
import pandas as pd
import sklearn
import numpy as np
import seaborn as sb
import re
import os
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure stopwords exist
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__,template_folder='./templates',static_folder='./static')

loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vectorizer.pkl", 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
corpus = []

def fake_news_det(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
     
    return prediction

        

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        print(pred)
        def predi(pred):
            if pred[0] == 1:
              res="Prediction of the News :  Looking Fake News📰"
            else:
              res="Prediction of the News : Looking Real News📰 "
            return res
        result=predi(pred)
        return render_template("prediction.html",  prediction_text="{}".format(result))
    else:
        return render_template('prediction.html', prediction="Something went wrong")



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides a dynamic port
    app.run(host='0.0.0.0', port=port)

