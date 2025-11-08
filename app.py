# ===============================================
# Fake News Detection Web App
# Developed by: Pratima Sahu
# College: Madhav Institute of Technology and Science (EEIoT)
# ===============================================

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

# Ensure required NLTK data
# Ensure required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize Flask app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load model and vectorizer
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

loaded_model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ===============================
# Preprocessing Function
# ===============================
def preprocess_text(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)  # Remove special chars
    review = review.lower()
    tokens = nltk.word_tokenize(review)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(filtered)

# ===============================
# Prediction Function
# ===============================
def fake_news_det(news):
    cleaned_text = preprocess_text(news)
    vectorized = vectorizer.transform([cleaned_text])
    prediction = loaded_model.predict(vectorized)
    return prediction

# ===============================
# Flask Routes
# ===============================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news = request.form.get('news')
        if not news or not news.strip():
            return render_template('prediction.html', prediction_text="⚠️ Please enter some text.")

        pred = fake_news_det(news)
        if pred[0] == 1:
            result = "Prediction of the News: 📰 Looking Fake News"
        else:
            result = "Prediction of the News: 📰 Looking Real News"

        return render_template('prediction.html', prediction_text=result)

    except Exception as e:
        print(f"Error in /predict: {e}")
        return render_template('prediction.html', prediction_text="❌ Internal Error: " + str(e))

# ===============================
# Main entry point
# ===============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render dynamic port
    app.run(host='0.0.0.0', port=port, debug=True)
