# ===============================================
# Fake News Detection Web App
# Developed by: Pratima Sahu
# College: Madhav Institute of Technology and Science (EEIoT)
# ===============================================

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # update name if 'vector.pkl'

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ===== Clean text =====
def clean_text(news):
    news = re.sub(r'[^a-zA-Z\s]', '', news)
    news = news.lower()
    tokens = nltk.word_tokenize(news)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# ===== Prediction logic =====
def predict_news(news_text):
    processed = clean_text(news_text)
    vector_input = vectorizer.transform([processed])
    prediction = model.predict(vector_input)[0]

    # Confidence
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector_input)[0]
        confidence = float(np.max(probs))
    else:
        confidence = 0.85  # fallback

    label = "FAKE" if prediction == 1 else "REAL"
    return label, round(confidence * 100, 2)

# ===== Routes =====
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'label': 'Invalid Input', 'confidence': 0.0})

        label, confidence = predict_news(text)
        return jsonify({'label': label, 'confidence': confidence})
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'label': 'Error', 'confidence': 0.0})

if __name__ == '__main__':
    app.run(debug=True)
