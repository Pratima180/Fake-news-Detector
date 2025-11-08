# ===============================================
# Fake News Detection Web App
# Developed by: Pratima Sahu
# College: MITS Gwalior (EEIoT)
# ===============================================

from flask import Flask, render_template, request
import pickle
import nltk
import re
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
    nltk.download(pkg)

# -----------------------------------------------
# NLTK setup
# -----------------------------------------------
nltk_data_dir = "./nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

required_pkgs = ["wordnet", "omw-1.4", "stopwords", "punkt", "punkt_tab"]
for pkg in required_pkgs:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, download_dir=nltk_data_dir)

# -----------------------------------------------
# Initialize Flask
# -----------------------------------------------
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# -----------------------------------------------
# Load model & vectorizer
# -----------------------------------------------
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# -----------------------------------------------
# Preprocessing Function
# -----------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(news):
    news = re.sub(r'[^a-zA-Z\s]', '', news)
    news = news.lower()
    tokens = nltk.word_tokenize(news)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(filtered)

# -----------------------------------------------
# Prediction Function
# -----------------------------------------------
def predict_fake_news(news):
    cleaned = preprocess_text(news)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]

    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectorized)[0]
            confidence = round(max(probs) * 100, 2)
        elif hasattr(model, "decision_function"):
            raw_score = abs(model.decision_function(vectorized)[0])
            confidence = round(min(99.9, (1 / (1 + np.exp(-raw_score))) * 100), 2)
    except Exception as e:
        print("Confidence error:", e)

    label = "📰 Real News" if pred == 0 else "🚨 Fake News"
    return label, confidence

# -----------------------------------------------
# Flask Routes
# -----------------------------------------------
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
            return render_template('prediction.html',
                                   prediction_text="⚠️ Please enter some text.")

        label, confidence = predict_fake_news(news)

        if confidence:
            result = f"{label}<br><br>🔍 Confidence: <b>{confidence}%</b>"
        else:
            result = f"{label}<br><br>(Confidence unavailable)"

        return render_template('prediction.html', prediction_text=result)

    except Exception as e:
        print("Error in /predict:", e)
        return render_template('prediction.html',
                               prediction_text="❌ Internal Error: " + str(e))

# -----------------------------------------------
# Run Flask App
# -----------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
