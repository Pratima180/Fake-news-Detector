# ===============================================
# Fake News Detection Web App
# Developed by: Pratima Sahu
# College: Madhav Institute of Technology and Science (EEIoT)
# ===============================================

from flask import Flask, render_template, request
import os, re, pickle, nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------------------------
# NLTK setup — fixes "Resource wordnet not found"
# -----------------------------------------------
nltk_data_dir = "./nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download all required NLTK resources safely
required_pkgs = ["wordnet", "omw-1.4", "stopwords", "punkt", "punkt_tab"]
for pkg in required_pkgs:
    try:
        nltk.data.find(f"corpora/{pkg}")  # corpus-based check
    except LookupError:
        try:
            nltk.data.find(f"tokenizers/{pkg}")  # tokenizer-based check
        except LookupError:
            nltk.download(pkg, download_dir=nltk_data_dir)

# -----------------------------------------------
# Flask initialization
# -----------------------------------------------
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# -----------------------------------------------
# Load model and vectorizer
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
# Preprocessing tools
# -----------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(news):
    """Clean and lemmatize input text"""
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    tokens = nltk.word_tokenize(review)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(filtered)

# -----------------------------------------------
# Prediction logic
# -----------------------------------------------
def predict_fake_news(news):
    cleaned = preprocess_text(news)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]

    # Confidence using model probabilities (if available)
    try:
        probs = model.predict_proba(vectorized)[0]
        confidence = round(max(probs) * 100, 2)
    except:
        confidence = None

    label = "📰 Real News" if pred == 0 else "🚨 Fake News"
    return label, confidence

# -----------------------------------------------
# Flask routes
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
                                   prediction_text="⚠️ Please enter some news text.")

        label, confidence = predict_fake_news(news)

        if confidence:
            result = f"{label} <br><br>🔍 Confidence: <b>{confidence}%</b>"
        else:
            result = f"{label} <br><br>(Confidence unavailable)"

        return render_template('prediction.html', prediction_text=result)

    except Exception as e:
        print("Error in /predict:", e)
        return render_template('prediction.html',
                               prediction_text="❌ Internal Error: " + str(e))

# -----------------------------------------------
# Run the app
# -----------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

