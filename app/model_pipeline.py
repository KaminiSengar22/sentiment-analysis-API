import joblib
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

MODEL_PATH = Path(__file__).parent / "ml_models" / "sentiment_model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def predict_sentiment(text: str, model) -> dict:
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    confidence = round(float(max(probabilities)), 4)

    return {
        "sentiment": prediction,
        "confidence": confidence
    }