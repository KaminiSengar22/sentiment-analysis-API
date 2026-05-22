import pandas as pd
import nltk
import joblib
import os
import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')

# ── 1. Load dataset ──────────────────────────────────────────────
print("Loading dataset... (may take 1-2 mins first time)")
dataset = load_dataset("stanfordnlp/imdb")
train_df = pd.DataFrame(dataset['train'])
test_df  = pd.DataFrame(dataset['test'])

train_df['sentiment'] = train_df['label'].map({0: 'negative', 1: 'positive'})
test_df['sentiment']  = test_df['label'].map({0: 'negative', 1: 'positive'})

# ── 2. Clean text ────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

print("Cleaning text...")
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text']  = test_df['text'].apply(clean_text)

# ── 3. Train ─────────────────────────────────────────────────────
print("Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf',   LogisticRegression(max_iter=1000, C=1.0))
])
pipeline.fit(train_df['clean_text'], train_df['sentiment'])

# ── 4. Evaluate ──────────────────────────────────────────────────
y_pred = pipeline.predict(test_df['clean_text'])
print("\nClassification Report:")
print(classification_report(test_df['sentiment'], y_pred))

# ── 5. Save model ────────────────────────────────────────────────
os.makedirs('app/ml_models', exist_ok=True)
joblib.dump(pipeline, 'app/ml_models/sentiment_model.pkl')
print("\nModel saved successfully ✅")