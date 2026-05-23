# Sentiment Analysis REST API

A production-style REST API that predicts sentiment (positive/negative) from text input,
built with FastAPI and a scikit-learn NLP pipeline trained on 50,000 IMDB reviews.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![Tests](https://img.shields.io/badge/tests-6%20passed-brightgreen)

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| ML Model | Logistic Regression + TF-IDF |
| NLP | NLTK (stopword removal, text cleaning) |
| Server | Uvicorn (ASGI) |
| Testing | pytest + HTTPX |

## Project Structure

```
sentiment-api/
├── app/
│   ├── main.py            # FastAPI app & routes
│   ├── model_pipeline.py  # ML inference logic
│   ├── schemas.py         # Request/response models
│   ├── config.py          # Environment config
│   └── ml_models/         # Saved .pkl model
├── notebooks/
│   └── train_model.py     # Model training script
├── tests/
│   └── test_api.py        # Full test suite (6 tests)
├── conftest.py
└── requirements.txt
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-api.git
cd sentiment-analysis-api

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (first time only, takes ~3 mins)
python notebooks/train_model.py

# 5. Run the API
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| GET | /health | Model status |
| POST | /predict | Predict sentiment |

## Example

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely fantastic!"}'
```

**Response:**
```json
{
  "text": "This product is absolutely fantastic!",
  "sentiment": "positive",
  "confidence": 0.9732,
  "message": "Prediction complete with 97.3% confidence"
}
```

## Model Performance

- Dataset: IMDB Movie Reviews (50,000 samples)
- Algorithm: Logistic Regression with TF-IDF bigrams
- Accuracy: ~91% on test set

## Running Tests

```bash
pytest tests/ -v
```
