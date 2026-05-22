import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_positive_sentiment():
    response = client.post("/predict", json={"text": "I love this product, it is amazing!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5

def test_negative_sentiment():
    response = client.post("/predict", json={"text": "This is terrible and I hate it."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"

def test_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422   # Pydantic validation error

def test_long_text():
    long_text = "good " * 1000
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code in [200, 422]