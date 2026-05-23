import pytest
from fastapi.testclient import TestClient
from app.main import app

# This tells TestClient to run lifespan (startup/shutdown) so model loads
client = TestClient(app, raise_server_exceptions=False)

@pytest.fixture(autouse=True, scope="session")
def start_app():
    with TestClient(app) as c:
        yield c

def test_root(start_app):
    response = start_app.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health(start_app):
    response = start_app.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True

def test_positive_sentiment(start_app):
    response = start_app.post("/predict", json={"text": "I love this product, it is absolutely amazing!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5

def test_negative_sentiment(start_app):
    response = start_app.post("/predict", json={"text": "This is terrible, I hate it completely."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"

def test_empty_text(start_app):
    response = start_app.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_confidence_range(start_app):
    response = start_app.post("/predict", json={"text": "This was an okay experience."})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0