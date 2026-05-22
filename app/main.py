from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.schemas import SentimentRequest, SentimentResponse
from app.model_pipeline import load_model, predict_sentiment

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model loaded once at startup ──────────────────────────────────
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML model...")
    ml_model["sentiment"] = load_model()
    logger.info("Model loaded successfully ✅")
    yield
    ml_model.clear()

# ── App init ──────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts sentiment (positive/negative) from text using NLP.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Sentiment Analysis API is running 🚀"}

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "model_loaded": "sentiment" in ml_model}

@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
def predict(request: SentimentRequest):
    if "sentiment" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predict_sentiment(request.text, ml_model["sentiment"])
        return SentimentResponse(
            text=request.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            message=f"Prediction complete with {result['confidence']*100:.1f}% confidence"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")