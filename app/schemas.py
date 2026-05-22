from pydantic import BaseModel, Field

class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        example="This movie was absolutely amazing!"
    )

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    message: str