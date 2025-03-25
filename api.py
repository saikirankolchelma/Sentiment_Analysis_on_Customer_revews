from fastapi import FastAPI
from pydantic import BaseModel
from model import SentimentModel
from data_processing import DataProcessor
from utils import setup_logging


logger = setup_logging()
app = FastAPI()
data_processor = DataProcessor()
model = SentimentModel()

logger.info("Using pre-trained model or freshly initialized model.")

class ReviewRequest(BaseModel):
    review_text: str

@app.post("/predict/")
async def predict_sentiment(request: ReviewRequest):
    try:
        sentiment, confidence = model.predict(request.review_text)
        data_processor.store_review(request.review_text, sentiment, confidence)
        return {"review": request.review_text, "sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error in API prediction: {e}")
        raise

@app.get("/reviews/")
async def get_reviews():
    try:
        reviews = data_processor.get_all_reviews()
        return [{"id": r[0], "review_text": r[1], "sentiment": r[2], "confidence": r[3]} for r in reviews]
    except Exception as e:
        logger.error(f"Error fetching reviews: {e}")
        raise