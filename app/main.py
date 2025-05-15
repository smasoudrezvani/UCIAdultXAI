from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from app.dashboard import dash_app
from fastapi.middleware.wsgi import WSGIMiddleware

from pydantic import BaseModel
from typing import Optional

from app.pipeline import MLPipeline
from app.logger import logger
from app.config import settings
from fastapi.responses import JSONResponse

app = FastAPI(
    title="ML InsightHub API",
    description="Predict income class using multiple ML models",
    version="0.1.0"
)

# Enable CORS (optional but often needed for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse (BaseModel) :
    model: str
    accuracy: float
    report: str

@app.get("/predict", response_model=PredictResponse)
def predict_income (model_type: Optional[str] = Query("logistic", enum=["logistic", "randomforest", "xgboost", "catboost"])) :
    """
    Run ML pipeline using selected model and return classification metrics.
    """
    try:
        logger.info(f"API call received: /predict?model_type={model_type}")
        pipeline = MLPipeline(model_type=model_type)
        result = pipeline.run()
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        return {
            "model": model_type,
            "accuracy": 0.0,
            "report": f"Failed to generate prediction: {e}"
        }

@app.get("/")
def root():
    return {"message": "Welcome to ML InsightHub API!"}


# Mount Dash at slash dashboard (/dashboard)
app.mount("/dashboard", WSGIMiddleware(dash_app.server))