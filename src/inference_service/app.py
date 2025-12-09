"""FastAPI application for NFL spread prediction."""

from fastapi import FastAPI, HTTPException

from src.inference_service.model_loader import DummyModel, load_model, predict
from src.inference_service.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="NFL Spread Prediction API",
    version="1.0",
    description="Predicts NFL game spreads using an XGBoost model trained on historical data",
)

model = load_model()


@app.get("/")
async def root():
    """Root endpoint with basic API info."""
    return {"message": "NFL Spread Prediction API", "version": app.version}


@app.get("/health")
async def health():
    """Health probe indicating whether the model is loaded."""
    return {"status": "healthy", "model_loaded": not isinstance(model, DummyModel)}


@app.post("/predict", response_model=PredictionResponse)
async def predict_spread(payload: PredictionRequest):
    """Predict spread from a validated request payload."""
    try:
        return predict(model, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
