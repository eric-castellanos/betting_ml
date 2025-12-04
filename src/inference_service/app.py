from fastapi import FastAPI, HTTPException

from src.inference_service.model_loader import load_model, predict
from src.inference_service.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="NFL Spread Prediction API", version="1.0")

model = load_model()


@app.post("/predict", response_model=PredictionResponse)
async def predict_spread(payload: PredictionRequest):
    try:
        return predict(model, payload)
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
