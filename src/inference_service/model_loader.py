import logging
import os
from typing import Any

import mlflow
import pandas as pd

from src.inference_service.dummy_data import build_dummy_features
from src.inference_service.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/nfl_spread_model/1")


class DummyModel:
    def predict(self, df: pd.DataFrame) -> list[float]:
        return [0.0 for _ in range(len(df))]


def load_model(model_uri: str = DEFAULT_MODEL_URI) -> Any:
    try:
        logger.info("Loading MLflow model", extra={"model_uri": model_uri})
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception:
        logger.exception("Failed to load MLflow model, falling back to dummy model")
        return DummyModel()


def _model_version(model: Any) -> str:
    try:
        metadata = getattr(model, "metadata", None)
        run_id = getattr(metadata, "run_id", None)
        if run_id:
            return str(run_id)
    except Exception:
        pass
    return os.getenv("MLFLOW_MODEL_VERSION", "unknown")


def predict(model: Any, payload: PredictionRequest) -> PredictionResponse:
    features = build_dummy_features(payload)
    df = pd.DataFrame([features])
    preds = model.predict(df)
    predicted_spread = float(preds[0]) if len(preds) else 0.0

    return PredictionResponse(
        predicted_spread=predicted_spread,
        model_version=_model_version(model),
        input=payload,
    )
