"""Model loading and prediction utilities for the inference service."""

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
    """Fallback model used when MLflow loading fails."""

    # pylint: disable=too-few-public-methods
    def predict(self, df: pd.DataFrame) -> list[float]:
        """Return zeroed predictions for a given dataframe."""
        return [0.0 for _ in range(len(df))]


def load_model(model_uri: str = DEFAULT_MODEL_URI) -> Any:
    """
    Load an MLflow model; return a dummy model on failure.

    Broad exceptions are intentional to avoid breaking the service on startup.
    """
    # pylint: disable=broad-exception-caught
    try:
        logger.info("Loading MLflow model", extra={"model_uri": model_uri})
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception:
        logger.exception("Failed to load MLflow model, falling back to dummy model")
        return DummyModel()


def _model_version(model: Any) -> str:
    """Extract a model version identifier (run_id) when available."""
    # pylint: disable=broad-exception-caught
    try:
        metadata = getattr(model, "metadata", None)
        run_id = getattr(metadata, "run_id", None)
        if run_id:
            return str(run_id)
    except Exception:
        logger.exception("Failed to extract model version from model metadata")
    return os.getenv("MLFLOW_MODEL_VERSION", "unknown")


def predict(model: Any, payload: PredictionRequest) -> PredictionResponse:
    """
    Run prediction for a validated request payload.

    Feature names are aligned to the XGBoost booster when available.
    """
    features = build_dummy_features(payload)
    expected_cols: list[str] | None = None
    # pylint: disable=protected-access,broad-exception-caught
    try:
        expected_cols = model._model_impl.xgb_model.get_booster().feature_names
    except Exception:
        logger.warning("Could not extract feature names from model, proceeding without reindexing")
        expected_cols = None

    df = pd.DataFrame([features])
    if expected_cols:
        df = df.reindex(columns=expected_cols, fill_value=0.0)
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except Exception:
        logger.exception(
            "Non-numeric feature values encountered during prediction",
            extra={
                "features": {
                    col: df[col].iloc[0] for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
                }
            },
        )
        raise
    df = df.fillna(0.0)

    preds = model.predict(df)
    predicted_spread = float(preds[0]) if len(preds) else 0.0

    return PredictionResponse(
        predicted_spread=predicted_spread,
        model_version=_model_version(model),
        input=payload,
    )
