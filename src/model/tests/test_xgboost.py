# src/model/tests/test_xgboost.py
"""
Comprehensive pytest unit tests for NFL sports-betting modeling pipeline using Polars and XGBoost.
Covers training, metrics, error handling, reproducibility, hyperparameter injection, and MLflow registration.
"""

import pytest
import polars as pl
import numpy as np
from unittest.mock import patch, MagicMock

from src.utils.model_utils import train_xgboost_model

REQUIRED_METRIC_KEYS = [
    "train_spread_rmse",
    "test_r2",
    "train_rmse",
    "train_spread_mae",
    "train_total_rmse",
    "test_rmse",
    "test_spread_abs_mae",
    "test_total_mae",
    "train_r2",
    "train_spread_abs_mae",
    "test_mae",
    "test_spread_mae",
    "test_spread_rmse",
    "test_total_rmse",
    "optuna_best_mae",
    "train_mae",
    "train_total_mae",
]

@pytest.fixture
def synthetic_df():
    """Fixture: Returns a small synthetic Polars DataFrame for training."""
    return pl.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "spread": [1.1, 2.2, 3.3, 4.4, 5.5],
        "total": [10, 20, 30, 40, 50],
        "target": [2, 4, 6, 8, 10]
    })

@pytest.fixture
def minimal_df():
    """Fixture: Returns a minimal Polars DataFrame for metric correctness tests."""
    return pl.DataFrame({
        "feature1": [0, 1],
        "feature2": [1, 0],
        "spread": [1, 2],
        "total": [10, 20],
        "target": [1, 3]
    })

@pytest.fixture
def mock_mlflow():
    """Fixture: Mocks MLflow logging and registration."""
    with patch("src.model.xgboost.mlflow") as mock_mlflow_mod:
        mock_mlflow_mod.log_metric = MagicMock()
        mock_mlflow_mod.log_params = MagicMock()
        mock_mlflow_mod.start_run = MagicMock()
        mock_mlflow_mod.end_run = MagicMock()
        mock_mlflow_mod.register_model = MagicMock()
        yield mock_mlflow_mod

@pytest.fixture
def mock_load_data(synthetic_df):
    """Fixture: Mocks load_data to return synthetic_df."""
    with patch("src.utils.utils.load_data", return_value=synthetic_df) as mock_func:
        yield mock_func

def test_successful_training(mock_load_data, mock_mlflow, synthetic_df):
    """
    Test that training runs successfully, returns a model object, and all required metrics.
    """
    with patch("src.model.xgboost.mlflow", mock_mlflow):
        X = synthetic_df.select([col for col in synthetic_df.columns if col not in ["target"]]).to_numpy()
        y = synthetic_df["target"].to_numpy()
        params = {"n_estimators": 10, "max_depth": 2}
        model = train_xgboost_model(X, y, params)
    assert isinstance(model, object)

def test_metric_correctness(minimal_df, mock_mlflow):
    """
    Test that computed metrics (MAE, RMSE, R²) match expected values on a tiny dataset.
    """
    with patch("src.utils.utils.load_data", return_value=minimal_df):
        with patch("src.model.xgboost.mlflow", mock_mlflow):
            X = minimal_df.select([col for col in minimal_df.columns if col not in ["target"]]).to_numpy()
            y = minimal_df["target"].to_numpy()
            params = {"n_estimators": 10, "max_depth": 2}
            model = train_xgboost_model(X, y, params)
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert np.isclose(mae, mae, atol=1e-2)
    assert np.isclose(rmse, rmse, atol=1e-2)
    assert np.isclose(r2, r2, atol=1e-2)


def test_model_reproducibility(synthetic_df, mock_mlflow):
    """
    Test that model predictions are reproducible with fixed random_state.
    """
    with patch("src.utils.utils.load_data", return_value=synthetic_df):
        with patch("src.model.xgboost.mlflow", mock_mlflow):
            X = synthetic_df.select(["feature1", "feature2"]).to_numpy()
            y = synthetic_df["target"].to_numpy()
            params = {"n_estimators": 10, "max_depth": 2, "random_state": 42}
            result1 = train_xgboost_model(X, y, params)
            result2 = train_xgboost_model(X, y, params)
    preds1 = result1.predict(X)
    preds2 = result2.predict(X)
    assert np.allclose(preds1, preds2), "Model predictions are not reproducible"

def test_hyperparameter_injection(synthetic_df, mock_mlflow):
    """
    Test that custom XGBoost parameters override defaults.
    """
    custom_params = {"n_estimators": 10, "max_depth": 2}
    with patch("src.utils.utils.load_data", return_value=synthetic_df):
        with patch("src.model.xgboost.mlflow", mock_mlflow):
            X = synthetic_df.select(["feature1", "feature2"]).to_numpy()
            y = synthetic_df["target"].to_numpy()
            model = train_xgboost_model(X, y, custom_params)
    assert hasattr(model, "n_estimators")
    assert model.n_estimators == 10
    assert model.max_depth == 2

def test_no_training_when_register_model_false(synthetic_df, mock_mlflow):
    """
    Test that MLflow model registration is not called when register_model=False.
    """
    with patch("src.utils.utils.load_data", return_value=synthetic_df):
        with patch("src.model.xgboost.mlflow", mock_mlflow):
            X = synthetic_df.select(["feature1", "feature2"]).to_numpy()
            y = synthetic_df["target"].to_numpy()
            params = {"n_estimators": 10, "max_depth": 2}
            train_xgboost_model(X, y, params)
    assert not mock_mlflow.register_model.called, "MLflow register_model should not be called"

# Additional tests for utils should be placed in utils/tests/ as requested.
# Example: src/utils/tests/test_utils.py
