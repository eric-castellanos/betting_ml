"""
Unit tests for src/utils/optuna/base.py
"""
import numpy as np
from src.utils.optuna import base

def test_run_study_returns_best_params():
    """
    Test that run_study returns best params, value, and study object.
    """
    def objective(trial):
        return trial.suggest_float("x", -1, 1) ** 2
    best_params, best_value, study = base.run_study(objective, n_trials=5)
    assert isinstance(best_params, dict)
    assert isinstance(best_value, float)
    assert hasattr(study, "best_params")


def test_compute_metrics_basic():
    """
    Test compute_metrics returns correct MAE, RMSE, R2 for simple arrays.
    """
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 2])
    metrics = base.compute_metrics(y_true, y_pred)
    assert np.isclose(metrics["mae"], 0.333, atol=1e-2)
    assert np.isclose(metrics["rmse"], 0.577, atol=1e-2)
    assert np.isclose(metrics["r2"], 0.5, atol=1e-2)


def test_compute_metrics_spread_total():
    """
    Test compute_metrics returns spread and total metrics when home/away provided.
    """
    y_true_home = np.array([10, 20])
    y_true_away = np.array([7, 17])
    y_pred_home = np.array([11, 19])
    y_pred_away = np.array([8, 16])
    y_true = np.array([1, 2])
    y_pred = np.array([1, 2])
    metrics = base.compute_metrics(
        y_true=y_true, y_pred=y_pred,
        y_true_home=y_true_home, y_true_away=y_true_away,
        y_pred_home=y_pred_home, y_pred_away=y_pred_away
    )
    assert "spread_mae" in metrics
    assert "total_mae" in metrics
