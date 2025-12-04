"""
Unit tests for src/utils/optuna/xgb.py
"""
import numpy as np
import polars as pl
from src.utils.optuna import xgb

def test_xgb_search_space_keys():
    """
    Test that xgb_search_space returns all required hyperparameter keys.
    """
    class DummyTrial:
        def suggest_int(self, name, low, high): return 10
        def suggest_float(self, name, low, high, log=False): return 0.1
    trial = DummyTrial()
    params = xgb.xgb_search_space(trial)
    expected_keys = [
        "n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma",
        "reg_lambda", "reg_alpha", "subsample", "colsample_bytree", "random_state", "n_jobs", "tree_method"
    ]
    for key in expected_keys:
        assert key in params


def test__compute_split_metrics_returns_metrics():
    """
    Test that _compute_split_metrics returns a metrics dict for valid input.
    """
    df = pl.DataFrame({
        "game_id": [1, 1, 2, 2],
        "is_home": [1, 0, 1, 0],
        "target": [10, 7, 20, 17]
    })
    preds = np.array([11, 8, 19, 16])
    metrics = xgb._compute_split_metrics(df, preds, "target", "is_home")
    assert isinstance(metrics, dict)
    assert "spread_mae" in metrics
    assert "total_mae" in metrics
