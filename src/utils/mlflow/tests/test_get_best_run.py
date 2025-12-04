"""
Unit tests for src/utils/mlflow/get_best_run.py
"""
import pytest
from unittest.mock import patch, MagicMock
from src.utils.mlflow import get_best_run

def test_find_best_run_returns_best():
    """
    Test that find_best_run returns the best run when runs exist.
    """
    mock_experiment = MagicMock(experiment_id="123")
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.iloc.__getitem__.return_value = {"run_id": "run_1", "metrics.val_mae": 0.1}
    with patch("src.utils.mlflow.get_best_run.mlflow.get_experiment_by_name", return_value=mock_experiment), \
         patch("src.utils.mlflow.get_best_run.mlflow.search_runs", return_value=mock_df):
        result = get_best_run.find_best_run("exp", "val_mae", False)
        assert result["run_id"] == "run_1"

def test_find_best_run_no_experiment():
    """
    Test that find_best_run raises ValueError if experiment not found.
    """
    with patch("src.utils.mlflow.get_best_run.mlflow.get_experiment_by_name", return_value=None):
        with pytest.raises(ValueError):
            get_best_run.find_best_run("exp", "val_mae", False)

def test_find_best_run_no_runs():
    """
    Test that find_best_run returns None if no runs found.
    """
    mock_experiment = MagicMock(experiment_id="123")
    mock_df = MagicMock()
    mock_df.empty = True
    with patch("src.utils.mlflow.get_best_run.mlflow.get_experiment_by_name", return_value=mock_experiment), \
         patch("src.utils.mlflow.get_best_run.mlflow.search_runs", return_value=mock_df):
        result = get_best_run.find_best_run("exp", "val_mae", False)
        assert result is None
