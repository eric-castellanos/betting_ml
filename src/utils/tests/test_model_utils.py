"""
Unit tests for src/utils/model_utils.py
"""
import pytest
import polars as pl
import os
from src.utils import model_utils

def test_save_feature_importance_plot(tmp_path):
    """
    Test saving a feature importance plot as PNG.
    """
    df = pl.DataFrame({"feature": ["a", "b"], "importance": [0.8, 0.2]})
    out_path = tmp_path / "plot.png"
    model_utils.save_feature_importance_plot(df, str(out_path))
    assert os.path.exists(out_path)

def test_save_feature_importance_plot_empty(tmp_path):
    """
    Test that no plot is saved for empty DataFrame.
    """
    df = pl.DataFrame({"feature": [], "importance": []})
    out_path = tmp_path / "plot.png"
    model_utils.save_feature_importance_plot(df, str(out_path))
    assert not os.path.exists(out_path)

def test_drop_score_columns():
    """
    Test dropping score columns from DataFrame.
    """
    df = pl.DataFrame({
        "final_home_score": [1],
        "final_away_score": [2],
        "other": [3]
    })
    result = model_utils.drop_score_columns(df)
    assert "final_home_score" not in result.columns
    assert "final_away_score" not in result.columns
    assert "other" in result.columns
