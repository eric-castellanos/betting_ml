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


def test_create_points_scored_target_uses_home_flag():
    df = pl.DataFrame(
        {
            "final_home_score": [21, 17],
            "final_away_score": [10, 28],
            "is_home": [1, 0],
        }
    )
    result = model_utils.create_points_scored_target(df, "is_home")
    assert result["points_scored"].to_list() == [21, 28]


def test_create_points_scored_target_missing_flag_raises():
    df = pl.DataFrame({"final_home_score": [10], "final_away_score": [7]})
    with pytest.raises(ValueError):
        model_utils.create_points_scored_target(df, "is_home")


def test_time_based_train_test_split_returns_expected_rows():
    df = pl.DataFrame({"season": [2020, 2021, 2022], "value": [1, 2, 3]})
    train_df, test_df = model_utils.time_based_train_test_split(df, season_col="season", cutoff_season=2022)
    assert set(train_df["season"].to_list()) == {2020, 2021}
    assert set(test_df["season"].to_list()) == {2022}


def test_time_based_train_test_split_missing_column():
    df = pl.DataFrame({"year": [2020, 2021]})
    with pytest.raises(ValueError):
        model_utils.time_based_train_test_split(df, season_col="season", cutoff_season=2021)


def test_winsorize_train_and_apply_to_test_clips_extremes():
    train_df = pl.DataFrame({"value": [1.0, 2.0, 100.0]})
    test_df = pl.DataFrame({"value": [0.0, 50.0]})

    train_out, test_out = model_utils.winsorize_train_and_apply_to_test(
        train_df, test_df, cols=["value"], lower=0.0, upper=0.5
    )

    assert train_out["value"].max() == pytest.approx(2.0)
    assert test_out["value"].max() == pytest.approx(2.0)


def test_winsorize_train_and_apply_to_test_invalid_bounds():
    df = pl.DataFrame({"value": [1, 2]})
    with pytest.raises(ValueError):
        model_utils.winsorize_train_and_apply_to_test(df, df, cols=["value"], lower=0.6, upper=0.5)


def test_encode_categoricals_handles_unseen_posteam():
    train_df = pl.DataFrame(
        {
            "posteam": ["BUF", "KC"],
            "home_away_flag": ["home", "away"],
            "roof_type": ["indoors", "indoors"],
            "surface": ["turf", "grass"],
        }
    )
    test_df = pl.DataFrame(
        {
            "posteam": ["NE"],
            "home_away_flag": ["away"],
            "roof_type": ["domed"],
            "surface": ["grass"],
        }
    )

    train_enc, test_enc = model_utils.encode_categoricals_train_and_apply_to_test(
        train_df, test_df, is_home_col="is_home"
    )

    assert {"is_home", "posteam_id"}.issubset(train_enc.columns)
    assert {"is_home", "posteam_id"}.issubset(test_enc.columns)
    assert set(test_enc.filter(pl.col("posteam") == "NE")["posteam_id"].to_list()) == {-1}
    assert any(col.startswith("roof_type_") for col in train_enc.columns)


def test_prepare_model_data_filters_non_numeric_columns():
    df = pl.DataFrame(
        {
            "num1": [1, 2],
            "num2": [3.0, 4.0],
            "team": ["A", "B"],
            "target": [5, 6],
        }
    )
    X, y = model_utils.prepare_model_data(df, target="target")
    assert "team" not in X.columns
    assert X.shape[1] == 2
    assert y.tolist() == [5, 6]
