import logging

import numpy as np
import polars as pl
import polars.exceptions as ple
import pytest

from src.utils.model_utils import compute_spreads, evaluate_xgboost_performance, train_xgboost_model


@pytest.fixture
def simple_training_data():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    y = np.array([1.0, 2.0, 2.0, 3.0])
    return X, y


@pytest.fixture
def game_df():
    return pl.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "is_home": [1, 0, 1, 0],
            "points": [24, 17, 30, 20],
            "final_home_score": [24, 24, 30, 30],
            "final_away_score": [17, 17, 20, 20],
        }
    )


@pytest.fixture
def game_predictions():
    return np.array([27.0, 20.0, 28.0, 21.0])


def test_train_xgboost_model_trains_and_is_deterministic(simple_training_data):
    X, y = simple_training_data
    params = {"n_estimators": 5, "max_depth": 2, "random_state": 0, "verbosity": 0}

    model = train_xgboost_model(X, y, params)
    preds = model.predict(X)

    model_repeat = train_xgboost_model(X, y, params)
    preds_repeat = model_repeat.predict(X)

    assert preds.shape == y.shape
    assert np.allclose(preds, preds_repeat)
    assert model.n_estimators == 5


def test_evaluate_xgboost_performance_returns_expected_metrics(game_df, game_predictions):
    logger = logging.getLogger("test-xgb-eval")

    metrics = evaluate_xgboost_performance(
        game_df,
        predictions=game_predictions,
        target_col="points",
        game_id_col="game_id",
        is_home_col="is_home",
        logger=logger,
    )

    expected_keys = {"mae", "rmse", "r2", "spread_mae", "spread_abs_mae", "spread_rmse", "total_mae", "total_rmse"}
    assert expected_keys.issubset(metrics)
    assert metrics["mae"] == pytest.approx(2.25, rel=0.05)
    assert metrics["spread_mae"] == pytest.approx(1.5, rel=0.05)
    assert metrics["total_mae"] == pytest.approx(3.5, rel=0.05)


def test_compute_spreads_requires_score_columns(game_predictions):
    df_missing = pl.DataFrame({"game_id": [1], "is_home": [1]})
    preds = pl.Series("preds", game_predictions[:1])

    with pytest.raises(ple.ColumnNotFoundError):
        compute_spreads(df_missing, preds, is_home_col="is_home")


def test_compute_spreads_adds_predicted_spread(game_df, game_predictions):
    preds = pl.Series("preds", game_predictions)

    result = compute_spreads(game_df, preds, is_home_col="is_home")

    assert {"predicted_spread", "actual_spread", "actual_total"}.issubset(result.columns)

    game1 = result.filter(pl.col("game_id") == 1)
    assert float(game1["predicted_spread"][0]) == pytest.approx(7.0)
    assert float(game1["actual_spread"][0]) == pytest.approx(7.0)
    assert float(game1["actual_total"][0]) == pytest.approx(41.0)
