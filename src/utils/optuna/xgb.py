import numpy as np
import optuna
import polars as pl
import mlflow
from xgboost import XGBRegressor

from src.utils.optuna.base import compute_metrics, log_trial_to_mlflow


def xgb_search_space(trial: optuna.trial.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def _compute_split_metrics(df: pl.DataFrame, preds: np.ndarray, target_col: str, is_home_col: str) -> dict:
    y_true = df[target_col].to_numpy()
    y_pred = np.asarray(preds)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    game_view = (
        df.select(
            [
                pl.col("game_id"),
                pl.col(is_home_col),
                pl.col(target_col),
                pl.Series(name="prediction_temp", values=y_pred),
            ]
        )
        .group_by("game_id")
        .agg(
            [
                pl.col("prediction_temp").filter(pl.col(is_home_col) == 1).first().alias("home_pred"),
                pl.col("prediction_temp").filter(pl.col(is_home_col) == 0).first().alias("away_pred"),
                pl.col(target_col).filter(pl.col(is_home_col) == 1).first().alias("home_actual"),
                pl.col(target_col).filter(pl.col(is_home_col) == 0).first().alias("away_actual"),
            ]
        )
        .drop_nulls()
    )

    home_pred = game_view["home_pred"].to_numpy()
    away_pred = game_view["away_pred"].to_numpy()
    home_actual = game_view["home_actual"].to_numpy()
    away_actual = game_view["away_actual"].to_numpy()

    metrics = compute_metrics(
        y_true_f,
        y_pred_f,
        y_true_home=home_actual,
        y_true_away=away_actual,
        y_pred_home=home_pred,
        y_pred_away=away_pred,
    )
    return metrics


def xgb_objective_factory(
    X_train,
    y_train,
    X_val,
    y_val,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    target_col: str,
    is_home_col: str,
    mlflow_enabled: bool = False,
):
    def objective(trial: optuna.trial.Trial):
        params = xgb_search_space(trial)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_preds = model.predict(X_train)
        preds = model.predict(X_val)
        train_metrics = _compute_split_metrics(train_df, train_preds, target_col, is_home_col)
        val_metrics = _compute_split_metrics(val_df, preds, target_col, is_home_col)

        if mlflow_enabled:
            with mlflow.start_run(nested=True, run_name=f"optuna-trial-{trial.number}"):
                mlflow.log_params(params)
                for k, v in train_metrics.items():
                    mlflow.log_metric(f"train_{k}", v)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.log_metric("trial_number", trial.number)

        return val_metrics["mae"]

    return objective
