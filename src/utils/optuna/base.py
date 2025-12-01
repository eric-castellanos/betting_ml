from typing import Any, Callable, Tuple

import mlflow
import optuna
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_study(
    objective: Callable[[optuna.trial.Trial], float],
    n_trials: int = 30,
    study_name: str = "optuna_study",
    direction: str = "minimize",
) -> Tuple[dict, float, optuna.Study]:
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value, study


def compute_metrics(
    y_true,
    y_pred,
    y_true_home=None,
    y_true_away=None,
    y_pred_home=None,
    y_pred_away=None,
) -> dict:
    base_rmse = mean_squared_error(y_true, y_pred) ** 0.5
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": base_rmse,
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan"),
    }

    if (
        y_true_home is not None
        and y_true_away is not None
        and y_pred_home is not None
        and y_pred_away is not None
    ):
        spread_true = y_true_home - y_true_away
        spread_pred = y_pred_home - y_pred_away
        total_true = y_true_home + y_true_away
        total_pred = y_pred_home + y_pred_away

        total_rmse = mean_squared_error(total_true, total_pred) ** 0.5
        spread_rmse = mean_squared_error(spread_true, spread_pred) ** 0.5

        metrics.update(
            {
                "spread_mae": mean_absolute_error(spread_true, spread_pred),
                "spread_abs_mae": float(np.mean(np.abs(spread_true - spread_pred))),
                "spread_rmse": spread_rmse,
                "total_mae": mean_absolute_error(total_true, total_pred),
                "total_rmse": total_rmse,
            }
        )

    return metrics


def log_trial_to_mlflow(params: dict, metrics: dict, trial_number: int) -> None:
    if mlflow.active_run() is None:
        return
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("trial_number", trial_number)
