"""
Leak-free training and evaluation pipeline for XGBoost using Polars features.
"""

import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import mlflow
import mlflow.xgboost
import polars as pl
import xgboost as xgb

from src.utils.utils import load_data
from src.utils.model_utils import (
    save_feature_importance_plot,
    drop_score_columns,
    create_points_scored_target,
    time_based_train_test_split,
    winsorize_train_and_apply_to_test,
    encode_categoricals_train_and_apply_to_test,
    prepare_model_data,
    train_xgboost_model,
    evaluate_xgboost_performance,
)
from src.utils.optuna.xgb import xgb_objective_factory
from src.utils.optuna.base import run_study

DEFAULT_WINSOR_COLS = [
    "success_rate",
    "pct_drives_scored",
    "avg_drive_yards_penalized",
    "total_epa",
    "penalty_yards_total",
    "temp",
    "wind",
    "pass_epa_mean",
    "pass_yards_avg",
    "comp_pct",
    "rush_epa_mean",
    "rush_yards_avg",
]

# optimal parameters that minimized test_spread_mae were discovered via MLFlow
DEFAULT_XGB_PARAMS = {
    "n_estimators": 736,
    "learning_rate": 0.010169447982817973,
    "max_depth": 3,
    "subsample": 0.6046483846510212,
    "colsample_bytree": 0.6354945532725483,
    "min_child_weight": 3,
    "gamma": 4.640514341164363,
    "reg_lambda": 4.801569857474743,
    "reg_alpha": 4.428280402339789,
    "random_state": 42,
    "n_jobs": -1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def load_feature_dataset(
    year: int = 2020,
    bucket: str = "sports-betting-ml",
    output_key_template: str = "processed/features_2020-2024.parquet",
    filename: str = "features_2020-2024.parquet",
    local: bool = False,
    local_path: Optional[str] = None,
) -> pl.DataFrame:
    key = output_key_template.format(year=year)
    return load_data(bucket=bucket, key=key, filename=filename, local=local, local_path=local_path)




def train_and_evaluate(
    df: pl.DataFrame,
    season_col: str,
    cutoff_season: int,
    is_home_col: str,
    winsor_cols: list[str],
    lower: float,
    upper: float,
    logger: logging.Logger,
    tune: bool = False,
    mlflow_enabled: bool = False,
    register_model: bool = False,
) -> dict:
    train_df, test_df = time_based_train_test_split(df, season_col=season_col, cutoff_season=cutoff_season)

    if is_home_col not in train_df.columns:
        if "home_away_flag" not in train_df.columns or "home_away_flag" not in test_df.columns:
            raise ValueError(f"Column '{is_home_col}' or 'home_away_flag' not found for target creation.")
        train_df = train_df.with_columns((pl.col("home_away_flag") == "home").cast(pl.Int8).alias(is_home_col))
        test_df = test_df.with_columns((pl.col("home_away_flag") == "home").cast(pl.Int8).alias(is_home_col))
        train_df = train_df.drop("home_away_flag", strict=False)
        test_df = test_df.drop("home_away_flag", strict=False)

    train_df = create_points_scored_target(train_df, is_home_col=is_home_col)
    test_df = create_points_scored_target(test_df, is_home_col=is_home_col)

    train_df = drop_score_columns(train_df)
    test_df = drop_score_columns(test_df)

    winsor_cols_present = [c for c in winsor_cols if c in train_df.columns]
    train_df, test_df = winsorize_train_and_apply_to_test(
        train_df, test_df, cols=winsor_cols, lower=lower, upper=upper
    )
    train_df, test_df = encode_categoricals_train_and_apply_to_test(train_df, test_df, is_home_col=is_home_col)

    X_train, y_train = prepare_model_data(train_df, "points_scored")
    X_test, _ = prepare_model_data(test_df, "points_scored")

    X_train_pd = X_train.to_pandas()
    X_test_pd = X_test.to_pandas()

    best_params = None
    best_score = None
    best_params_path: Path | None = None

    if tune:
        split_idx = int(0.8 * len(X_train_pd))
        X_train_tune = X_train_pd.iloc[:split_idx]
        y_train_tune = y_train[:split_idx]
        X_val_tune = X_train_pd.iloc[split_idx:]
        y_val_tune = y_train[split_idx:]

        train_df_tune = train_df.slice(0, split_idx)
        val_df_tune = train_df.slice(split_idx)

        objective = xgb_objective_factory(
            X_train_tune,
            y_train_tune,
            X_val_tune,
            y_val_tune,
            train_df_tune,
            val_df_tune,
            target_col="points_scored",
            is_home_col=is_home_col,
            mlflow_enabled=mlflow_enabled,
        )

        best_params, best_score, study = run_study(
            objective,
            n_trials=30,
            study_name="xgb_optuna_search",
            direction="minimize",
        )

        best_params_path = Path("artifacts") / "best_params.json"
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

        logger.info("Optuna tuning complete", extra={"best_score": best_score, "best_params": best_params})
        tuned_params = {**DEFAULT_XGB_PARAMS, **best_params}
        model = train_xgboost_model(X_train_pd, y_train, tuned_params)
    else:
        model = train_xgboost_model(X_train_pd, y_train, DEFAULT_XGB_PARAMS)

    preds_train = model.predict(X_train_pd)
    preds_test = model.predict(X_test_pd)

    logger.info("Evaluating train split", extra={"split": "train"})
    train_metrics = evaluate_xgboost_performance(
        train_df,
        predictions=preds_train,
        target_col="points_scored",
        game_id_col="game_id",
        is_home_col=is_home_col,
        logger=logger,
    )
    logger.info("Evaluating test split", extra={"split": "test"})
    test_metrics = evaluate_xgboost_performance(
        test_df,
        predictions=preds_test,
        target_col="points_scored",
        game_id_col="game_id",
        is_home_col=is_home_col,
        logger=logger,
    )
    logger.info("Training and evaluation complete", extra={"cutoff_season": cutoff_season})

    feature_count = X_train_pd.shape[1]
    train_rows = train_df.height
    test_rows = test_df.height
    train_split_ratio = train_rows / (train_rows + test_rows)

    importances = model.get_booster().get_score(importance_type="gain")
    feature_importance_df = (
        pl.DataFrame({"feature": list(importances.keys()), "importance": list(importances.values())})
        .sort("importance", descending=True)
        if importances
        else pl.DataFrame({"feature": [], "importance": []})
    )

    feature_plot_path = Path("artifacts") / "feature_importance.png"
    save_feature_importance_plot(feature_importance_df, str(feature_plot_path))

    return {
        "model": model,
        "predictions": {"train": preds_train, "test": preds_test},
        "metrics": {"train": train_metrics, "test": test_metrics},
        "feature_importance": feature_importance_df,
        "feature_plot_path": feature_plot_path,
        "feature_count": feature_count,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_split_ratio": train_split_ratio,
        "winsor": {"lower": lower, "upper": upper, "columns": winsor_cols_present},
        "cutoff_season": cutoff_season,
        "best_params_path": best_params_path,
        "best_params": best_params,
        "best_score": best_score,
        "register_model": register_model,
    }


def log_mlflow_run(results: dict, is_home_col: str) -> None:
    params_to_log = {**DEFAULT_XGB_PARAMS}
    if results.get("best_params"):
        params_to_log.update(results["best_params"])
    mlflow.log_params(params_to_log)
    if results.get("best_score") is not None:
        mlflow.log_metric("optuna_best_mae", results["best_score"])

    mlflow.log_params(
        {
            "feature_count": results["feature_count"],
            "train_rows": results["train_rows"],
            "test_rows": results["test_rows"],
            "train_split_ratio": results["train_split_ratio"],
            "winsor_lower": results["winsor"]["lower"],
            "winsor_upper": results["winsor"]["upper"],
            "winsor_columns": ",".join(results["winsor"]["columns"]),
            "is_home_col": is_home_col,
            "categorical_encoding": "posteam_id mapping; roof_type one-hot; is_home flag",
        }
    )

    metrics = results["metrics"]
    for split_name, split_metrics in metrics.items():
        for k, v in split_metrics.items():
            mlflow.log_metric(f"{split_name}_{k}", v)

    mlflow.log_dict(metrics, "metrics.json")

    n_jobs_param = DEFAULT_XGB_PARAMS.get("n_jobs", 0)
    cores_used = os.cpu_count() if n_jobs_param == -1 else n_jobs_param

    runtime_info = {
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "polars_version": pl.__version__,
        "xgboost_version": xgb.__version__,
        "cpu_cores_available": os.cpu_count(),
        "xgboost_n_jobs": n_jobs_param,
        "xgboost_cpu_cores_used": cores_used,
    }

    mlflow.log_params(
        {
            "xgboost_cpu_cores_used": runtime_info["xgboost_cpu_cores_used"],
        }
    )
    mlflow.log_dict(runtime_info, "runtime_info.json")

    model_dump = results["model"].get_booster().get_dump()
    mlflow.log_text("\n".join(model_dump), "model_dump.txt")

    if results.get("best_params_path"):
        mlflow.log_artifact(str(results["best_params_path"]))

    feature_importance_df = results["feature_importance"]
    if not feature_importance_df.is_empty():
        fi_pd = feature_importance_df.to_pandas()
        try:
            mlflow.log_table(fi_pd, "feature_importance.json")
        except AttributeError:
            csv_path = Path("artifacts") / "feature_importance.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fi_pd.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

    if results["feature_plot_path"].exists():
        mlflow.log_artifact(str(results["feature_plot_path"]))

    if results.get("register_model"):
        logger.info("Registering model in MLflow Model Registry...")
        mlflow.xgboost.log_model(
            xgb_model=results["model"],
            artifact_path="model",
            registered_model_name="nfl_spread_model",
        )
    else:
        logger.info("Model registration disabled; skipping model logging.")

@click.command()
@click.option("--year", default=2020, show_default=True, type=int, help="Feature set year (if key template uses it).")
@click.option("--bucket", default="sports-betting-ml", show_default=True, help="S3 bucket for features.")
@click.option(
    "--output-key-template",
    default="processed/features_2020-2024.parquet",
    show_default=True,
    help="S3 key template for features parquet.",
)
@click.option("--filename", default="features_2020-2024.parquet", show_default=True, help="Features filename.")
@click.option("--local", is_flag=True, help="Load features from local_path instead of S3.")
@click.option("--local-path", default=None, type=str, help="Local directory containing features file.")
@click.option("--lower", default=0.01, show_default=True, type=float, help="Lower quantile for winsorizing.")
@click.option("--upper", default=0.99, show_default=True, type=float, help="Upper quantile for winsorizing.")
@click.option(
    "--col",
    "cols",
    multiple=True,
    default=DEFAULT_WINSOR_COLS,
    show_default=True,
    help="Columns to winsorize; pass multiple --col flags.",
)
@click.option("--season-col", default="season_year", show_default=True, help="Season column for time-based split.")
@click.option("--cutoff-season", default=2024, show_default=True, type=int, help="Cutoff season for test split.")
@click.option("--is-home-col", default="is_home", show_default=True, help="Column indicating home team flag.")
@click.option(
    "--mlflow/--no-mlflow",
    "mlflow_enabled",
    default=False,
    show_default=True,
    help="Enable MLflow experiment tracking.",
)
@click.option(
    "--tune/--no-tune",
    "tune",
    default=False,
    show_default=True,
    help="Enable Optuna hyperparameter tuning before training.",
)
@click.option(
    "--register-model/--no-register-model",
    "register_model",
    default=False,
    show_default=True,
    help="Log and register the trained model in MLflow.",
)
def main(
    year: int,
    bucket: str,
    output_key_template: str,
    filename: str,
    local: bool,
    local_path: Optional[str],
    lower: float,
    upper: float,
    cols: tuple[str, ...],
    season_col: str,
    cutoff_season: int,
    is_home_col: str,
    mlflow_enabled: bool,
    tune: bool,
    register_model: bool,
) -> None:
    """
    CLI to run leak-free time-based training and evaluation.
    """
    try:
        features_df = load_feature_dataset(
            year=year,
            bucket=bucket,
            output_key_template=output_key_template,
            filename=filename,
            local=local,
            local_path=local_path,
        )
        if mlflow_enabled:
            mlflow.set_experiment("nfl-xgb")
            with mlflow.start_run():
                results = train_and_evaluate(
                    df=features_df,
                    season_col=season_col,
                cutoff_season=cutoff_season,
                is_home_col=is_home_col,
                winsor_cols=list(cols),
                lower=lower,
                upper=upper,
                logger=logger,
                tune=tune,
                mlflow_enabled=mlflow_enabled,
                register_model=register_model,
            )
                log_mlflow_run(results, is_home_col=is_home_col)
        else:
            train_and_evaluate(
                df=features_df,
                season_col=season_col,
                cutoff_season=cutoff_season,
                is_home_col=is_home_col,
                winsor_cols=list(cols),
                lower=lower,
                upper=upper,
                logger=logger,
                tune=tune,
                mlflow_enabled=mlflow_enabled,
                register_model=register_model,
            )
    except Exception:
        logger.exception("Failed to load data or train/evaluate model")
        sys.exit(1)


if __name__ == "__main__":
    main()
