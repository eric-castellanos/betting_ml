"""Run out-of-sample inference on 2025 NFL data using a registered XGBoost model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
import mlflow
import polars as pl
from mlflow.exceptions import MlflowException
from polars import exceptions as ple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.model_utils import compute_spreads


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics(
    df: pl.DataFrame,
    preds: pl.Series,
    spread_col: str = "spread",
    total_col: Optional[str] = "total_score",
) -> dict:
    if spread_col in df.columns:
        actual_spread = df[spread_col].to_numpy()
    elif {"final_home_score", "final_away_score"}.issubset(set(df.columns)):
        actual_spread = (df["final_home_score"] - df["final_away_score"]).to_numpy()
    else:
        raise ple.ColumnNotFoundError(f"{spread_col} not found and no final scores available to derive spread")
    predicted_spread = preds.to_numpy()

    metrics = {
        "rmse": mean_squared_error(actual_spread, predicted_spread) ** 0.5,
        "mae": mean_absolute_error(actual_spread, predicted_spread),
        "r2": r2_score(actual_spread, predicted_spread) if len(actual_spread) > 1 else float("nan"),
        "spread_rmse": mean_squared_error(actual_spread, predicted_spread) ** 0.5,
        "spread_mae": mean_absolute_error(actual_spread, predicted_spread),
    }

    if total_col in df.columns or {"final_home_score", "final_away_score"}.issubset(set(df.columns)):
        if total_col in df.columns:
            actual_total = df[total_col].to_numpy()
        else:
            actual_total = (df["final_home_score"] + df["final_away_score"]).to_numpy()

        if "home_score" in df.columns and "away_score" in df.columns:
            predicted_total = df["home_score"].to_numpy() + predicted_spread
        elif "final_home_score" in df.columns and "final_away_score" in df.columns:
            predicted_total = df["final_home_score"].to_numpy() + predicted_spread
        else:
            predicted_total = predicted_spread

        metrics["total_rmse"] = mean_squared_error(actual_total, predicted_total) ** 0.5
        metrics["total_mae"] = mean_absolute_error(actual_total, predicted_total)

    return metrics


def save_predictions(df: pl.DataFrame, preds: pl.Series, output_path: Path, spread_col: str = "spread") -> None:
    predicted_spread = preds.to_numpy()
    actual_spread = df[spread_col].to_numpy()
    out_df = df.with_columns(
        [
            pl.Series("predicted_spread", predicted_spread),
            pl.Series("prediction_error", predicted_spread - actual_spread),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    logger.info("Saved predictions", extra={"output_path": str(output_path), "rows": out_df.height})


def align_features(df: pl.DataFrame, expected_features: list[str]) -> pl.DataFrame:
    """
    Ensure the DataFrame has exactly the expected features in the right order.
    Missing columns are added with zeros; extra columns are dropped.
    """
    present_cols = set(df.columns)
    missing = [c for c in expected_features if c not in present_cols]
    extra = [c for c in df.columns if c not in expected_features]

    if missing:
        logger.warning("Adding missing feature columns with zeros", extra={"missing": missing})
        df = df.with_columns([pl.lit(0).alias(col) for col in missing])
    if extra:
        logger.info("Dropping unused feature columns", extra={"dropped": extra})
        df = df.drop(extra)

    return df.select(expected_features)


@click.command()
@click.option(
    "--input-path",
    default="data/processed/features_2025.parquet",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to 2025 features parquet.",
)
@click.option(
    "--output-path",
    default="data/predictions/predictions_2025.parquet",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to save predictions parquet.",
)
@click.option(
    "--model-uri",
    default="models:/nfl_spread_model/1",
    show_default=True,
    help="MLflow model URI to load for inference.",
)
def main(input_path: Path, output_path: Path, model_uri: str) -> None:
    """
    Perform out-of-sample inference on 2025 NFL data using a registered XGBoost model.
    """
    try:
        logger.info("Loading features", extra={"input_path": str(input_path)})
        df = pl.read_parquet(input_path)
    except FileNotFoundError:
        logger.exception("Input file not found", extra={"input_path": str(input_path)})
        raise
    except ple.PolarsError:
        logger.exception("Failed to read input parquet", extra={"input_path": str(input_path)})
        raise

    try:
        logger.info("Loading model from MLflow", extra={"model_uri": model_uri})
        model = mlflow.pyfunc.load_model(model_uri)
        expected_features: list[str] = []
        try:
            booster = model._model_impl.xgb_model.get_booster()  # type: ignore[attr-defined]
            expected_features = booster.feature_names or []
        except Exception:
            expected_features = []
    except MlflowException:
        logger.exception("Failed to load model from MLflow", extra={"model_uri": model_uri})
        raise

    try:
        logger.info("Running predictions", extra={"rows": df.height})
        if expected_features:
            df_aligned = align_features(df, expected_features)
            df_pd = df_aligned.to_pandas()
            logger.info("Aligning to model features", extra={"num_features": len(expected_features)})
        else:
            df_pd = df.to_pandas()
            numeric_cols = df_pd.select_dtypes(include=["number", "bool"]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns available for prediction.")
            logger.info("Using numeric columns for inference", extra={"num_features": len(numeric_cols)})
            df_pd = df_pd[numeric_cols]
        preds = model.predict(df_pd)
        preds_series = pl.Series(name="predicted_points", values=preds)
    except (ValueError, TypeError):
        logger.exception("Failed during prediction")
        raise

    try:
        df_enriched = compute_spreads(df, preds_series)
        metrics = compute_metrics(
            df_enriched,
            df_enriched["predicted_spread"],
            spread_col="actual_spread",
            total_col="actual_total",
        )
    except (ValueError, TypeError, ple.ColumnNotFoundError):
        logger.exception("Failed to compute metrics")
        raise

    try:
        logger.info("Logging metrics to MLflow", extra=metrics)
        mlflow.set_experiment("spread_inference")
        with mlflow.start_run(run_name="inference_2025"):
            mlflow.log_params({"model_uri": model_uri, "input_path": str(input_path)})
            mlflow.log_metrics(metrics)
    except MlflowException:
        logger.exception("Failed to log metrics to MLflow")
        raise

    try:
        save_predictions(df_enriched, df_enriched["predicted_spread"], output_path, spread_col="actual_spread")
    except (ple.PolarsError, OSError):
        logger.exception("Failed to save predictions", extra={"output_path": str(output_path)})
        raise

    logger.info("Inference complete", extra={"metrics": metrics})


if __name__ == "__main__":
    main()
