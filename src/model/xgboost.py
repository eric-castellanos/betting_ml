"""
Helpers for fetching and preparing feature data for the xgboost model.
"""

from typing import Optional
import logging
import sys

import click
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.utils.utils import load_data

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

def load_feature_dataset(
    year: int = 2020,
    bucket: str = "sports-betting-ml",
    output_key_template: str = "processed/features_{year}.parquet",
    filename: str = "2020_pbp_data.parquet",
    local: bool = False,
    local_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load the processed feature dataset that was written by the feature_engineering CLI.

    The defaults mirror the CLI options shown in feature_engineering.py so this will
    load from s3://sports-betting-ml/processed/features_{year}.parquet/<filename>.
    Set local=True and provide local_path to read from disk instead.
    """
    key = output_key_template.format(year=year)
    return load_data(bucket=bucket, key=key, filename=filename, local=local, local_path=local_path)


def winsorize_features(
    df: pl.DataFrame,
    cols: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """
    Winsorize selected feature columns at the given lower/upper quantiles.
    Missing columns are ignored.
    """
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(f"Invalid quantile bounds: lower={lower}, upper={upper}")

    present = [c for c in cols if c in df.columns]

    if not present:
        logger.warning("No winsorization columns present; returning original DataFrame.")
        return df

    bounds = df.select(
        [pl.col(c).quantile(lower).alias(f"{c}_low") for c in present]
        + [pl.col(c).quantile(upper).alias(f"{c}_high") for c in present]
    ).to_dict(as_series=False)

    out = df

    for c in present:
        low = bounds[f"{c}_low"][0]
        high = bounds[f"{c}_high"][0]
        out = out.with_columns(pl.col(c).clip(low, high))

    logger.info(
        "Winsorized columns=%s (lower=%.3f, upper=%.3f)",
        present,
        lower,
        upper,
    )

    return out


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Encode categorical fields for model input.
    """
    roof_cols = ["roof_type"]
    roof_present = [c for c in roof_cols if c in df.columns]

    base = (
        df.with_columns(
            [
                pl.col("posteam").cast(pl.Categorical).to_physical().alias("posteam"),
                (pl.col("home_away_flag") == "home").cast(pl.Int8).alias("is_home"),
                (pl.col("surface") == "turf").cast(pl.Int8).alias("is_turf"),
                pl.col("roof_type").cast(pl.Categorical),
            ]
        )
        .drop(["home_away_flag", "surface"])
    )
    encoded = base.to_dummies(columns=roof_present) if roof_present else base
    logger.info(
        "Encoded categoricals; added dummies for roof and binary flags for home/surface",
        extra={
            "posteam_dtype": str(encoded.schema.get("posteam")),
            "dummy_cols": [c for c in encoded.columns if any(c.startswith(f"{r}_") for r in roof_present)],
            "binary_cols": ["is_home", "is_turf"],
        },
    )
    return encoded


def prepare_model_data(df: pl.DataFrame, target: str) -> tuple:
    """
    Split out features/target for xgboost (Polars is supported directly).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    feature_candidates = df.drop(target)
    numeric_types = list(pl.NUMERIC_DTYPES) + [pl.Boolean]
    numeric_cols = feature_candidates.select(pl.col(numeric_types)).columns
    
    if not numeric_cols:
        raise ValueError("No numeric feature columns available for training.")

    X = feature_candidates.select(numeric_cols)
    y = df[target]

    logger.info(
        "Prepared model data",
        extra={
            "target": target,
            "n_features": len(numeric_cols),
            "dropped_cols": [c for c in feature_candidates.columns if c not in numeric_cols],
        },
    )
    return X, y

def train_xgboost_model(
    X,
    y,
    target: str,
    model_params: Optional[dict] = None,
):
    """
    Train an XGBoost regression model on the provided data.
    """
    params = model_params or {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)

    logger.info("Training xgboost model", extra={"task": "regression", "target": target, "rows": len(X), "features": X.shape[1]})
    model.fit(X, y)
    return model


def create_points_scored_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Derive points_scored per team row: use home_score when is_home==1, otherwise away_score.
    Returns a new DataFrame without mutating the input.
    """
    return df.with_columns(
        pl.when(pl.col("is_home") == 1)
        .then(pl.col("final_home_score"))
        .otherwise(pl.col("final_away_score"))
        .alias("points_scored")
    )


def evaluate_xgboost_performance(
    df: pl.DataFrame,
    predictions: np.ndarray,
    target_col: str,
    game_id_col: str,
    is_home_col: str,
    logger: logging.Logger,
) -> None:
    """
    Evaluate team-level and game-level (betting) metrics for an XGBoost regression model.
    """
    # Team-level metrics
    y_true = df[target_col].to_numpy()
    y_pred = np.asarray(predictions)
    team_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[team_mask]
    y_pred_f = y_pred[team_mask]
    team_mae = mean_absolute_error(y_true_f, y_pred_f)
    team_rmse = mean_squared_error(y_true_f, y_pred_f) ** 0.5
    team_r2 = r2_score(y_true_f, y_pred_f) if len(y_true_f) > 1 else float("nan")

    # Game-level metrics
    game_view = (
        df.select(
            [
                pl.col(game_id_col),
                pl.col(is_home_col),
                pl.col(target_col),
                pl.Series(name="prediction_temp", values=y_pred),
            ]
        )
        .group_by(game_id_col)
        .agg(
            [
                pl.col("prediction_temp").filter(pl.col(is_home_col) == 1).first().alias("home_pred"),
                pl.col("prediction_temp").filter(pl.col(is_home_col) == 0).first().alias("away_pred"),
                pl.col(target_col).filter(pl.col(is_home_col) == 1).first().alias("home_actual"),
                pl.col(target_col).filter(pl.col(is_home_col) == 0).first().alias("away_actual"),
            ]
        )
    )

    game_view = game_view.drop_nulls()

    spread_pred = (game_view["home_pred"] - game_view["away_pred"]).to_numpy()
    spread_actual = (game_view["home_actual"] - game_view["away_actual"]).to_numpy()
    total_pred = (game_view["home_pred"] + game_view["away_pred"]).to_numpy()
    total_actual = (game_view["home_actual"] + game_view["away_actual"]).to_numpy()

    spread_mask = np.isfinite(spread_pred) & np.isfinite(spread_actual)
    total_mask = np.isfinite(total_pred) & np.isfinite(total_actual)

    spread_mae = mean_absolute_error(spread_actual[spread_mask], spread_pred[spread_mask])
    spread_rmse = mean_squared_error(spread_actual[spread_mask], spread_pred[spread_mask]) ** 0.5
    total_mae = mean_absolute_error(total_actual[total_mask], total_pred[total_mask])
    total_rmse = mean_squared_error(total_actual[total_mask], total_pred[total_mask]) ** 0.5

    metrics_payload = {
        "team_metrics": {
            "mae": float(team_mae),
            "rmse": float(team_rmse),
            "r2": float(team_r2),
        },
        "betting_metrics": {
            "spread_mae": float(spread_mae),
            "spread_rmse": float(spread_rmse),
            "total_mae": float(total_mae),
            "total_rmse": float(total_rmse),
        },
        "num_games": int(game_view.height),
        "num_rows": int(df.height),
    }
    # Log both in the message (for stdout visibility) and keep structured payload in extra.
    logger.info("XGBoost performance metrics: %s", metrics_payload, extra=metrics_payload)


@click.command()
@click.option("--year", default=2020, show_default=True, type=int, help="Feature set year.")
@click.option("--bucket", default="sports-betting-ml", show_default=True, help="S3 bucket for features.")
@click.option(
    "--output-key-template",
    default="processed/features_{year}.parquet",
    show_default=True,
    help="S3 key template for features parquet.",
)
@click.option("--filename", default="2020_pbp_data.parquet", show_default=True, help="Features filename.")
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
@click.option("--target", required=True, default="points_scored", show_default=True, help="Target column for training.")
@click.option("--game-id-col", default="game_id", show_default=True, help="Game identifier column.")
@click.option("--test-size", default=0.2, show_default=True, type=float, help="Test split size fraction.")
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
    target: str,
    game_id_col: str,
    test_size: float,
) -> None:
    """
    CLI to load features and winsorize selected columns.
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
        winsorized = winsorize_features(features_df, list(cols), lower=lower, upper=upper)
        encoded = encode_categoricals(winsorized)
        with_target = create_points_scored_target(encoded) if target == "points_scored" else encoded
        X, y = prepare_model_data(with_target, target)

        indices = np.arange(len(with_target))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

        X_train = X[train_idx].to_pandas()
        X_test = X[test_idx].to_pandas()
        y_train = y[train_idx].to_numpy()
        y_test = y[test_idx].to_numpy()
        test_df = with_target[test_idx]
        train_df = with_target[train_idx]

        model = train_xgboost_model(X_train, y_train, target=target)
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        logger.info("Logging training performance metrics")
        evaluate_xgboost_performance(
            train_df,
            predictions=preds_train,
            target_col=target,
            game_id_col=game_id_col,
            is_home_col="is_home",
            logger=logger,
        )
        logger.info("Logging test performance metrics")
        evaluate_xgboost_performance(
            test_df,
            predictions=preds_test,
            target_col=target,
            game_id_col=game_id_col,
            is_home_col="is_home",
            logger=logger,
        )
        logger.info("Training and evaluation complete")
    except Exception:
        logger.exception("Failed to load or winsorize features")
        sys.exit(1)


if __name__ == "__main__":
    main()
