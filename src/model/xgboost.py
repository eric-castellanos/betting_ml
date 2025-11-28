"""
Leak-free training and evaluation pipeline for XGBoost using Polars features.
"""

from typing import Optional
import logging
import sys

import click
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

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

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def drop_score_columns(df: pl.DataFrame) -> pl.DataFrame:
    score_cols = {
        "final_home_score",
        "final_away_score",
        "home_score",
        "away_score",
        "posteam_score",
        "defteam_score",
        "score_differential",
        "score_differential_post",
    }
    to_drop = [c for c in score_cols if c in df.columns and c != "points_scored"]
    return df.drop(to_drop) if to_drop else df.clone()


def create_points_scored_target(df: pl.DataFrame, is_home_col: str) -> pl.DataFrame:
    if is_home_col not in df.columns:
        raise ValueError(f"Column '{is_home_col}' not found in DataFrame.")
    return df.with_columns(
        pl.when(pl.col(is_home_col) == 1)
        .then(pl.col("final_home_score"))
        .otherwise(pl.col("final_away_score"))
        .alias("points_scored")
    )


def time_based_train_test_split(
    df: pl.DataFrame,
    season_col: str,
    cutoff_season: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if season_col not in df.columns:
        raise ValueError(f"Column '{season_col}' not found in DataFrame.")

    train_df = df.filter(pl.col(season_col) < cutoff_season)
    test_df = df.filter(pl.col(season_col) == cutoff_season)

    if train_df.height == 0:
        raise ValueError(f"No training rows found for seasons < {cutoff_season}.")
    if test_df.height == 0:
        raise ValueError(f"No test rows found for season == {cutoff_season}.")

    train_seasons = sorted(train_df.select(pl.col(season_col).unique()).to_series().to_list())
    test_seasons = sorted(test_df.select(pl.col(season_col).unique()).to_series().to_list())

    logger.info(
        "Time-based split complete",
        extra={
            "cutoff_season": cutoff_season,
            "train_rows": train_df.height,
            "test_rows": test_df.height,
            "train_seasons": train_seasons,
            "test_seasons": test_seasons,
        },
    )

    return train_df, test_df


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


def winsorize_train_and_apply_to_test(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cols: list[str],
    lower: float,
    upper: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(f"Invalid quantile bounds: lower={lower}, upper={upper}")

    present = [c for c in cols if c in train_df.columns]
    if not present:
        logger.info("No winsorization columns present", extra={"lower": lower, "upper": upper})
        return train_df.clone(), test_df.clone()

    bounds = train_df.select(
        [pl.col(c).quantile(lower).alias(f"{c}_low") for c in present]
        + [pl.col(c).quantile(upper).alias(f"{c}_high") for c in present]
    ).to_dict(as_series=False)

    train_out = train_df
    test_out = test_df

    for c in present:
        low = bounds[f"{c}_low"][0]
        high = bounds[f"{c}_high"][0]
        train_out = train_out.with_columns(pl.col(c).clip(low, high))
        if c in test_out.columns:
            test_out = test_out.with_columns(pl.col(c).clip(low, high))

    logger.info(
        "Winsorized train/test",
        extra={"columns": present, "lower": lower, "upper": upper},
    )

    return train_out, test_out


def encode_categoricals_train_and_apply_to_test(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    is_home_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    team_lookup = pl.DataFrame(
        {
            "posteam": train_df.select(pl.col("posteam").unique()).to_series().to_list(),
            "posteam_id": list(range(len(train_df.select(pl.col("posteam").unique()).to_series().to_list()))),
        }
    )

    def build_base(df: pl.DataFrame, split: str) -> pl.DataFrame:
        cols = []
        if "home_away_flag" in df.columns:
            cols.append((pl.col("home_away_flag") == "home").cast(pl.Int8).alias(is_home_col))
        elif is_home_col in df.columns:
            cols.append(pl.col(is_home_col).cast(pl.Int8).alias(is_home_col))
        else:
            raise ValueError("Missing home indicator: expected 'home_away_flag' or the provided is_home_col.")
        if "surface" in df.columns:
            cols.append((pl.col("surface") == "turf").cast(pl.Int8).alias("is_turf"))
        out = df.with_columns(cols).drop(["home_away_flag", "surface"], strict=False)
        out = out.join(team_lookup, on="posteam", how="left")
        if split == "test":
            unseen_mask = out["posteam_id"].is_null()
            if unseen_mask.any():
                unseen = out.filter(unseen_mask).select("posteam").unique().to_series().to_list()
                logger.warning("Unseen posteam categories in test data", extra={"unseen_posteams": sorted(unseen)})
            out = out.with_columns(pl.col("posteam_id").fill_null(-1))
        return out

    train_base = build_base(train_df, split="train")
    test_base = build_base(test_df, split="test")

    roof_categories = [
        c for c in train_df.select(pl.col("roof_type").unique()).to_series().to_list() if c is not None
    ]

    def add_roof_dummies(df: pl.DataFrame, is_train: bool) -> pl.DataFrame:
        if "roof_type" not in df.columns or not roof_categories:
            return df.drop("roof_type", strict=False)
        dummy_cols = [
            pl.when(pl.col("roof_type") == cat).then(1).otherwise(0).alias(f"roof_type_{cat}")
            for cat in roof_categories
        ]
        if not is_train:
            unseen = {
                val for val in df.select(pl.col("roof_type").unique()).to_series().to_list() if val is not None
            } - set(roof_categories)
            if unseen:
                logger.warning("Unseen roof_type categories in test data", extra={"unseen_roof_types": sorted(unseen)})
        out = df.with_columns(dummy_cols)
        return out.drop("roof_type", strict=False)

    train_encoded = add_roof_dummies(train_base, is_train=True)
    test_encoded = add_roof_dummies(test_base, is_train=False)

    logger.info(
        "Encoded categoricals for train/test",
        extra={"roof_categories": roof_categories},
    )

    return train_encoded, test_encoded


def prepare_model_data(df: pl.DataFrame, target: str) -> tuple[pl.DataFrame, np.ndarray]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    cleaned = drop_score_columns(df)
    feature_candidates = cleaned.drop(target)
    numeric_types = list(pl.NUMERIC_DTYPES) + [pl.Boolean]
    numeric_cols = feature_candidates.select(pl.col(numeric_types)).columns

    if not numeric_cols:
        raise ValueError("No numeric feature columns available for training.")

    dropped_cols = [c for c in feature_candidates.columns if c not in numeric_cols]
    X = feature_candidates.select(numeric_cols)
    y = cleaned[target].to_numpy()

    logger.info(
        "Prepared model data",
        extra={
            "target": target,
            "n_numeric_features": len(numeric_cols),
            "dropped_columns": dropped_cols,
        },
    )
    return X, y


def train_xgboost_model(X, y, params) -> XGBRegressor:
    model = XGBRegressor(**params)
    logger.info(
        "Training xgboost model",
        extra={"rows": len(X), "features": X.shape[1], "params": params},
    )
    model.fit(X, y)
    return model


def evaluate_xgboost_performance(
    df: pl.DataFrame,
    predictions: np.ndarray,
    target_col: str,
    game_id_col: str,
    is_home_col: str,
    logger: logging.Logger,
) -> None:
    y_true = df[target_col].to_numpy()
    y_pred = np.asarray(predictions)
    team_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[team_mask]
    y_pred_f = y_pred[team_mask]
    team_mae = mean_absolute_error(y_true_f, y_pred_f)
    team_rmse = mean_squared_error(y_true_f, y_pred_f) ** 0.5
    team_r2 = r2_score(y_true_f, y_pred_f) if len(y_true_f) > 1 else float("nan")

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
    logger.info("XGBoost performance metrics: %s", metrics_payload, extra=metrics_payload)


def train_and_evaluate(
    df: pl.DataFrame,
    season_col: str,
    cutoff_season: int,
    is_home_col: str,
    winsor_cols: list[str],
    lower: float,
    upper: float,
    logger: logging.Logger,
) -> None:
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

    train_df, test_df = winsorize_train_and_apply_to_test(train_df, test_df, cols=winsor_cols, lower=lower, upper=upper)
    train_df, test_df = encode_categoricals_train_and_apply_to_test(train_df, test_df)

    X_train, y_train = prepare_model_data(train_df, "points_scored")
    X_test, _ = prepare_model_data(test_df, "points_scored")

    X_train_pd = X_train.to_pandas()
    X_test_pd = X_test.to_pandas()

    model = train_xgboost_model(X_train_pd, y_train, DEFAULT_XGB_PARAMS)
    preds_train = model.predict(X_train_pd)
    preds_test = model.predict(X_test_pd)

    logger.info("Evaluating train split", extra={"split": "train"})
    evaluate_xgboost_performance(
        train_df,
        predictions=preds_train,
        target_col="points_scored",
        game_id_col="game_id",
        is_home_col=is_home_col,
        logger=logger,
    )
    logger.info("Evaluating test split", extra={"split": "test"})
    evaluate_xgboost_performance(
        test_df,
        predictions=preds_test,
        target_col="points_scored",
        game_id_col="game_id",
        is_home_col=is_home_col,
        logger=logger,
    )
    logger.info("Training and evaluation complete", extra={"cutoff_season": cutoff_season})


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
        train_and_evaluate(
            df=features_df,
            season_col=season_col,
            cutoff_season=cutoff_season,
            is_home_col=is_home_col,
            winsor_cols=list(cols),
            lower=lower,
            upper=upper,
            logger=logger,
        )
    except Exception:
        logger.exception("Failed to load or winsorize features")
        sys.exit(1)


if __name__ == "__main__":
    main()
