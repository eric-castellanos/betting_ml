from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import logging

logger = logging.getLogger(__name__)


def save_feature_importance_plot(
    feature_importance_df: pl.DataFrame,
    output_path: str,
    top_n: Optional[int] = None,
) -> None:
    """
    Save a bar plot of feature importance values sorted in descending order.

    Args:
        feature_importance_df: Polars DataFrame with columns ["feature", "importance"].
        output_path: Destination for the PNG plot.
        top_n: If provided, limit plot to the top N features.
    """
    if feature_importance_df.is_empty():
        return

    plot_df = feature_importance_df.sort("importance", descending=True)
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    df_pd = plot_df.to_pandas()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df_pd["feature"], df_pd["importance"], color="#4c6ef5")
    plt.xlabel("Gain (importance)")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
) -> dict:
    y_true = df[target_col].to_numpy()
    y_pred = np.asarray(predictions)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

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
        .drop_nulls()
    )

    home_pred = game_view["home_pred"].to_numpy()
    away_pred = game_view["away_pred"].to_numpy()
    home_actual = game_view["home_actual"].to_numpy()
    away_actual = game_view["away_actual"].to_numpy()

    metrics = {
        "mae": mean_absolute_error(y_true_f, y_pred_f),
        "rmse": mean_squared_error(y_true_f, y_pred_f, squared=False),
        "r2": r2_score(y_true_f, y_pred_f) if len(y_true_f) > 1 else float("nan"),
        "spread_mae": mean_absolute_error(home_actual - away_actual, home_pred - away_pred),
        "spread_abs_mae": float(np.mean(np.abs((home_actual - away_actual) - (home_pred - away_pred)))),
        "spread_rmse": mean_squared_error(home_actual - away_actual, home_pred - away_pred, squared=False),
        "total_mae": mean_absolute_error(home_actual + away_actual, home_pred + away_pred),
        "total_rmse": mean_squared_error(home_actual + away_actual, home_pred + away_pred, squared=False),
    }

    logger.info(
        "XGBoost performance metrics: %s",
        {"metrics": metrics, "num_games": int(game_view.height), "num_rows": int(df.height)},
        extra={"metrics": metrics, "num_games": int(game_view.height), "num_rows": int(df.height)},
    )
    return metrics


def compute_spreads(df: pl.DataFrame, preds: pl.Series, is_home_col: str = "is_home") -> pl.DataFrame:
    """
    Compute predicted/actual spread and total from team-level predictions.
    """
    required_cols = {"game_id", "final_home_score", "final_away_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ple.ColumnNotFoundError(f"Missing required columns for spread computation: {sorted(missing)}")

    if is_home_col not in df.columns:
        if "home_away_flag" in df.columns:
            df = df.with_columns((pl.col("home_away_flag") == "home").cast(pl.Int8).alias(is_home_col))
        else:
            raise ple.ColumnNotFoundError(f"Missing home indicator column: {is_home_col}")

    df_with_preds = df.with_columns(pl.Series("predicted_points", preds))
    game_spreads = (
        df_with_preds.group_by("game_id")
        .agg(
            [
                pl.col("predicted_points").filter(pl.col(is_home_col) == 1).first().alias("pred_home"),
                pl.col("predicted_points").filter(pl.col(is_home_col) == 0).first().alias("pred_away"),
                pl.col("final_home_score").max().alias("final_home_score"),
                pl.col("final_away_score").max().alias("final_away_score"),
            ]
        )
        .with_columns(
            [
                (pl.col("pred_home") - pl.col("pred_away")).alias("predicted_spread"),
                (pl.col("final_home_score") - pl.col("final_away_score")).alias("actual_spread"),
                (pl.col("final_home_score") + pl.col("final_away_score")).alias("actual_total"),
            ]
        )
    )

    return df_with_preds.join(game_spreads, on="game_id", how="left")
