"""
Helpers for fetching and preparing feature data for the xgboost model.
"""

from typing import Optional
import logging
import sys

import click
import polars as pl

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
        print(winsorized.head())
    except Exception:
        logger.exception("Failed to load or winsorize features")
        sys.exit(1)


if __name__ == "__main__":
    main()
