""" Script for computing features for sports betting ML models """

import logging
import click
import polars as pl
import pandera.polars as pa

from src.utils.utils import (
    save_data,
    load_data,
    polars_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d"
)

logger = logging.getLogger(__name__)


# -----------------------------
#  Pandera Schema Definitions
# -----------------------------
class RawPlayByPlaySchema(pa.DataFrameModel):
    """
    Schema for raw play-by-play data used in feature engineering.
    """
    # --- Identifiers ---
    game_id: str
    posteam: str

    # --- Core play metrics ---
    epa: pl.Float64 = pa.Field(nullable=False, in_range={"min_value": -10, "max_value": 10})
    success: pl.Float64 = pa.Field(nullable=False, in_range={"min_value": 0, "max_value": 1})
    yards_gained: pl.Float64 = pa.Field(nullable=True)

    # --- Drive outcomes ---
    drive_ended_with_score: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    drive_yards_penalized: pl.Float64 = pa.Field(nullable=True)
    drive_time_of_possession: str = pa.Field(nullable=True)

    # --- Turnovers & Penalties ---
    interception: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    fumble_lost: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    penalty_yards: pl.Float64 = pa.Field(nullable=True)

    # --- Downs ---
    third_down_converted: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    third_down_failed: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    fourth_down_converted: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    fourth_down_failed: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})

    # --- Filters & subsets ---
    pass_attempt: pl.Float64 = pa.Field(nullable=True)
    rush_attempt: pl.Float64 = pa.Field(nullable=True)
    yardline_100: pl.Float64 = pa.Field(nullable=True)

    # --- Passing & rushing ---
    passing_yards: pl.Float64 = pa.Field(nullable=True)
    rushing_yards: pl.Float64 = pa.Field(nullable=True)
    complete_pass: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})

    # --- Red zone ---
    touchdown: pl.Float64 = pa.Field(nullable=True)

    # --- Situational ---
    posteam_type: str = pa.Field(nullable=True, isin=["home", "away"])
    roof: str = pa.Field(nullable=True)
    surface: str = pa.Field(nullable=True)
    temp: pl.Int32 = pa.Field(nullable=True, in_range={"min_value": -20, "max_value": 130})
    wind: pl.Int32 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 60})

    # --- Scores ---
    home_score: pl.Int32 = pa.Field(nullable=True)
    away_score: pl.Int32 = pa.Field(nullable=True)


class TeamGameFeaturesSchema(pa.DataFrameModel):
    """ Schema for team-game level features computed from play-by-play data. """
    game_id: str
    posteam: str

    # Scoring/Efficiency
    avg_epa: pl.Float64
    total_epa: pl.Float64
    success_rate: pl.Float64
    pct_plays_over_15yds: pl.Float64
    pct_drives_scored: pl.Float64
    avg_drive_yards_penalized: pl.Float64

    # Turnovers & penalties
    turnover_count: pl.Float64
    penalty_yards_total: pl.Float64

    # Down success
    third_down_success_rate: pl.Float64

    # Situational
    home_away_flag: str
    roof_type: str
    surface: str
    temp: pl.Int32
    wind: pl.Int32

    # Passing
    pass_epa_mean: pl.Float64
    pass_yards_avg: pl.Float64
    comp_pct: pl.Float64

    # Rushing
    rush_epa_mean: pl.Float64
    rush_yards_avg: pl.Float64

    # Red zone
    td_rate_in_redzone: pl.Float64

    final_home_score: pl.Int32
    final_away_score: pl.Int32


# -----------------------------
#  Feature Engineering Logic
# -----------------------------
def compute_team_game_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates play-by-play data to compute team-level game features for sports analytics.

    This function groups the input DataFrame by "game_id" and "posteam" and computes a variety of
    statistical features, including means, sums, first values, maximums, and conditional aggregations
    for passing, rushing, red zone, and situational metrics. Custom features such as percentage of
    plays over 15 yards, turnover count, and third down success rate are also calculated.

    Parameters:
        df (pl.DataFrame): The raw play-by-play data as a Polars DataFrame.

    Returns:
        pl.DataFrame: A DataFrame with one row per team per game, containing aggregated features.
    """
    mean_cols = {
        "epa": "avg_epa",
        "success": "success_rate",
        "drive_ended_with_score": "pct_drives_scored",
        "drive_yards_penalized": "avg_drive_yards_penalized",
    }
    mean_aggs = [pl.col(c).mean().alias(a) for c, a in mean_cols.items()]

    sum_cols = {"epa": "total_epa", "penalty_yards": "penalty_yards_total"}
    sum_aggs = [pl.col(c).sum().alias(a) for c, a in sum_cols.items()]

    first_cols = {
        "posteam_type": "home_away_flag",
        "roof": "roof_type",
        "surface": "surface",
        "temp": "temp",
        "wind": "wind",
    }
    first_aggs = [pl.col(c).first().alias(a) for c, a in first_cols.items()]

    max_cols = {"home_score": "final_home_score", "away_score": "final_away_score"}
    max_aggs = [pl.col(c).max().alias(a) for c, a in max_cols.items()]

    filters = [
        (
            pl.col("pass_attempt") == 1,
            [("epa", "pass_epa_mean"), ("passing_yards", "pass_yards_avg"), ("complete_pass", "comp_pct")],
        ),
        (
            pl.col("rush_attempt") == 1,
            [("epa", "rush_epa_mean"), ("rushing_yards", "rush_yards_avg")],
        ),
        (
            pl.col("yardline_100") <= 20,
            [("touchdown", "td_rate_in_redzone")],
        ),
    ]

    conditional_aggs = [
        pl.when(condition).then(pl.col(col)).mean().alias(alias)
        for condition, columns in filters
        for col, alias in columns
    ]

    custom_aggs = [
        (pl.col("yards_gained") > 15).mean().alias("pct_plays_over_15yds"),
        (pl.col("interception") + pl.col("fumble_lost")).sum().alias("turnover_count"),
        (
            (
                pl.col("third_down_converted") + pl.col("fourth_down_converted")
            ).sum()
            / (
                pl.col("third_down_converted")
                + pl.col("third_down_failed")
                + pl.col("fourth_down_converted")
                + pl.col("fourth_down_failed")
            ).sum()
        ).alias("third_down_success_rate"),
    ]

    aggs = mean_aggs + sum_aggs + first_aggs + max_aggs + conditional_aggs + custom_aggs

    return df.group_by(["game_id", "posteam"]).agg(aggs)


# -----------------------------
#   Main Validation & Saving
# -----------------------------
def run_feature_engineering(
    local: bool,
    year: int,
    input_path: str | None,
    output_path: str | None,
    filename: str | None,
    bucket: str | None,
    input_key_template: str,
    output_key_template: str,
):
    """
    Runs the end-to-end feature engineering pipeline for sports betting ML models.

    This function loads raw play-by-play data (from a local path or S3, depending on the `local` flag),
    validates it against the expected schema, computes team-level game features, validates the output,
    and saves the processed features either locally or to S3.

    Parameters:
        local (bool): If True, load and save data locally; otherwise, use S3.
        year (int): The season year to process.
        input_path (str | None): Local input file path (required if local=True).
        output_path (str | None): Local output directory (required if local=True).
        filename (str | None): Filename for input/output data.
        bucket (str | None): S3 bucket name (required if local=False).
        input_key_template (str): S3 key template for raw data.
        output_key_template (str): S3 key template for processed output.

    Raises:
        ValueError: If required paths are not provided for the selected mode.

    Returns:
        None
    """

    logger.info(f"Starting feature engineering for year={year}, local={local}")

    # --- Load data ---
    if local:
        if not input_path:
            raise ValueError("You must provide --input-path when using --local")
        logger.info(f"Loading local data from: {input_path}")
        raw = load_data(local=True, filename=filename, local_path=input_path)
    else:
        key = input_key_template.format(year=year)
        logger.info(f"Loading S3 data: s3://{bucket}/{key}")
        raw = load_data(filename=filename, bucket=bucket, key=key, local=False)

    # --- Schema validate ---
    RawPlayByPlaySchema.validate(raw.lazy()).collect()

    # --- Compute features ---
    processed = compute_team_game_features(raw)

    # --- Validate output ---
    TeamGameFeaturesSchema.validate(processed.lazy()).collect()

    logger.info("Feature engineering completed successfully.")
    logger.info(f"Processed dataset info:\n{polars_info(processed)}")

    # --- Save output ---
    if local:
        if not output_path:
            raise ValueError("You must provide --output-path when using --local")
        filename = f"features_{year}.parquet"
        logger.info(f"Saving processed dataset locally: {filename}")
        save_data(data=processed, filename=filename, local=True, local_path=output_path)
    else:
        key = output_key_template.format(year=year)
        logger.info(f"Saving processed dataset to S3: s3://{bucket}/{key}")
        save_data(processed, filename=filename, bucket=bucket, key=key, local=False)

# -----------------------------
#             CLICK CLI
# -----------------------------

@click.command()
@click.option("--local", is_flag=True, help="Run locally instead of using S3")
@click.option("--year", default=2020, type=int, show_default=True, help="Season year to process")
@click.option("--input-path", type=str, default=None, help="Local input file path")
@click.option("--output-path", type=str, default=None, help="Local output directory")
@click.option("--bucket", type=str, default="sports-betting-ml", help="S3 bucket")
@click.option("--filename", type=str, default="2020_pbp_data.parquet", help="local filename")
@click.option(
    "--input-key-template",
    default="raw-data",
    show_default=True,
    help="S3 key template for raw data",
)
@click.option(
    "--output-key-template",
    default="processed/features_{year}.parquet",
    show_default=True,
    help="S3 key template for processed output",
)
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
def run(local, year, input_path, output_path, filename, bucket, input_key_template, output_key_template, debug):
    """Run the end-to-end feature engineering pipeline."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    run_feature_engineering(
        local=local,
        year=year,
        input_path=input_path,
        output_path=output_path,
        filename=filename,
        bucket=bucket,
        input_key_template=input_key_template,
        output_key_template=output_key_template,
    )


if __name__ == "__main__":
    run()
