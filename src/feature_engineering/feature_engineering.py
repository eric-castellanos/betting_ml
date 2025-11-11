import logging

import polars as pl
import pandera.polars as pa

from src.utils.utils import save_data_s3, load_data_s3, polars_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(level)s - %(message)s - %(lineno)d")

logger = logging.getLogger(__name__)

class RawPlayByPlaySchema(pa.DataFrameModel):
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
    drive_time_of_possession: str = pa.Field(nullable=True)  # often string like "5:32"

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
    pass_attempt: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    rush_attempt: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})
    yardline_100: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 100})

    # --- Passing & rushing ---
    passing_yards: pl.Float64 = pa.Field(nullable=True)
    rushing_yards: pl.Float64 = pa.Field(nullable=True)
    complete_pass: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})

    # --- Red zone ---
    touchdown: pl.Float64 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 1})

    # --- Situational ---
    posteam_type: str = pa.Field(nullable=True, isin=["home", "away"])
    roof: str = pa.Field(nullable=True)
    surface: str = pa.Field(nullable=True)
    temp: pl.Int32 = pa.Field(nullable=True, in_range={"min_value": -20, "max_value": 130})
    wind: pl.Int32 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 60})

    # --- Scores ---
    home_score: pl.Int32 = pa.Field(nullable=True, in_range={"min_value" : 0, "max_value" : 100})
    away_score: pl.Int32 = pa.Field(nullable=True, in_range={"min_value" : 0, "max_value" : 100})

class TeamGameFeaturesSchema(pa.DataFrameModel):
    game_id: str
    posteam: str

    # --- Scoring & efficiency ---
    avg_epa: pl.Float64
    total_epa: pl.Float64
    success_rate: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value": 1})

    # --- Explosiveness ---
    pct_plays_over_15yds: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value": 1})

    # --- Drive outcomes ---
    pct_drives_scored: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value": 1})
    avg_drive_yards_penalized: pl.Float64

    # --- Turnovers & penalties ---
    turnover_count: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value" : float("inf")})
    penalty_yards_total: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value" : float("inf")})

    # --- Down success ---
    third_down_success_rate: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value": 1})

    # --- Situational ---
    home_away_flag: str = pa.Field(isin=["home", "away"])
    roof_type: str = pa.Field(nullable=True)
    surface: str = pa.Field(nullable=True)
    temp: pl.Int32 = pa.Field(nullable=True, in_range={"min_value": -20, "max_value": 130})
    wind: pl.Int32 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 60})

    # --- Passing ---
    pass_epa_mean: pl.Float64
    pass_yards_avg: pl.Float64
    comp_pct: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value": 1})

    # --- Rushing ---
    rush_epa_mean: pl.Float64
    rush_yards_avg: pl.Float64

    # --- Red zone ---
    td_rate_in_redzone: pl.Float64 = pa.Field(in_range={"min_value": 0, "max_value": 1})

    # --- Scores ---
    final_home_score: pl.Int32 = pa.Field(nullable=True, in_range={"min_value" : 0, "max_value" : 100})
    final_away_score: pl.Int32 = pa.Field(nullable=True, in_range={"min_value" : 0, "max_value" : 100})


import polars as pl


def compute_team_game_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes per-team, per-game feature aggregates efficiently.
    Dynamically builds aggregations using lists for both standard
    and conditional (filtered) aggregations.
    """

    # --- Simple mean aggregations ---
    mean_cols = {
        "epa": "avg_epa",
        "success": "success_rate",
        "drive_ended_with_score": "pct_drives_scored",
        "drive_yards_penalized": "avg_drive_yards_penalized",
    }
    mean_aggs = [pl.col(c).mean().alias(a) for c, a in mean_cols.items()]

    # --- Simple sum aggregations ---
    sum_cols = {"epa": "total_epa", "penalty_yards": "penalty_yards_total"}
    sum_aggs = [pl.col(c).sum().alias(a) for c, a in sum_cols.items()]

    # --- Simple first aggregations (categorical/contextual) ---
    first_cols = {
        "posteam_type": "home_away_flag",
        "roof": "roof_type",
        "surface": "surface",
        "temp": "temp",
        "wind": "wind",
    }
    first_aggs = [pl.col(c).first().alias(a) for c, a in first_cols.items()]

    max_cols = {
        "home_score" : "final_home_score",
        "away_score" : "final_away_score"
    }
    max_aggs = [pl.col(c).max().alias(a) for c,a in max_cols.items()]

    # --- Conditional filters and their associated columns ---
    # Each tuple: (filter_expression, [(column_name, alias_name), ...])
    filters = [
        # Passing plays
        (
            pl.col("pass_attempt") == 1,
            [
                ("epa", "pass_epa_mean"),
                ("passing_yards", "pass_yards_avg"),
                ("complete_pass", "comp_pct"),
            ],
        ),
        # Rushing plays
        (
            pl.col("rush_attempt") == 1,
            [
                ("epa", "rush_epa_mean"),
                ("rushing_yards", "rush_yards_avg"),
            ],
        ),
        # Red zone plays
        (
            pl.col("yardline_100") <= 20,
            [
                ("touchdown", "td_rate_in_redzone"),
            ],
        ),
    ]

    # Build conditional aggregations dynamically
    conditional_aggs = [
        pl.when(condition).then(pl.col(col)).mean().alias(alias)
        for condition, columns in filters
        for col, alias in columns
    ]

    # --- Custom expressions that don’t fit the pattern ---
    custom_aggs = [
        # Explosiveness
        (pl.col("yards_gained") > 15).mean().alias("pct_plays_over_15yds"),

        # Turnovers
        (pl.col("interception") + pl.col("fumble_lost")).sum().alias("turnover_count"),

        # Down success rate
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

    # --- Combine all aggregations ---
    aggs = mean_aggs + sum_aggs + first_aggs + max_aggs + conditional_aggs + custom_aggs

    # --- Single efficient groupby ---
    result = df.group_by(["game_id", "posteam"]).agg(aggs)

    return result

def validate_df(raw : pl.DataFrame) -> pl.DataFrame:
    try:
        processed = compute_team_game_features(raw)
    except pa.errors.SchemaError as e:
        logger.error(f"Something is wrong with the raw dataframe's schema: {e}")
    except (pl.exceptions.ColumnNotFoundError, KeyError) as e:
        logger.error(f"Column not found: {e}")
    except ValueError as e:
        logger.error(f"Trying to perform invalid computation: {e}")
    finally:
        logger.info(f"Data transformations succeeded. Summary statisitcs for dataframe: {processed.describe()}")
        logger.info(f"Info for the feature engineered dataset: {polars_info(processed)}")
        logger.info("Feature engineering process complete.")

    return processed

if __name__ == "__main__":
    ## Will use 2020 dataset for development only for now ##
    raw = load_data_s3("sports-betting-ml", "raw-data/2020_pbp_data.parquet")
    #import pdb; pdb.set_trace()
    RawPlayByPlaySchema.validate(raw.lazy()).collect()
    processed = validate_df(raw)
    TeamGameFeaturesSchema.validate(processed.lazy()).collect()