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


def compute_team_game_features(df: pl.DataFrame) -> pl.DataFrame:
    # Pre-filter for subsets you need just once
    pass_df = df.filter(pl.col("pass_attempt") == 1)
    rush_df = df.filter(pl.col("rush_attempt") == 1)
    redzone_df = df.filter(pl.col("yardline_100") <= 20)

    # Base groupby
    grouped = df.group_by(["game_id", "posteam"]).agg([
        # --- Scoring & Efficiency ---
        pl.col("epa").mean().alias("avg_epa"),
        pl.col("epa").sum().alias("total_epa"),
        pl.col("success").mean().alias("success_rate"),

        # --- Explosiveness ---
        (pl.col("yards_gained") > 15).mean().alias("pct_plays_over_15yds"),

        # --- Drive outcomes ---
        pl.col("drive_ended_with_score").mean().alias("pct_drives_scored"),
        pl.col("drive_yards_penalized").mean().alias("avg_drive_yards_penalized"),

        # --- Turnovers & Penalties ---
        (pl.col("interception") + pl.col("fumble_lost")).sum().alias("turnover_count"),
        pl.col("penalty_yards").sum().alias("penalty_yards_total"),

        # --- Down success ---
        (
            (
                pl.col("third_down_converted") + pl.col("fourth_down_converted")
            ).sum() /
            (
                pl.col("third_down_converted") + pl.col("third_down_failed") +
                pl.col("fourth_down_converted") + pl.col("fourth_down_failed")
            ).sum()
        ).alias("third_down_success_rate"),

        # --- Situational ---
        pl.col("posteam_type").first().alias("home_away_flag"),
        pl.col("roof").first().alias("roof_type"),
        pl.col("surface").first().alias("surface"),
        pl.col("temp").first().alias("temp"),
        pl.col("wind").first().alias("wind"),
    ])

    # --- Merge in specialized subsets (passing, rushing, redzone) ---
    pass_stats = pass_df.group_by(["game_id", "posteam"]).agg([
        pl.col("epa").mean().alias("pass_epa_mean"),
        pl.col("passing_yards").mean().alias("pass_yards_avg"),
        pl.col("complete_pass").mean().alias("comp_pct"),
    ])

    rush_stats = rush_df.group_by(["game_id", "posteam"]).agg([
        pl.col("epa").mean().alias("rush_epa_mean"),
        pl.col("rushing_yards").mean().alias("rush_yards_avg"),
    ])

    redzone_stats = redzone_df.group_by(["game_id", "posteam"]).agg([
        pl.col("touchdown").mean().alias("td_rate_in_redzone"),
    ])

    # --- Merge them all together ---
    result = (
        grouped
        .join(pass_stats, on=["game_id", "posteam"], how="left")
        .join(rush_stats, on=["game_id", "posteam"], how="left")
        .join(redzone_stats, on=["game_id", "posteam"], how="left")
    )

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